# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import brainstate


class TestForLoop(unittest.TestCase):
    def test_for_loop(self):
        a = brainstate.ShortTermState(0.)
        b = brainstate.ShortTermState(0.)

        def f(i):
            a.value += (1 + b.value)
            return a.value

        n_iter = 10
        ops = np.arange(n_iter)
        r = brainstate.transform.for_loop(f, ops)

        print(a)
        print(b)
        self.assertTrue(a.value == n_iter)
        self.assertTrue(jnp.allclose(r, ops + 1))

    def test_checkpointed_for_loop(self):
        a = brainstate.ShortTermState(0.)
        b = brainstate.ShortTermState(0.)

        def f(i):
            a.value += (1 + b.value)
            return a.value

        n_iter = 18
        ops = jnp.arange(n_iter)
        r = brainstate.transform.checkpointed_for_loop(f, ops, base=2, pbar=brainstate.transform.ProgressBar())

        print(a)
        print(b)
        print(r)
        self.assertTrue(a.value == n_iter)
        self.assertTrue(jnp.allclose(r, ops + 1))


class TestCheckpointedScanSkipPath(unittest.TestCase):
    """A length that is not a power of ``base`` exercises the counter-bump
    skip path of ``_bounded_while_loop`` (regression guard for the H2 fix)."""

    def test_carry_collect_and_state_values(self):
        st = brainstate.ShortTermState(jnp.asarray(0.0))

        def f(carry, x):
            st.value = st.value + 1.0
            return carry + x, carry * 2.0

        xs = jnp.arange(5.0)
        carry, ys = brainstate.transform.checkpointed_scan(f, init=0.0, xs=xs, base=2)
        self.assertEqual(float(carry), 10.0)
        self.assertTrue(bool(jnp.allclose(ys, jnp.asarray([0., 0., 2., 6., 12.]))))
        self.assertEqual(float(st.value), 5.0)


class TestScanValidation(unittest.TestCase):
    """Tests for scan() input-validation branches."""

    def test_scan_non_callable_raises(self):
        """Passing a non-callable as f should raise TypeError."""
        with self.assertRaises(TypeError):
            brainstate.transform.scan(42, 0.0, jnp.arange(3.0))

    def test_scan_no_array_leaf_raises(self):
        """xs leaves without .shape should raise ValueError."""
        with self.assertRaises(ValueError):
            brainstate.transform.scan(lambda c, x: (c, x), 0.0, [1, 2, 3])

    def test_scan_mismatched_lengths_raises(self):
        """xs arrays with different leading lengths should raise ValueError."""
        xs_a = jnp.arange(3.0)
        xs_b = jnp.arange(4.0)
        with self.assertRaises(ValueError):
            brainstate.transform.scan(lambda c, x: (c, x[0]), 0.0, (xs_a, xs_b))

    def test_scan_no_xs_no_length_raises(self):
        """Calling scan with xs=None and no length should raise ValueError."""
        with self.assertRaises(ValueError):
            brainstate.transform.scan(lambda c, x: (c, x), 0.0, [])

    def test_scan_length_disagrees_with_xs_raises(self):
        """Explicit length that disagrees with xs leading axis should raise ValueError."""
        xs = jnp.arange(5.0)
        with self.assertRaises(ValueError):
            brainstate.transform.scan(lambda c, x: (c, x), 0.0, xs, length=3)

    def test_scan_pbar_invalid_type_raises(self):
        """Passing an unsupported pbar type should raise TypeError."""
        xs = jnp.arange(4.0)
        with self.assertRaises(TypeError):
            brainstate.transform.scan(lambda c, x: (c, x), 0.0, xs, pbar="invalid")

    def test_scan_length_matches_xs(self):
        """Explicit length matching xs does not raise and runs correctly."""
        xs = jnp.arange(4.0)
        carry, ys = brainstate.transform.scan(lambda c, x: (c + x, x), 0.0, xs, length=4)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))
        self.assertTrue(jnp.allclose(ys, xs))


class TestScanReverse(unittest.TestCase):
    """Tests for scan() reverse and unroll options."""

    def test_scan_reverse_true(self):
        """scan with reverse=True threads carry from right; total carry equals forward."""
        def f(carry, x):
            return carry + x, x * 2.0

        xs = jnp.arange(1.0, 5.0)
        carry_fwd, ys_fwd = brainstate.transform.scan(f, 0.0, xs, reverse=False)
        carry_rev, ys_rev = brainstate.transform.scan(f, 0.0, xs, reverse=True)
        # total carry should be the same (same elements summed, order doesn't matter)
        self.assertTrue(jnp.allclose(carry_fwd, carry_rev))
        # outputs for f(carry,x)=x*2 are position-aligned and identical regardless of direction
        self.assertTrue(jnp.allclose(ys_rev, ys_fwd))

    def test_scan_unroll_2(self):
        """scan with unroll=2 should produce same result as unroll=1."""
        def f(carry, x):
            return carry + x, x

        xs = jnp.arange(6.0)
        carry1, ys1 = brainstate.transform.scan(f, 0.0, xs, unroll=1)
        carry2, ys2 = brainstate.transform.scan(f, 0.0, xs, unroll=2)
        self.assertTrue(jnp.allclose(carry1, carry2))
        self.assertTrue(jnp.allclose(ys1, ys2))

    def test_scan_unroll_true(self):
        """scan with unroll=True (full unroll) should produce same result as unroll=1."""
        def f(carry, x):
            return carry * 2.0 + x, carry + x

        xs = jnp.arange(1.0, 5.0)
        carry1, ys1 = brainstate.transform.scan(f, 1.0, xs, unroll=1)
        carry2, ys2 = brainstate.transform.scan(f, 1.0, xs, unroll=True)
        self.assertTrue(jnp.allclose(carry1, carry2))
        self.assertTrue(jnp.allclose(ys1, ys2))


class TestScanPbar(unittest.TestCase):
    """Tests for scan() with ProgressBar and integer pbar."""

    def test_scan_pbar_progressbar_instance(self):
        """scan with a ProgressBar instance runs without error and returns correct result."""
        def f(carry, x):
            return carry + x, x

        xs = jnp.arange(5.0)
        pbar = brainstate.transform.ProgressBar(freq=2)
        carry, ys = brainstate.transform.scan(f, 0.0, xs, pbar=pbar)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))
        self.assertTrue(jnp.allclose(ys, xs))

    def test_scan_pbar_integer(self):
        """scan with an integer pbar creates a ProgressBar internally and returns correct result."""
        def f(carry, x):
            return carry + x, x * 2.0

        xs = jnp.arange(4.0)
        carry, ys = brainstate.transform.scan(f, 0.0, xs, pbar=2)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))
        self.assertTrue(jnp.allclose(ys, xs * 2.0))


class TestScanDisableJit(unittest.TestCase):
    """Tests for scan() in jax.disable_jit() mode (non-JIT paths)."""

    def test_scan_disable_jit_forward(self):
        """scan in disable_jit mode runs the plain Python loop and returns correct result."""
        def f(carry, x):
            return carry + x, x

        xs = jnp.arange(1.0, 5.0)
        with jax.disable_jit():
            carry, ys = brainstate.transform.scan(f, 0.0, xs)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))
        self.assertTrue(jnp.allclose(ys, xs))

    def test_scan_disable_jit_reverse(self):
        """scan in disable_jit mode with reverse=True iterates in reverse order."""
        order = []

        def f(carry, x):
            order.append(float(x))
            return carry + x, x

        xs = jnp.arange(1.0, 5.0)
        with jax.disable_jit():
            carry, ys = brainstate.transform.scan(f, 0.0, xs, reverse=True)
        # iteration order should be reversed (4, 3, 2, 1)
        self.assertEqual(order, [4.0, 3.0, 2.0, 1.0])
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))

    def test_scan_disable_jit_zero_length_raises(self):
        """scan in disable_jit mode with length=0 raises ValueError."""
        def f(carry, x):
            return carry, x

        with jax.disable_jit():
            with self.assertRaises(ValueError):
                # Must use xs=None with length=0, but that hits a different error first.
                # Instead use an empty array to trigger the zero-length path.
                brainstate.transform.scan(f, 0.0, jnp.zeros((0,)))

    def test_scan_disable_jit_pbar_progressbar(self):
        """scan in disable_jit mode with ProgressBar pbar unwraps carry correctly."""
        def f(carry, x):
            return carry + x, x

        xs = jnp.arange(1.0, 4.0)
        pbar = brainstate.transform.ProgressBar(freq=1)
        with jax.disable_jit():
            carry, ys = brainstate.transform.scan(f, 0.0, xs, pbar=pbar)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))

    def test_scan_disable_jit_pbar_integer(self):
        """scan in disable_jit mode with integer pbar works correctly."""
        def f(carry, x):
            return carry + 1.0, x

        xs = jnp.arange(3.0)
        with jax.disable_jit():
            carry, ys = brainstate.transform.scan(f, 0.0, xs, pbar=1)
        self.assertTrue(jnp.allclose(carry, 3.0))


class TestScanStateAccumulation(unittest.TestCase):
    """Tests for scan() with State side-effects."""

    def test_scan_state_write_accumulates(self):
        """State written each step should hold the final value after scan completes."""
        acc = brainstate.ShortTermState(0.0)

        def f(carry, x):
            acc.value = acc.value + x
            return carry + x, x

        xs = jnp.arange(1.0, 5.0)
        carry, ys = brainstate.transform.scan(f, 0.0, xs)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))
        self.assertTrue(jnp.allclose(acc.value, jnp.sum(xs)))

    def test_scan_pytree_carry(self):
        """scan with a dict carry accumulates all leaf values correctly."""
        def f(carry, x):
            new = {'a': carry['a'] + x, 'b': carry['b'] - x}
            return new, carry['a']

        xs = jnp.arange(1.0, 4.0)
        init = {'a': 0.0, 'b': 10.0}
        carry, ys = brainstate.transform.scan(f, init, xs)
        self.assertTrue(jnp.allclose(carry['a'], jnp.sum(xs)))
        self.assertTrue(jnp.allclose(carry['b'], 10.0 - jnp.sum(xs)))


class TestCheckpointedScanValidation(unittest.TestCase):
    """Tests for checkpointed_scan() input-validation branches."""

    def test_checkpointed_scan_non_callable_raises(self):
        """Passing a non-callable as f should raise TypeError."""
        with self.assertRaises(TypeError):
            brainstate.transform.checkpointed_scan(99, 0.0, jnp.arange(4.0))

    def test_checkpointed_scan_no_array_leaf_raises(self):
        """xs leaves without .shape should raise ValueError."""
        with self.assertRaises(ValueError):
            brainstate.transform.checkpointed_scan(
                lambda c, x: (c, x), 0.0, [1, 2, 3]
            )

    def test_checkpointed_scan_mismatched_lengths_raises(self):
        """xs arrays with different leading lengths should raise ValueError."""
        xs_a = jnp.arange(3.0)
        xs_b = jnp.arange(5.0)
        with self.assertRaises(ValueError):
            brainstate.transform.checkpointed_scan(
                lambda c, x: (c, x[0]), 0.0, (xs_a, xs_b)
            )

    def test_checkpointed_scan_no_xs_no_length_raises(self):
        """Empty xs with no length should raise ValueError."""
        with self.assertRaises(ValueError):
            brainstate.transform.checkpointed_scan(lambda c, x: (c, x), 0.0, [])

    def test_checkpointed_scan_length_disagrees_raises(self):
        """Explicit length that disagrees with xs should raise ValueError."""
        xs = jnp.arange(5.0)
        with self.assertRaises(ValueError):
            brainstate.transform.checkpointed_scan(
                lambda c, x: (c, x), 0.0, xs, length=3
            )

    def test_checkpointed_scan_length_matches_xs(self):
        """Explicit length matching xs does not raise and runs correctly."""
        xs = jnp.arange(4.0)
        carry, ys = brainstate.transform.checkpointed_scan(
            lambda c, x: (c + x, x), 0.0, xs, length=4
        )
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs), atol=1e-5))


class TestCheckpointedScanBasic(unittest.TestCase):
    """Tests for basic correctness of checkpointed_scan()."""

    def test_checkpointed_scan_basic(self):
        """checkpointed_scan produces same carry and ys as plain scan."""
        def f(carry, x):
            return carry + x, x * 2.0

        xs = jnp.arange(1.0, 9.0)  # 8 elements (power of 2 for base=2)
        carry_ref, ys_ref = brainstate.transform.scan(f, 0.0, xs)
        carry_cp, ys_cp = brainstate.transform.checkpointed_scan(f, 0.0, xs, base=2)
        self.assertTrue(jnp.allclose(carry_ref, carry_cp, atol=1e-5))
        self.assertTrue(jnp.allclose(ys_ref, ys_cp, atol=1e-5))

    def test_checkpointed_scan_pbar_progressbar(self):
        """checkpointed_scan with a ProgressBar instance runs without error."""
        def f(carry, x):
            return carry + x, x

        xs = jnp.arange(1.0, 5.0)
        pbar = brainstate.transform.ProgressBar(freq=1)
        carry, ys = brainstate.transform.checkpointed_scan(f, 0.0, xs, pbar=pbar)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))

    def test_checkpointed_scan_pbar_integer(self):
        """checkpointed_scan with integer pbar creates ProgressBar internally."""
        def f(carry, x):
            return carry + x, x

        xs = jnp.arange(4.0)
        carry, ys = brainstate.transform.checkpointed_scan(f, 0.0, xs, pbar=2)
        self.assertTrue(jnp.allclose(carry, jnp.sum(xs)))

    def test_checkpointed_scan_state_write(self):
        """checkpointed_scan accumulates State writes across iterations."""
        acc = brainstate.ShortTermState(0.0)

        def f(carry, x):
            acc.value = acc.value + x
            return carry + x, x

        xs = jnp.arange(1.0, 5.0)
        carry, ys = brainstate.transform.checkpointed_scan(f, 0.0, xs, base=2)
        self.assertTrue(jnp.allclose(acc.value, jnp.sum(xs), atol=1e-5))


class TestForLoopEdgeCases(unittest.TestCase):
    """Edge-case tests for for_loop()."""

    def test_for_loop_reverse(self):
        """for_loop with reverse=True gives the same element-wise output as forward for pure functions."""
        def f(x):
            return x * 2.0

        xs = jnp.arange(1.0, 5.0)
        ys_rev = brainstate.transform.for_loop(f, xs, reverse=True)
        ys_fwd = brainstate.transform.for_loop(f, xs, reverse=False)
        # For a stateless f, outputs are position-aligned and identical
        self.assertTrue(jnp.allclose(ys_rev, ys_fwd))

    def test_for_loop_unroll_2(self):
        """for_loop with unroll=2 produces same result as unroll=1."""
        def f(x):
            return x * 3.0

        xs = jnp.arange(6.0)
        ys1 = brainstate.transform.for_loop(f, xs, unroll=1)
        ys2 = brainstate.transform.for_loop(f, xs, unroll=2)
        self.assertTrue(jnp.allclose(ys1, ys2))

    def test_for_loop_multiple_xs(self):
        """for_loop with multiple positional xs arrays computes element-wise."""
        def f(a, b):
            return a + b

        xs_a = jnp.arange(1.0, 5.0)
        xs_b = jnp.ones(4) * 10.0
        ys = brainstate.transform.for_loop(f, xs_a, xs_b)
        self.assertTrue(jnp.allclose(ys, xs_a + 10.0))

    def test_for_loop_pbar_integer(self):
        """for_loop with integer pbar runs without error."""
        def f(x):
            return x + 1.0

        xs = jnp.arange(4.0)
        ys = brainstate.transform.for_loop(f, xs, pbar=2)
        self.assertTrue(jnp.allclose(ys, xs + 1.0))

    def test_for_loop_pbar_progressbar(self):
        """for_loop with ProgressBar instance runs without error."""
        def f(x):
            return x * 2.0

        xs = jnp.arange(4.0)
        pbar = brainstate.transform.ProgressBar(freq=1)
        ys = brainstate.transform.for_loop(f, xs, pbar=pbar)
        self.assertTrue(jnp.allclose(ys, xs * 2.0))


class TestCheckpointedForLoop(unittest.TestCase):
    """Tests for checkpointed_for_loop()."""

    def test_checkpointed_for_loop_basic(self):
        """checkpointed_for_loop output matches for_loop output."""
        def f(x):
            return x * 2.0

        xs = jnp.arange(1.0, 5.0)
        ys_ref = brainstate.transform.for_loop(f, xs)
        ys_cp = brainstate.transform.checkpointed_for_loop(f, xs)
        self.assertTrue(jnp.allclose(ys_ref, ys_cp, atol=1e-5))

    def test_checkpointed_for_loop_multiple_xs(self):
        """checkpointed_for_loop with multiple xs args computes element-wise."""
        def f(a, b):
            return a * b

        xs_a = jnp.arange(1.0, 5.0)
        xs_b = jnp.arange(2.0, 6.0)
        ys_cp = brainstate.transform.checkpointed_for_loop(f, xs_a, xs_b)
        ys_ref = brainstate.transform.for_loop(f, xs_a, xs_b)
        self.assertTrue(jnp.allclose(ys_ref, ys_cp, atol=1e-5))

    def test_checkpointed_for_loop_pbar_integer(self):
        """checkpointed_for_loop with integer pbar runs without error."""
        def f(x):
            return x + 1.0

        xs = jnp.arange(4.0)
        ys = brainstate.transform.checkpointed_for_loop(f, xs, pbar=2)
        self.assertTrue(jnp.allclose(ys, xs + 1.0))


class TestCheckpointedScanLengthValidation(unittest.TestCase):
    """checkpointed_scan with zero iterations must raise a clear ValueError
    instead of crashing later in ``math.log(0, base)`` (audit Tier C)."""

    def test_zero_length_xs_raises_value_error(self):
        def step(carry, x):
            return carry + x, carry

        with self.assertRaisesRegex(ValueError, 'length'):
            brainstate.transform.checkpointed_scan(step, 0.0, jnp.zeros((0, 2)))

    def test_zero_length_explicit_raises_value_error(self):
        def step(carry, x):
            return carry + x, carry

        with self.assertRaisesRegex(ValueError, 'length'):
            brainstate.transform.checkpointed_scan(step, 0.0, jnp.zeros((0,)), length=0)
