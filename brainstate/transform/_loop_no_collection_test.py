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


from unittest import TestCase

import jax
import jax.numpy as jnp

import brainstate


class TestWhileLoop(TestCase):
    def test1(self):
        a = brainstate.State(1.)
        b = brainstate.State(20.)

        def cond(_):
            return a.value < b.value

        def body(_):
            a.value += 1.

        brainstate.transform.while_loop(cond, body, None)

        print(a.value, b.value)

    def test2(self):
        a = brainstate.State(1.)
        b = brainstate.State(20.)

        def cond(x):
            return a.value < b.value

        def body(x):
            a.value += x
            return x

        r = brainstate.transform.while_loop(cond, body, 1.)

        print(a.value, b.value, r)


class TestWhileLoopCoverage(TestCase):
    """Validation, the disable-jit fast path, and the cond-write guard."""

    def test_non_callable_arguments_raise(self):
        """Non-callable ``cond_fun``/``body_fun`` raise ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.while_loop(1, lambda v: v, 0.0)

    def test_disable_jit_runs_python_loop(self):
        """Under ``disable_jit`` the loop runs eagerly in Python."""
        a = brainstate.State(jnp.array(0.0))

        def cond(v):
            return a.value < 5

        def body(v):
            a.value = a.value + 1.0
            return v

        with jax.disable_jit():
            brainstate.transform.while_loop(cond, body, 0.0)
        self.assertEqual(int(a.value), 5)

    def test_cond_writing_state_raises(self):
        """A ``cond_fun`` that writes a state raises ``ValueError``."""
        a = brainstate.State(jnp.array(0.0))

        def cond(v):
            a.value = a.value + 1.0  # illegal write in the condition
            return a.value < 5

        def body(v):
            return v

        with self.assertRaises(ValueError):
            brainstate.transform.while_loop(cond, body, 0.0)


class TestBoundedWhileLoop(TestCase):
    """``bounded_while_loop`` validation and execution with state."""

    def test_negative_max_steps_raises(self):
        """A negative ``max_steps`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            brainstate.transform.bounded_while_loop(
                lambda v: True, lambda v: v, 0.0, max_steps=-1
            )

    def test_zero_max_steps_returns_init(self):
        """``max_steps=0`` returns the (arrayified) initial value unchanged."""
        out = brainstate.transform.bounded_while_loop(
            lambda v: v < 10, lambda v: v + 1, 3.0, max_steps=0
        )
        self.assertTrue(bool(jnp.allclose(out, 3.0)))

    def test_runs_until_condition_false(self):
        """The loop advances state until the condition is false (within bound)."""
        i = brainstate.State(jnp.array(0))

        def cond(v):
            return i.value < 5

        def body(v):
            i.value = i.value + 1
            return v

        brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=32)
        self.assertEqual(int(i.value), 5)

    def test_max_steps_caps_iterations(self):
        """The loop stops at ``max_steps`` even if the condition stays true."""
        i = brainstate.State(jnp.array(0))

        def cond(v):
            return i.value < 1000  # never satisfied within the bound

        def body(v):
            i.value = i.value + 1
            return v

        brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=4, base=2)
        self.assertLessEqual(int(i.value), 4)

    def test_cond_writing_state_raises(self):
        """A ``cond_fun`` that writes a state raises ``ValueError``."""
        a = brainstate.State(jnp.array(0.0))

        def cond(v):
            a.value = a.value + 1.0
            return a.value < 5

        def body(v):
            return v

        with self.assertRaises(ValueError):
            brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=10)


class TestBoundedWhileLoopSemantics(TestCase):
    """``bounded_while_loop`` must match ``while_loop`` whenever the loop
    exits before ``max_steps`` (audit H2: the skip path corrupted the carry,
    ``max_steps=1`` ignored the condition, and vmapped lanes over-stepped).
    """

    def _python_loop(self, cond, body, val, max_steps):
        steps = 0
        while steps < max_steps and cond(val):
            val = body(val)
            steps += 1
        return val

    def test_early_exit_value(self):
        cond = lambda x: x < 3.0
        body = lambda x: x + 1.0
        out = brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=8, base=2)
        self.assertEqual(float(out), 3.0)

    def test_early_exit_value_default_base(self):
        cond = lambda x: x < 3.0
        body = lambda x: x + 1.0
        out = brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=4)
        self.assertEqual(float(out), 3.0)

    def test_max_steps_one_respects_condition(self):
        cond = lambda x: x < 3.0
        body = lambda x: x + 5.0
        out = brainstate.transform.bounded_while_loop(cond, body, 100.0, max_steps=1)
        self.assertEqual(float(out), 100.0)

        out = brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=1)
        self.assertEqual(float(out), 5.0)

    def test_vmap_per_lane_exit(self):
        cond = lambda x: x < 3.0
        body = lambda x: x + 1.0

        def run(x0):
            return brainstate.transform.bounded_while_loop(cond, body, x0, max_steps=8, base=2)

        out = jax.vmap(run)(jnp.asarray([0.0, 0.9]))
        self.assertTrue(bool(jnp.allclose(out, jnp.asarray([3.0, 3.9]))))

    def test_matches_python_semantics(self):
        cond = lambda x: x < 100.0
        body = lambda x: x * 2.0 + 1.0
        for init in (0.0, 50.0, 200.0):
            for max_steps, base in ((1, 16), (3, 2), (8, 2), (10, 16), (100, 4)):
                out = brainstate.transform.bounded_while_loop(
                    cond, body, init, max_steps=max_steps, base=base
                )
                expected = self._python_loop(cond, body, init, max_steps)
                self.assertEqual(
                    float(out), float(expected),
                    msg=f'init={init} max_steps={max_steps} base={base}'
                )

    def test_stateful_early_exit(self):
        counter = brainstate.State(jnp.asarray(0.0))

        def cond(x):
            return counter.value < 3.0

        def body(x):
            counter.value = counter.value + 1.0
            return x + 10.0

        out = brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=8, base=2)
        self.assertEqual(float(counter.value), 3.0)
        self.assertEqual(float(out), 30.0)

    def test_stateful_capped(self):
        counter = brainstate.State(jnp.asarray(0.0))

        def cond(x):
            return counter.value < 1000.0

        def body(x):
            counter.value = counter.value + 1.0
            return x + 10.0

        out = brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=2, base=2)
        self.assertEqual(float(counter.value), 2.0)
        self.assertEqual(float(out), 20.0)

    def test_grad_through_early_exit(self):
        def f(x):
            return brainstate.transform.bounded_while_loop(
                lambda v: v < 5.0, lambda v: v * 2.0, x, max_steps=8, base=2
            )

        self.assertEqual(float(f(1.0)), 8.0)
        self.assertEqual(float(jax.grad(f)(1.0)), 8.0)


class TestWhileLoopDisableJitVmap(TestCase):
    """Under ``jax.disable_jit()`` the Python fast path cannot concretize a
    vmapped predicate; the except clause must fall through to the primitive
    ``lax.while_loop`` instead of crashing (audit M39).

    The except previously referenced ``jax.core.ConcretizationTypeError``,
    which does not exist (``jax.core`` has no such attribute), so evaluating
    the handler raised ``AttributeError`` and masked the intended fallback.
    """

    def test_vmap_under_disable_jit_returns_correct_result(self):
        cond = lambda v: jnp.all(v < 3)
        body = lambda v: v + 1
        run = lambda init: brainstate.transform.while_loop(cond, body, init)
        with jax.disable_jit():
            out = jax.vmap(run)(jnp.array([0, 1, 2]))
        self.assertTrue(bool(jnp.array_equal(out, jnp.array([3, 3, 3]))))


class TestBoundedWhileLoopBaseValidation(TestCase):
    """bounded_while_loop must validate ``base`` up front with a clear
    ValueError instead of crashing inside ``math.log(max_steps, base)``
    (audit M40).

    base=1 previously raised ``ZeroDivisionError`` (log base 1 divides by
    ``math.log(1) == 0``) and base<=0 raised ``ValueError: math domain
    error``; base=True must also be rejected (bool subclasses int and
    collapses to the broken base==1 path).
    """

    def test_base_one_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'base'):
            brainstate.transform.bounded_while_loop(
                lambda v: v < 3, lambda v: v + 1, jnp.asarray(0), max_steps=10, base=1
            )

    def test_base_zero_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'base'):
            brainstate.transform.bounded_while_loop(
                lambda v: v < 3, lambda v: v + 1, jnp.asarray(0), max_steps=10, base=0
            )

    def test_base_negative_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'base'):
            brainstate.transform.bounded_while_loop(
                lambda v: v < 3, lambda v: v + 1, jnp.asarray(0), max_steps=10, base=-2
            )

    def test_base_true_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'base'):
            brainstate.transform.bounded_while_loop(
                lambda v: v < 3, lambda v: v + 1, jnp.asarray(0), max_steps=10, base=True
            )

    def test_valid_base_still_runs(self):
        out = brainstate.transform.bounded_while_loop(
            lambda v: v < 3, lambda v: v + 1, jnp.asarray(0), max_steps=10, base=2
        )
        self.assertEqual(int(out), 3)
