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

"""Tests for the transform plumbing in ``transform/_util.py``.

Two layers are covered: multi-branch ``State`` merging exercised through
``cond``/``switch`` (the ``wrap_single_fun_in_multi_branches`` path), and the
gradient helpers ``warp_grad_fn`` / ``tree_random_split`` exercised directly.
"""

import unittest

import jax
import jax.numpy as jnp

import brainstate
from brainstate.transform._util import warp_grad_fn, tree_random_split


class TestMultiBranchStateMerging(unittest.TestCase):
    """Branches that write states must merge consistently across cond/switch."""

    def test_cond_branches_write_same_state(self):
        """``cond`` branches writing the same state pick the taken branch's value."""
        st = brainstate.State(jnp.array(0.0))

        def true_branch(x):
            st.value = jnp.array(1.0) + x

        def false_branch(x):
            st.value = jnp.array(-1.0) + x

        brainstate.transform.cond(True, true_branch, false_branch, 0.0)
        self.assertTrue(bool(jnp.allclose(st.value, 1.0)))

        brainstate.transform.cond(False, true_branch, false_branch, 0.0)
        self.assertTrue(bool(jnp.allclose(st.value, -1.0)))

    def test_switch_selects_branch(self):
        """``switch`` dispatches to the indexed branch and merges its write."""
        st = brainstate.State(jnp.array(0.0))

        def make(v):
            def branch(x):
                st.value = jnp.array(float(v)) + x

            return branch

        brainstate.transform.switch(1, [make(10), make(20), make(30)], 0.0)
        self.assertTrue(bool(jnp.allclose(st.value, 20.0)))

    def test_branches_writing_disjoint_states_merge(self):
        """Branches writing different states leave both consistently updated."""
        a = brainstate.State(jnp.array(0.0))
        b = brainstate.State(jnp.array(0.0))

        def t(x):
            a.value = jnp.array(5.0)

        def f(x):
            b.value = jnp.array(7.0)

        brainstate.transform.cond(True, t, f, 0.0)
        self.assertTrue(bool(jnp.allclose(a.value, 5.0)))
        # untaken branch's state keeps its previous (read) value
        self.assertTrue(bool(jnp.allclose(b.value, 0.0)))


class TestWarpGradFn(unittest.TestCase):
    """``warp_grad_fn`` rebinds selected positional arguments."""

    def test_int_argnum_rebinds_single_arg(self):
        """An int argnum produces a one-argument closure and returns that arg."""
        x = jnp.array([1.0, 2.0])
        y = jnp.array([10.0, 20.0])
        new_fn, param = warp_grad_fn(lambda a, b: a + b, 0, (x, y), {})
        self.assertTrue(bool(jnp.allclose(param, x)))
        self.assertTrue(bool(jnp.allclose(new_fn(jnp.array([3.0, 4.0])), jnp.array([13.0, 24.0]))))

    def test_int_argnum_out_of_range_raises(self):
        """An int argnum out of range raises ``AssertionError``."""
        with self.assertRaises(AssertionError):
            warp_grad_fn(lambda a: a, 5, (jnp.ones((2,)),), {})

    def test_sequence_argnums_rebinds_multiple_args(self):
        """A sequence argnums produces a closure over those positions."""
        x = jnp.array([1.0])
        y = jnp.array([2.0])
        z = jnp.array([3.0])
        new_fn, params = warp_grad_fn(lambda a, b, c: a + b + c, (0, 2), (x, y, z), {})
        self.assertEqual(len(params), 2)
        self.assertTrue(bool(jnp.allclose(params[0], x)))
        self.assertTrue(bool(jnp.allclose(params[1], z)))
        out = new_fn([jnp.array([10.0]), jnp.array([30.0])])
        self.assertTrue(bool(jnp.allclose(out, jnp.array([42.0]))))

    def test_sequence_argnums_out_of_range_raises(self):
        """A sequence argnum out of range raises ``AssertionError``."""
        with self.assertRaises(AssertionError):
            warp_grad_fn(lambda a, b: a + b, (0, 9), (jnp.ones((2,)), jnp.ones((2,))), {})


class TestTreeRandomSplit(unittest.TestCase):
    """``tree_random_split`` produces one key per leaf of the target tree."""

    def test_split_matches_target_structure(self):
        """Keys mirror the structure of ``target``."""
        key = brainstate.random.split_key()
        target = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}
        keys = tree_random_split(key, target=target)
        self.assertEqual(set(keys.keys()), {"a", "b"})
        # Typed PRNG keys compare via their raw key data.
        a_data = tuple(jax.random.key_data(keys["a"]).tolist())
        b_data = tuple(jax.random.key_data(keys["b"]).tolist())
        self.assertNotEqual(a_data, b_data)

    def test_split_with_explicit_treedef(self):
        """An explicit ``treedef`` is honored over ``target``."""
        key = brainstate.random.split_key()
        treedef = jax.tree.structure([0, 0, 0])
        keys = tree_random_split(key, treedef=treedef)
        self.assertEqual(len(keys), 3)


if __name__ == "__main__":
    unittest.main()
