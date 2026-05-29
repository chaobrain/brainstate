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

import brainstate


class TestRemat(unittest.TestCase):
    def test_basic_remat(self):
        module = brainstate.transform.remat(brainstate.nn.Linear(2, 3))
        y = module(jnp.ones((1, 2)))
        assert y.shape == (1, 3)

    def test_remat_with_scan(self):
        class ScanLinear(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = brainstate.nn.Linear(3, 3)

            def __call__(self, x: jax.Array):
                @brainstate.transform.remat
                def fun(x: jax.Array, _):
                    x = self.linear(x)
                    return x, None

                return brainstate.transform.scan(fun, x, None, length=10)[0]

        m = ScanLinear()

        assert m.linear.weight.value['weight'].shape == (3, 3)
        assert m.linear.weight.value['bias'].shape == (3,)

        y = m(jnp.ones((10, 3)))
        assert y.shape == (10, 3)


class TestCheckpointAliasAndDecorator(unittest.TestCase):
    def test_remat_alias_matches_checkpoint(self):
        """remat is an alias for checkpoint (line 151 import-time assignment)."""
        from brainstate.transform._grad_checkpoint import remat, checkpoint
        self.assertIs(remat, checkpoint)

    def test_checkpoint_as_decorator_with_parentheses(self):
        """checkpoint() called with parentheses (Missing() path, line 151) returns a decorator."""
        @brainstate.transform.checkpoint(prevent_cse=False)
        def f(x):
            return x * 2.0

        result = f(jnp.array([1.0, 2.0]))
        self.assertTrue(jnp.allclose(result, jnp.array([2.0, 4.0])))

    def test_checkpoint_with_policy(self):
        """checkpoint with an explicit policy parameter exercises the policy path."""
        policy = jax.checkpoint_policies.everything_saveable

        @brainstate.transform.checkpoint(policy=policy)
        def g(x):
            return jnp.sin(x)

        result = g(jnp.array([0.0, jnp.pi / 2]))
        self.assertTrue(jnp.allclose(result, jnp.sin(jnp.array([0.0, jnp.pi / 2]))))


if __name__ == '__main__':
    unittest.main()
