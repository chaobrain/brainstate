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

import brainstate


class TestModule(unittest.TestCase):
    def test_states(self):
        class A(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.State(brainstate.random.random(10, 20))
                self.b = brainstate.State(brainstate.random.random(10, 20))

        class B(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()
                self.b = brainstate.State(brainstate.random.random(10, 20))

        b = B()
        print()
        print(b.states())
        print(b.states())
        print(b.states())
        print(b.states())

    def test_reg_loss_aggregation(self):
        """Test reg_loss() method for regularization aggregation."""
        from brainstate.nn import ParaM, L2Reg, L1Reg
        import jax.numpy as jnp

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                # Parameter with L2 regularization
                self.param1 = ParaM(jnp.ones(10), reg=L2Reg(0.1), fit_par=True)
                # Parameter with L1 regularization
                self.param2 = ParaM(jnp.ones(5), reg=L1Reg(0.05), fit_par=True)
                # Parameter without regularization
                self.param3 = ParaM(jnp.ones(3), fit_par=True)

        mod = TestModule()

        # Get total regularization loss
        total_loss = mod.reg_loss()
        self.assertGreater(total_loss, 0.0)

        # Get loss from L2-regularized params only (compute manually)
        all_params = mod.para_modules()
        l2_params = {k: v for k, v in all_params.items() if isinstance(v.reg, L2Reg)}
        l2_loss = sum(p.reg_loss() for p in l2_params.values())
        self.assertGreater(l2_loss, 0.0)
        self.assertLess(l2_loss, total_loss)

        # Empty module should return 0.0
        empty_mod = brainstate.nn.Module()
        self.assertEqual(empty_mod.reg_loss(), 0.0)

    def test_named_params(self):
        """Test named_params() iterator."""
        from brainstate.nn import ParaM
        import jax.numpy as jnp

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = ParaM(jnp.ones(10), fit_par=True)
                self.beta = ParaM(jnp.ones(5), fit_par=True)

        mod = TestModule()
        named = list(mod.named_para_modules())

        self.assertEqual(len(named), 2)
        names = [name for name, _ in named]
        self.assertTrue(any('alpha' in n for n in names))
        self.assertTrue(any('beta' in n for n in names))

    def test_backward_compatibility(self):
        """Test that existing modules still work."""
        # Module with update() override should work as before
        class OldStyleModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.state = brainstate.State(brainstate.random.rand(10))

            def update(self, x):
                return x + self.state.value

        mod = OldStyleModule()
        import jax.numpy as jnp
        x = jnp.ones(10)
        result = mod.update(x)
        self.assertEqual(result.shape, (10,))

        # Existing functionality should still work
        states = mod.states()
        self.assertEqual(len(states), 1)
