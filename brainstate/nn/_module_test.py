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

    def test_init_all_states(self):
        model = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 10),
            brainstate.nn.GRUCell(10, 20),
            brainstate.nn.Linear(10, 10),
        )
        a = model.init_all_states()
        print(a)

        b = model.init_all_states(vmap_size=10)
        print(b)

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

    def test_children(self):
        """Test children() method for getting immediate child modules."""
        from brainstate.nn import ParaM
        import jax.numpy as jnp

        class SubModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = ParaM(jnp.ones(10))

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = SubModule()
                self.layer2 = SubModule()
                # Nested module - should not appear in children()
                self.layer1.nested = SubModule()

        mod = TestModule()

        # Get all immediate children
        children = mod.children()
        for child in children:
            self.assertIsInstance(child, SubModule)

    def test_named_children(self):
        """Test named_children() iterator."""

        class SubModule(brainstate.nn.Module):
            pass

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = SubModule()
                self.beta = SubModule()

        mod = TestModule()
        named = list(mod.named_children())

        self.assertEqual(len(named), 2)
        names = [name for name, _ in named]
        self.assertTrue(any('alpha' in n for n in names))
        self.assertTrue(any('beta' in n for n in names))

    def test_modules(self):
        """Test modules() method for getting all modules in tree."""

        class SubModule(brainstate.nn.Module):
            pass

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = SubModule()
                self.layer2 = SubModule()

        mod = TestModule()

        # Test with include_self=True (default)
        all_modules = tuple(mod.modules(include_self=True))
        self.assertEqual(len(all_modules), 3)  # mod + layer1 + layer2

        # Test with include_self=False
        children_only = tuple(mod.modules(include_self=False))
        self.assertEqual(len(children_only), 2)  # layer1 + layer2

    def test_named_modules(self):
        """Test named_modules() iterator."""

        class SubModule(brainstate.nn.Module):
            pass

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = SubModule()
                self.layer2 = SubModule()

        mod = TestModule()

        # Test with include_self=True (default)
        named = list(mod.named_modules(include_self=True))
        self.assertEqual(len(named), 3)  # mod + layer1 + layer2

        # Check that empty string is used for root
        names = [name for name, _ in named]
        self.assertTrue('' in names)  # Root module
        self.assertTrue(any('layer1' in n for n in names))
        self.assertTrue(any('layer2' in n for n in names))

        # Test with prefix
        named_with_prefix = list(mod.named_modules(prefix='model', include_self=False))
        names_with_prefix = [name for name, _ in named_with_prefix]
        self.assertTrue(any('model.layer1' in n for n in names_with_prefix))

    def test_parameters(self):
        """Test parameters() method (PyTorch-compatible alias)."""
        from brainstate.nn import ParaM
        import jax.numpy as jnp

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = ParaM(jnp.ones(10), fit_par=True)
                self.param2 = ParaM(jnp.ones(5), fit_par=True)

        mod = TestModule()

        # Test recurse=True (default)
        params = mod.parameters(recurse=True)
        self.assertEqual(len(params), 2)

        # Should be equivalent to para_modules()
        para_mods = mod.para_modules()
        self.assertEqual(len(params), len(para_mods))

    def test_named_parameters(self):
        """Test named_parameters() iterator (PyTorch-compatible alias)."""
        from brainstate.nn import ParaM
        import jax.numpy as jnp

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = ParaM(jnp.ones(10), fit_par=True)
                self.beta = ParaM(jnp.ones(5), fit_par=True)

        mod = TestModule()

        # Test without prefix
        named = list(mod.named_parameters())
        self.assertEqual(len(named), 2)
        names = [name for name, _ in named]
        self.assertTrue(any('alpha' in n for n in names))
        self.assertTrue(any('beta' in n for n in names))

        # Test with prefix
        named_with_prefix = list(mod.named_parameters(prefix='model'))
        names_with_prefix = [name for name, _ in named_with_prefix]
        self.assertTrue(any('model.alpha' in n for n in names_with_prefix))

    def test_integration_pytorch_style(self):
        """Test PyTorch-style iteration patterns work correctly."""
        from brainstate.nn import ParaM

        class Linear(brainstate.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.weight = ParaM(brainstate.random.rand(in_size, out_size))

        class MLP(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(10, 20)
                self.fc2 = Linear(20, 5)

        model = MLP()

        # Test PyTorch-style parameter iteration
        param_count = 0
        for param in model.parameters().values():
            param_count += 1
        self.assertEqual(param_count, 2)

        # Test PyTorch-style named parameter iteration
        named_params = list(model.named_parameters())
        self.assertEqual(len(named_params), 2)

        # Test PyTorch-style module iteration
        # Note: ParaM instances are also modules, so count includes them
        module_count = 0
        for module in model.modules():
            module_count += 1
        self.assertEqual(module_count, 5)  # MLP + fc1 + fc2 + 2 ParaM

        # Test PyTorch-style children iteration
        children = list(model.children())
        self.assertEqual(len(children), 2)  # fc1 + fc2
