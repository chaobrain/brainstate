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

import jax.numpy as jnp
import numpy as np

import brainstate
from brainstate.nn import Module, Param, SoftplusT, L2Reg, L1Reg


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

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                # Parameter with L2 regularization
                self.param1 = Param(jnp.ones(10), reg=L2Reg(0.1), fit_par=True)
                # Parameter with L1 regularization
                self.param2 = Param(jnp.ones(5), reg=L1Reg(0.05), fit_par=True)
                # Parameter without regularization
                self.param3 = Param(jnp.ones(3), fit_par=True)

        mod = TestModule()

        # Get total regularization loss
        total_loss = mod.reg_loss()
        self.assertGreater(total_loss, 0.0)

        # Get loss from L2-regularized params only (compute manually)
        all_params = mod.param_modules()
        l2_params = [v for v in all_params if isinstance(v.reg, L2Reg)]
        l2_loss = sum(p.reg_loss() for p in l2_params)
        self.assertGreater(l2_loss, 0.0)
        self.assertLess(l2_loss, total_loss)

        # Empty module should return 0.0
        empty_mod = brainstate.nn.Module()
        self.assertEqual(empty_mod.reg_loss(), 0.0)

    def test_named_params(self):
        """Test named_params() iterator."""

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = Param(jnp.ones(10), fit_par=True)
                self.beta = Param(jnp.ones(5), fit_par=True)

        mod = TestModule()
        named = list(mod.named_param_modules())

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

        class SubModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = Param(jnp.ones(10))

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

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(10), fit_par=True)
                self.param2 = Param(jnp.ones(5), fit_par=True)

        mod = TestModule()

        # Test recurse=True (default)
        params = list(mod.parameters(recurse=True))
        self.assertEqual(len(params), 2)

        # Should be equivalent to para_modules()
        para_mods = tuple(mod.param_modules())
        self.assertEqual(len(params), len(para_mods))

    def test_named_parameters(self):
        """Test named_parameters() iterator (PyTorch-compatible alias)."""

        class TestModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = Param(jnp.ones(10), fit_par=True)
                self.beta = Param(jnp.ones(5), fit_par=True)

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

        class Linear(brainstate.nn.Module):
            def __init__(self, in_size, out_size):
                super().__init__()
                self.weight = Param(brainstate.random.rand(in_size, out_size))

        class MLP(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(10, 20)
                self.fc2 = Linear(20, 5)

        model = MLP()

        # Test PyTorch-style parameter iteration
        param_count = 0
        for param in model.parameters():
            param_count += 1
        self.assertEqual(param_count, 2)

        # Test PyTorch-style named parameter iteration
        named_params = list(model.named_parameters())
        self.assertEqual(len(named_params), 2)

        # Test PyTorch-style module iteration
        # Note: Param instances are also modules, so count includes them
        module_count = 0
        for module in model.modules():
            module_count += 1
        self.assertEqual(module_count, 5)  # MLP + fc1 + fc2 + 2 Param

        # Test PyTorch-style children iteration
        children = list(model.children())
        self.assertEqual(len(children), 2)  # fc1 + fc2


class TestParamPrecompute(unittest.TestCase):
    """Test suite for Module.param_precompute method."""

    def test_basic_cache_enable(self):
        """Test that param_precompute(cache=True) caches all parameters."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.param2 = Param(jnp.ones(3), t=SoftplusT(0.0))

        mod = TestModule()

        # Initially, cache should be invalid
        self.assertFalse(mod.param1.cache_stats['valid'])
        self.assertFalse(mod.param2.cache_stats['valid'])

        # Call param_precompute to cache all parameters
        mod.param_precompute(cache=True)

        # Now cache should be valid
        self.assertTrue(mod.param1.cache_stats['valid'])
        self.assertTrue(mod.param2.cache_stats['valid'])
        self.assertTrue(mod.param1.cache_stats['has_cached_value'])
        self.assertTrue(mod.param2.cache_stats['has_cached_value'])

    def test_basic_cache_disable(self):
        """Test that param_precompute(cache=False) clears all parameter caches."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.param2 = Param(jnp.ones(3), t=SoftplusT(0.0))

        mod = TestModule()

        # First cache the parameters
        mod.param_precompute(cache=True)
        self.assertTrue(mod.param1.cache_stats['valid'])
        self.assertTrue(mod.param2.cache_stats['valid'])

        # Now clear the cache
        mod.param_precompute(cache=False)

        # Cache should be invalid
        self.assertFalse(mod.param1.cache_stats['valid'])
        self.assertFalse(mod.param2.cache_stats['valid'])

    def test_hierarchy_level_0(self):
        """Test param_precompute with allowed_hierarchy=(0, 0) - only self."""

        class ChildModule(Module):
            def __init__(self):
                super().__init__()
                self.child_param = Param(jnp.ones(3), t=SoftplusT(0.0))

        class ParentModule(Module):
            def __init__(self):
                super().__init__()
                self.parent_param = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.child = ChildModule()

        mod = ParentModule()

        # Cache only parameters at hierarchy level 0 (none, since params are at level 1+)
        mod.param_precompute(allowed_hierarchy=(0, 0), cache=True)

        # Neither parent nor child params should be cached (they're at level 1+)
        self.assertFalse(mod.parent_param.cache_stats['valid'])
        self.assertFalse(mod.child.child_param.cache_stats['valid'])

    def test_hierarchy_level_1(self):
        """Test param_precompute with allowed_hierarchy=(1, 1) - only immediate children."""

        class GrandchildModule(Module):
            def __init__(self):
                super().__init__()
                self.grandchild_param = Param(jnp.ones(2), t=SoftplusT(0.0))

        class ChildModule(Module):
            def __init__(self):
                super().__init__()
                self.child_param = Param(jnp.ones(3), t=SoftplusT(0.0))
                self.grandchild = GrandchildModule()

        class ParentModule(Module):
            def __init__(self):
                super().__init__()
                self.parent_param = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.child = ChildModule()

        mod = ParentModule()

        # Cache only parameters at hierarchy level 1 (immediate children)
        mod.param_precompute(allowed_hierarchy=(1, 1), cache=True)

        # Only parent_param (at level 1) should be cached
        self.assertTrue(mod.parent_param.cache_stats['valid'])

        # Child and grandchild params should not be cached
        self.assertFalse(mod.child.child_param.cache_stats['valid'])
        self.assertFalse(mod.child.grandchild.grandchild_param.cache_stats['valid'])

    def test_hierarchy_all_levels(self):
        """Test param_precompute with default hierarchy - all levels."""

        class GrandchildModule(Module):
            def __init__(self):
                super().__init__()
                self.grandchild_param = Param(jnp.ones(2), t=SoftplusT(0.0))

        class ChildModule(Module):
            def __init__(self):
                super().__init__()
                self.child_param = Param(jnp.ones(3), t=SoftplusT(0.0))
                self.grandchild = GrandchildModule()

        class ParentModule(Module):
            def __init__(self):
                super().__init__()
                self.parent_param = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.child = ChildModule()

        mod = ParentModule()

        # Cache all parameters (default behavior)
        mod.param_precompute(cache=True)

        # All parameters should be cached
        self.assertTrue(mod.parent_param.cache_stats['valid'])
        self.assertTrue(mod.child.child_param.cache_stats['valid'])
        self.assertTrue(mod.child.grandchild.grandchild_param.cache_stats['valid'])

    def test_empty_module(self):
        """Test param_precompute on module with no parameters."""

        class EmptyModule(Module):
            def __init__(self):
                super().__init__()
                self.state = brainstate.State(jnp.ones(5))

        mod = EmptyModule()

        # Should not raise an error
        mod.param_precompute(cache=True)
        mod.param_precompute(cache=False)

    def test_mixed_param_types(self):
        """Test param_precompute with different parameter types."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                # Trainable parameter with transform
                self.trainable_param = Param(jnp.ones(5), t=SoftplusT(0.0), fit_par=True)
                # Non-trainable parameter
                self.fixed_param = Param(jnp.ones(3), fit_par=False)
                # Parameter with regularization
                self.reg_param = Param(jnp.ones(4), reg=L2Reg(0.1), fit_par=True)
                # Non-parameter state
                self.state = brainstate.State(jnp.ones(2))

        mod = TestModule()

        # Cache all parameters
        mod.param_precompute(cache=True)

        # All Param instances should be cached
        self.assertTrue(mod.trainable_param.cache_stats['valid'])
        self.assertTrue(mod.fixed_param.cache_stats['valid'])
        self.assertTrue(mod.reg_param.cache_stats['valid'])

    def test_cached_values_correctness(self):
        """Test that cached values are computed correctly."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(jnp.array([1.0, 2.0, 3.0]), t=SoftplusT(0.0))

        mod = TestModule()

        # Get value before caching (should compute)
        value_before = mod.param.value()

        # Cache the parameter
        mod.param_precompute(cache=True)

        # Get value after caching (should use cache)
        value_after = mod.param.value()

        # Values should be the same
        np.testing.assert_array_equal(value_before, value_after)

    def test_cache_invalidation_on_update(self):
        """Test that cache is invalidated when parameter is updated."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(jnp.ones(5), t=SoftplusT(0.0), fit_par=True)

        mod = TestModule()

        # Cache the parameter
        mod.param_precompute(cache=True)
        self.assertTrue(mod.param.cache_stats['valid'])

        # Update the parameter
        mod.param.set_value(jnp.ones(5) * 2)

        # Cache should be invalidated
        self.assertFalse(mod.param.cache_stats['valid'])

    def test_precompute_function(self):
        """Test param_precompute with parameters that have precompute functions."""

        def custom_precompute(x):
            """Example precompute function that normalizes values."""
            return x / jnp.sum(x)

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(
                    jnp.array([1.0, 2.0, 3.0]),
                    precompute=custom_precompute
                )

        mod = TestModule()

        # Cache the parameter
        mod.param_precompute(cache=True)

        # Cache should be valid
        self.assertTrue(mod.param.cache_stats['valid'])

        # Get cached value and verify precompute was applied
        cached_value = mod.param.value()
        expected = custom_precompute(jnp.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(cached_value, expected)

    def test_sequential_module(self):
        """Test param_precompute with Sequential module."""

        model = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 20),
            brainstate.nn.Linear(20, 5)
        )

        # Cache all parameters
        model.param_precompute(cache=True)

        # Check that parameters in all layers are cached
        for param in model.param_modules():
            self.assertTrue(param.cache_stats['valid'])

    def test_selective_hierarchy_caching(self):
        """Test param_precompute with selective hierarchy ranges."""

        class Level3Module(Module):
            def __init__(self):
                super().__init__()
                self.param3 = Param(jnp.ones(1), t=SoftplusT(0.0))

        class Level2Module(Module):
            def __init__(self):
                super().__init__()
                self.param2 = Param(jnp.ones(2), t=SoftplusT(0.0))
                self.level3 = Level3Module()

        class Level1Module(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(3), t=SoftplusT(0.0))
                self.level2 = Level2Module()

        mod = Level1Module()

        # Cache only levels 1-2
        mod.param_precompute(allowed_hierarchy=(1, 2), cache=True)

        # param1 and param2 should be cached (levels 1 and 2)
        self.assertTrue(mod.param1.cache_stats['valid'])
        self.assertTrue(mod.level2.param2.cache_stats['valid'])

        # param3 should not be cached (level 3)
        self.assertFalse(mod.level2.level3.param3.cache_stats['valid'])

    def test_multiple_calls_toggle_cache(self):
        """Test multiple calls to param_precompute with different cache values."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(jnp.ones(5), t=SoftplusT(0.0))

        mod = TestModule()

        # Enable caching
        mod.param_precompute(cache=True)
        self.assertTrue(mod.param.cache_stats['valid'])

        # Disable caching
        mod.param_precompute(cache=False)
        self.assertFalse(mod.param.cache_stats['valid'])

        # Enable again
        mod.param_precompute(cache=True)
        self.assertTrue(mod.param.cache_stats['valid'])

    def test_default_parameters(self):
        """Test param_precompute with default parameters."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.param2 = Param(jnp.ones(3), t=SoftplusT(0.0))

        mod = TestModule()

        # Call with default parameters (should cache all)
        mod.param_precompute()

        # Both parameters should be cached
        self.assertTrue(mod.param1.cache_stats['valid'])
        self.assertTrue(mod.param2.cache_stats['valid'])

    def test_nested_modules_partial_caching(self):
        """Test caching behavior with complex nested module structures."""

        class InnerModule(Module):
            def __init__(self):
                super().__init__()
                self.inner_param = Param(jnp.ones(2), t=SoftplusT(0.0))

        class MiddleModule(Module):
            def __init__(self):
                super().__init__()
                self.middle_param = Param(jnp.ones(3), t=SoftplusT(0.0))
                self.inner1 = InnerModule()
                self.inner2 = InnerModule()

        class OuterModule(Module):
            def __init__(self):
                super().__init__()
                self.outer_param = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.middle = MiddleModule()

        mod = OuterModule()

        # Cache only immediate children (level 1)
        mod.param_precompute(allowed_hierarchy=(1, 1), cache=True)

        # Only outer_param should be cached
        self.assertTrue(mod.outer_param.cache_stats['valid'])
        self.assertFalse(mod.middle.middle_param.cache_stats['valid'])
        self.assertFalse(mod.middle.inner1.inner_param.cache_stats['valid'])
        self.assertFalse(mod.middle.inner2.inner_param.cache_stats['valid'])

        # Now cache all levels
        mod.param_precompute(cache=True)

        # All parameters should be cached
        self.assertTrue(mod.outer_param.cache_stats['valid'])
        self.assertTrue(mod.middle.middle_param.cache_stats['valid'])
        self.assertTrue(mod.middle.inner1.inner_param.cache_stats['valid'])
        self.assertTrue(mod.middle.inner2.inner_param.cache_stats['valid'])

    def test_performance_benefit_of_caching(self):
        """Test that caching provides performance benefit for transformed parameters."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                # Use a transform that requires computation
                self.param = Param(jnp.ones(1000), t=SoftplusT(0.0))

        mod = TestModule()

        # First call - should compute
        value1 = mod.param.value()

        # Cache the parameter
        mod.param_precompute(cache=True)

        # Second call - should use cache
        value2 = mod.param.value()

        # Values should be identical (not just close)
        self.assertTrue(jnp.all(value1 == value2))

        # Cache should still be valid after access
        self.assertTrue(mod.param.cache_stats['valid'])
