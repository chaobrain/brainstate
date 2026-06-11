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
from brainstate import _testing
from brainstate.nn import Module, Param, SoftplusT, L2Reg, L1Reg
from brainstate.nn._module import Sequential, ElementWiseBlock
from brainstate._error import BrainStateError


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
                self.param1 = Param(jnp.ones(10), reg=L2Reg(0.1), fit=True)
                # Parameter with L1 regularization
                self.param2 = Param(jnp.ones(5), reg=L1Reg(0.05), fit=True)
                # Parameter without regularization
                self.param3 = Param(jnp.ones(3), fit=True)

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
                self.alpha = Param(jnp.ones(10), fit=True)
                self.beta = Param(jnp.ones(5), fit=True)

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
                self.param1 = Param(jnp.ones(10), fit=True)
                self.param2 = Param(jnp.ones(5), fit=True)

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
                self.alpha = Param(jnp.ones(10), fit=True)
                self.beta = Param(jnp.ones(5), fit=True)

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
        """Test that param_precompute context manager caches all parameters."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.param2 = Param(jnp.ones(3), t=SoftplusT(0.0))

        mod = TestModule()

        # Initially, cache should be invalid
        self.assertFalse(mod.param1.cache_stats['valid'])
        self.assertFalse(mod.param2.cache_stats['valid'])

        # Use context manager to cache all parameters
        with mod.param_precompute():
            # Inside context, cache should be valid
            self.assertTrue(mod.param1.cache_stats['valid'])
            self.assertTrue(mod.param2.cache_stats['valid'])
            self.assertTrue(mod.param1.cache_stats['has_cached_value'])
            self.assertTrue(mod.param2.cache_stats['has_cached_value'])

    def test_basic_cache_disable(self):
        """Test that param_precompute context manager clears caches on exit."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.param2 = Param(jnp.ones(3), t=SoftplusT(0.0))

        mod = TestModule()

        # Use context manager - caches should be valid inside
        with mod.param_precompute():
            self.assertTrue(mod.param1.cache_stats['valid'])
            self.assertTrue(mod.param2.cache_stats['valid'])

        # After exiting context, cache should be cleared
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
        with mod.param_precompute(allowed_hierarchy=(0, 0)):
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
        with mod.param_precompute(allowed_hierarchy=(1, 1)):
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
        with mod.param_precompute():
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
        with mod.param_precompute():
            pass  # Should complete without errors

    def test_mixed_param_types(self):
        """Test param_precompute with different parameter types."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                # Trainable parameter with transform
                self.trainable_param = Param(jnp.ones(5), t=SoftplusT(0.0), fit=True)
                # Non-trainable parameter
                self.fixed_param = Param(jnp.ones(3), fit=False)
                # Parameter with regularization
                self.reg_param = Param(jnp.ones(4), reg=L2Reg(0.1), fit=True)
                # Non-parameter state
                self.state = brainstate.State(jnp.ones(2))

        mod = TestModule()

        # Cache all parameters
        with mod.param_precompute():
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

        # Cache the parameter and get value
        with mod.param_precompute():
            # Get value after caching (should use cache)
            value_after = mod.param.value()

            # Values should be the same
            np.testing.assert_array_equal(value_before, value_after)

    def test_cache_invalidation_on_update(self):
        """Test that cache is invalidated when parameter is updated inside context."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(jnp.ones(5), t=SoftplusT(0.0), fit=True)

        mod = TestModule()

        # Cache the parameter
        with mod.param_precompute():
            self.assertTrue(mod.param.cache_stats['valid'])

            # Update the parameter inside context
            mod.param.set_value(jnp.ones(5) * 2)

            # Cache should be invalidated by the update
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
        with mod.param_precompute():
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
        with model.param_precompute():
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
        with mod.param_precompute(allowed_hierarchy=(1, 2)):
            # param1 and param2 should be cached (levels 1 and 2)
            self.assertTrue(mod.param1.cache_stats['valid'])
            self.assertTrue(mod.level2.param2.cache_stats['valid'])

            # param3 should not be cached (level 3)
            self.assertFalse(mod.level2.level3.param3.cache_stats['valid'])

    def test_multiple_calls_toggle_cache(self):
        """Test multiple context manager calls cache and clear correctly."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(jnp.ones(5), t=SoftplusT(0.0))

        mod = TestModule()

        # First context - should cache
        with mod.param_precompute():
            self.assertTrue(mod.param.cache_stats['valid'])

        # After exiting, cache should be cleared
        self.assertFalse(mod.param.cache_stats['valid'])

        # Second context - should cache again
        with mod.param_precompute():
            self.assertTrue(mod.param.cache_stats['valid'])

        # After exiting again, cache should be cleared
        self.assertFalse(mod.param.cache_stats['valid'])

    def test_default_parameters(self):
        """Test param_precompute with default parameters."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param1 = Param(jnp.ones(5), t=SoftplusT(0.0))
                self.param2 = Param(jnp.ones(3), t=SoftplusT(0.0))

        mod = TestModule()

        # Call with default parameters (should cache all)
        with mod.param_precompute():
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
        with mod.param_precompute(allowed_hierarchy=(1, 1)):
            # Only outer_param should be cached
            self.assertTrue(mod.outer_param.cache_stats['valid'])
            self.assertFalse(mod.middle.middle_param.cache_stats['valid'])
            self.assertFalse(mod.middle.inner1.inner_param.cache_stats['valid'])
            self.assertFalse(mod.middle.inner2.inner_param.cache_stats['valid'])

        # Now cache all levels
        with mod.param_precompute():
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
        with mod.param_precompute():
            # Second call - should use cache
            value2 = mod.param.value()

            # Values should be identical (not just close)
            self.assertTrue(jnp.all(value1 == value2))

            # Cache should still be valid after access
            self.assertTrue(mod.param.cache_stats['valid'])

    def test_exception_safety(self):
        """Test that cache is cleared even when exception occurs in context."""

        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.param = Param(jnp.ones(5), t=SoftplusT(0.0))

        mod = TestModule()

        # Cache should be cleared even if exception occurs
        try:
            with mod.param_precompute():
                # Cache should be valid inside context
                self.assertTrue(mod.param.cache_stats['valid'])
                # Raise an exception
                raise ValueError("Test error")
        except ValueError:
            pass

        # Cache should still be cleared after exception
        self.assertFalse(mod.param.cache_stats['valid'])

    def test_nested_contexts(self):
        """Test nested param_precompute contexts."""

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

        # Outer context caches all parameters
        with mod.param_precompute():
            self.assertTrue(mod.parent_param.cache_stats['valid'])
            self.assertTrue(mod.child.child_param.cache_stats['valid'])

            # Inner context with different hierarchy
            with mod.param_precompute(allowed_hierarchy=(1, 1)):
                # Both should still be cached (inner context caches level 1)
                self.assertTrue(mod.parent_param.cache_stats['valid'])
                # Child param is at level 2, so not cached by inner context
                # But it was already cached by outer context
                self.assertTrue(mod.child.child_param.cache_stats['valid'])

            # After inner context exits, it clears level 1 params
            # Outer context doesn't re-cache until we exit and re-enter
            self.assertFalse(mod.parent_param.cache_stats['valid'])

        # After outer context exits, all caches should be cleared
        self.assertFalse(mod.parent_param.cache_stats['valid'])
        self.assertFalse(mod.child.child_param.cache_stats['valid'])


class TestModuleConstruction(unittest.TestCase):
    """Cover ``Module`` construction, naming, and size normalization."""

    def test_default_name_and_sizes_are_none(self):
        """A freshly constructed module has no name and undefined sizes."""
        mod = Module()
        self.assertIsNone(mod.name)
        self.assertIsNone(mod.in_size)
        self.assertIsNone(mod.out_size)

    def test_name_is_stored(self):
        """A string ``name`` is stored and exposed by the property."""
        mod = Module(name='layer')
        self.assertEqual(mod.name, 'layer')

    def test_non_string_name_raises(self):
        """A non-string ``name`` raises ``AssertionError``."""
        with self.assertRaises(AssertionError):
            Module(name=123)

    def test_name_is_read_only(self):
        """Assigning to ``name`` raises ``AttributeError``."""
        mod = Module(name='layer')
        with self.assertRaises(AttributeError):
            mod.name = 'other'

    def test_in_size_int_normalized_to_tuple(self):
        """An int ``in_size`` is normalized to a 1-tuple."""
        mod = Module()
        mod.in_size = 10
        self.assertEqual(mod.in_size, (10,))

    def test_in_size_sequence_preserved(self):
        """A tuple ``in_size`` is preserved as a tuple."""
        mod = Module()
        mod.in_size = (2, 3)
        self.assertEqual(mod.in_size, (2, 3))

    def test_in_size_numpy_scalar(self):
        """A numpy integer scalar is accepted as ``in_size``."""
        mod = Module()
        mod.in_size = np.int32(4)
        self.assertEqual(mod.in_size, (4,))

    def test_in_size_invalid_type_raises(self):
        """A non-int, non-sequence ``in_size`` raises ``AssertionError``."""
        mod = Module()
        with self.assertRaises(AssertionError):
            mod.in_size = 'bad'

    def test_out_size_int_normalized_to_tuple(self):
        """An int ``out_size`` is normalized to a 1-tuple."""
        mod = Module()
        mod.out_size = 5
        self.assertEqual(mod.out_size, (5,))

    def test_out_size_list_preserved(self):
        """A list ``out_size`` is stored as a tuple."""
        mod = Module()
        mod.out_size = [4, 5]
        self.assertEqual(mod.out_size, (4, 5))

    def test_out_size_invalid_type_raises(self):
        """A non-int, non-sequence ``out_size`` raises ``AssertionError``."""
        mod = Module()
        with self.assertRaises(AssertionError):
            mod.out_size = 'bad'


class TestModuleCall(unittest.TestCase):
    """Cover ``Module.update``/``__call__`` dispatch and operator support."""

    def test_bare_update_not_implemented(self):
        """The base ``update`` raises ``NotImplementedError`` when un-overridden."""
        mod = Module()
        with self.assertRaises(NotImplementedError):
            mod.update(1)

    def test_call_forwards_to_update(self):
        """``__call__`` forwards positional args to ``update``."""

        class _Scale(Module):
            """Module that scales its input by two."""

            def update(self, x):
                """Return ``x * 2``."""
                return x * 2

        mod = _Scale()
        self.assertEqual(mod(3.0), 6.0)

    def test_rrshift_operator(self):
        """``x >> module`` invokes ``module(x)``."""

        class _AddOne(Module):
            """Module that adds one to its input."""

            def update(self, x):
                """Return ``x + 1``."""
                return x + 1

        mod = _AddOne()
        self.assertEqual(5.0 >> mod, 6.0)

    def test_pretty_repr_item_strips_underscore(self):
        """``__pretty_repr_item__`` strips a leading underscore from names."""
        mod = Module(name='abc')
        self.assertEqual(mod.__pretty_repr_item__('_name', 'abc'), ('name', 'abc'))

    def test_pretty_repr_item_hides_none_private(self):
        """A private attribute with a ``None`` value is hidden from the repr."""
        mod = Module()
        self.assertIsNone(mod.__pretty_repr_item__('_in_size', None))

    def test_pretty_repr_item_keeps_public(self):
        """A public attribute is shown unchanged in the repr."""
        mod = Module()
        self.assertEqual(mod.__pretty_repr_item__('pub', 7), ('pub', 7))

    def test_repr_contains_name(self):
        """The default ``__repr__`` includes the module name."""
        mod = Module(name='abc')
        self.assertIn('abc', repr(mod))

    def test_pretty_repr_item_hides_none_public(self):
        """A public attribute with a ``None`` value is hidden from the repr."""
        mod = Module()
        self.assertIsNone(mod.__pretty_repr_item__('w_mask', None))

    def test_pretty_repr_item_hides_invisible(self):
        """Attributes listed in ``graph_invisible_attrs`` are hidden."""

        class WithInvisible(Module):
            graph_invisible_attrs = ('secret',)

        mod = WithInvisible()
        self.assertIsNone(mod.__pretty_repr_item__('secret', 123))

    def test_repr_identifies_type_and_hides_none(self):
        """The repr names the class and omits unset (``None``) attributes."""

        class Tiny(Module):
            def __init__(self):
                super().__init__()
                self.scale = 2.0
                self.bias = None

        r = repr(Tiny())
        self.assertTrue(r.startswith('Tiny('))
        self.assertIn('scale', r)
        self.assertNotIn('bias', r)

    def test_repr_empty_module(self):
        """A module with only unset attributes renders compactly."""
        self.assertEqual(repr(Module()), 'Module()')

    def test_repr_linear_omits_none_mask(self):
        """A ``Linear`` with no weight mask does not show ``w_mask=None``."""
        r = repr(brainstate.nn.Linear(3, 4))
        self.assertIn('Linear(', r)
        self.assertNotIn('w_mask', r)


class TestModuleStateCollection(unittest.TestCase):
    """Cover ``states``/``state_trees`` collection and filtering."""

    def _make_model(self):
        """Build a small two-state module for collection tests."""

        class _M(Module):
            """Module holding one generic state and one parameter state."""

            def __init__(self):
                super().__init__()
                self.s = brainstate.State(jnp.ones(3))
                self.p = Param(jnp.ones(2), fit=True)

        return _M()

    def test_states_collects_all(self):
        """``states`` returns all states in the module."""
        mod = self._make_model()
        collected = mod.states()
        self.assertEqual(len(collected), 2)

    def test_states_filtered(self):
        """``states`` with a filter returns only matching states."""
        mod = self._make_model()
        params = mod.states(brainstate.ParamState)
        self.assertEqual(len(params), 1)

    def test_state_trees_unfiltered(self):
        """``state_trees`` returns a nested state tree."""
        mod = self._make_model()
        tree = mod.state_trees()
        # The tree should contain both leaves.
        leaves = jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, brainstate.State))
        self.assertGreaterEqual(len(leaves), 1)

    def test_state_trees_filtered(self):
        """``state_trees`` with a filter narrows the returned tree."""
        mod = self._make_model()
        tree = mod.state_trees(brainstate.ParamState)
        self.assertIsNotNone(tree)

    def test_init_and_reset_state_are_noops(self):
        """The base ``init_state``/``reset_state`` hooks do nothing and return None."""
        mod = Module()
        self.assertIsNone(mod.init_state())
        self.assertIsNone(mod.reset_state())

    def test_named_parameters_no_recurse(self):
        """``named_parameters`` with ``recurse=False`` yields only direct params."""

        class _Inner(Module):
            """Inner module with one parameter."""

            def __init__(self):
                super().__init__()
                self.w = Param(jnp.ones(2), fit=True)

        class _Outer(Module):
            """Outer module with a direct param and a child module."""

            def __init__(self):
                super().__init__()
                self.direct = Param(jnp.ones(3), fit=True)
                self.inner = _Inner()

        mod = _Outer()
        shallow = list(mod.named_parameters(recurse=False))
        deep = list(mod.named_parameters(recurse=True))
        self.assertEqual(len(shallow), 1)
        self.assertEqual(len(deep), 2)


class TestSequentialContainer(unittest.TestCase):
    """Cover ``Sequential`` construction, indexing, and mutation methods."""

    def test_basic_construction_sizes(self):
        """A ``Sequential`` infers in/out sizes from its layers."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 20),
            brainstate.nn.Linear(20, 5),
        )
        self.assertEqual(seq.in_size, (10,))
        self.assertEqual(seq.out_size, (5,))

    def test_forward(self):
        """A ``Sequential`` chains its layers when called."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(8, 4),
            jax.nn.relu,
            brainstate.nn.Linear(4, 2),
        )
        x = brainstate.random.randn(3, 8)
        out = seq(x)
        self.assertEqual(out.shape, (3, 2))

    def test_getitem_int(self):
        """Integer indexing returns the layer at that position."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(8, 4),
            brainstate.nn.Linear(4, 2),
        )
        self.assertIsInstance(seq[0], brainstate.nn.Linear)

    def test_getitem_slice(self):
        """Slice indexing returns a new ``Sequential``."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(8, 4),
            brainstate.nn.Linear(4, 2),
        )
        sub = seq[0:1]
        self.assertIsInstance(sub, Sequential)
        self.assertEqual(len(sub.layers), 1)

    def test_getitem_tuple(self):
        """Tuple indexing returns a ``Sequential`` of the selected layers."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(8, 4),
            jax.nn.relu,
            brainstate.nn.Linear(4, 2),
        )
        sub = seq[(0, 2)]
        self.assertIsInstance(sub, Sequential)
        self.assertEqual(len(sub.layers), 2)

    def test_getitem_bad_key_raises(self):
        """An unsupported index type raises ``KeyError``."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(8, 4))
        with self.assertRaises(KeyError):
            seq['bad']

    def test_append_callable(self):
        """Appending a callable wraps it and preserves the output size."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(8, 4))
        seq.append(jax.nn.relu)
        self.assertEqual(len(seq.layers), 2)
        self.assertEqual(seq.out_size, (4,))

    def test_append_module_updates_out_size(self):
        """Appending a sized module updates the output size."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(8, 4))
        seq.append(brainstate.nn.Linear(4, 2))
        self.assertEqual(seq.out_size, (2,))

    def test_extend_modules(self):
        """Extending appends multiple modules with size inference."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(8, 4))
        seq.extend([jax.nn.relu, brainstate.nn.Linear(4, 2)])
        self.assertEqual(len(seq.layers), 3)
        self.assertEqual(seq.out_size, (2,))

    def test_insert_middle(self):
        """Inserting at a middle index recalculates downstream sizes."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(8, 4),
            brainstate.nn.Linear(4, 2),
        )
        seq.insert(1, jax.nn.relu)
        self.assertEqual(len(seq.layers), 3)
        self.assertTrue(callable(seq.layers[1]))

    def test_insert_negative_index(self):
        """A negative insert index follows Python list convention."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(8, 4),
            brainstate.nn.Linear(4, 2),
        )
        seq.insert(-1, jax.nn.tanh)
        self.assertEqual(len(seq.layers), 3)

    def test_insert_out_of_range_raises(self):
        """An out-of-range insert index raises ``IndexError``."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(8, 4))
        with self.assertRaises(IndexError):
            seq.insert(5, brainstate.nn.Linear(4, 2))

    def test_describer_layer_is_built(self):
        """A ``ParamDescriber`` layer is instantiated with the inferred in_size."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 30),
            brainstate.nn.Linear.desc(out_size=5),
        )
        self.assertEqual(seq.out_size, (5,))
        self.assertIsInstance(seq.layers[1], brainstate.nn.Linear)

    def test_update_failure_wrapped(self):
        """A layer failure during ``update`` is wrapped in ``BrainStateError``."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(10, 20))
        with self.assertRaises(BrainStateError):
            seq(jnp.ones((2, 5)))  # mismatched input dimension

    def test_unsupported_layer_type_raises(self):
        """Appending an unsupported (non-callable) layer raises ``BrainStateError``."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(8, 4))
        with self.assertRaises(BrainStateError):
            seq.append(12345)


class TestSequentialEmpty(unittest.TestCase):
    """Cover ``Sequential`` mutation methods on an empty container."""

    def _empty(self):
        """Build a ``Sequential`` instance with an empty layer list."""
        seq = Sequential.__new__(Sequential)
        Module.__init__(seq)
        seq.layers = []
        return seq

    def test_append_to_empty_raises(self):
        """Appending the first layer to an empty Sequential raises ``ValueError``."""
        seq = self._empty()
        with self.assertRaises(ValueError):
            seq.append(jax.nn.relu)

    def test_extend_empty_raises(self):
        """Extending an empty Sequential raises ``ValueError``."""
        seq = self._empty()
        with self.assertRaises(ValueError):
            seq.extend([brainstate.nn.Linear(2, 2)])

    def test_insert_nonzero_index_raises(self):
        """Inserting at a non-zero index into an empty Sequential raises ``ValueError``."""
        seq = self._empty()
        with self.assertRaises(ValueError):
            seq.insert(1, brainstate.nn.Linear(2, 2))

    def test_insert_callable_first_raises(self):
        """Inserting a callable as the first layer raises ``ValueError``."""
        seq = self._empty()
        with self.assertRaises(ValueError):
            seq.insert(0, lambda x: x)

    def test_insert_module_first_ok(self):
        """Inserting a module at index 0 of an empty Sequential succeeds."""
        seq = self._empty()
        seq.insert(0, brainstate.nn.Linear(3, 4))
        self.assertEqual(len(seq.layers), 1)
        self.assertEqual(seq.in_size, (3,))
        self.assertEqual(seq.out_size, (4,))

    def test_empty_slice_returns_empty_sequential(self):
        """M6: an out-of-range slice yields an empty Sequential, not a crash."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5))
        empty = seq[10:20]
        self.assertIsInstance(empty, Sequential)
        self.assertEqual(len(empty.layers), 0)

    def test_degenerate_slice_returns_empty_sequential(self):
        """M6: a degenerate ``[i:i]`` slice yields an empty Sequential."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5))
        empty = seq[1:1]
        self.assertIsInstance(empty, Sequential)
        self.assertEqual(len(empty.layers), 0)

    def test_empty_tuple_index_returns_empty_sequential(self):
        """M6: indexing with an empty tuple yields an empty Sequential."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5))
        empty = seq[()]
        self.assertIsInstance(empty, Sequential)
        self.assertEqual(len(empty.layers), 0)

    def test_nonempty_slice_still_works(self):
        """A normal sub-slice still produces a populated Sequential."""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5), brainstate.nn.Linear(5, 6)
        )
        sub = seq[1:]
        self.assertIsInstance(sub, Sequential)
        self.assertEqual(len(sub.layers), 2)


class TestModuleSizeSetterEdges(unittest.TestCase):
    """Cover numpy-typed and invalid inputs to the size setters."""

    def test_out_size_zero_d_integer_ndarray(self):
        """M7: a 0-d integer ``np.ndarray`` ``out_size`` is normalized to a tuple."""
        mod = Module()
        mod.out_size = np.array(6)
        self.assertEqual(mod.out_size, (6,))

    def test_out_size_numpy_integer_scalar(self):
        """M7: a numpy integer scalar (``np.generic``) ``out_size`` is accepted."""
        mod = Module()
        mod.out_size = np.int64(5)
        self.assertEqual(mod.out_size, (5,))

    def test_in_size_zero_d_integer_ndarray(self):
        """M7: a 0-d integer ``np.ndarray`` ``in_size`` is normalized to a tuple."""
        mod = Module()
        mod.in_size = np.array(4)
        self.assertEqual(mod.in_size, (4,))

    def test_in_size_numpy_integer_scalar(self):
        """A numpy integer scalar (``np.generic``) ``in_size`` is accepted."""
        mod = Module()
        mod.in_size = np.int32(3)
        self.assertEqual(mod.in_size, (3,))

    def test_in_size_non_integer_generic_raises(self):
        """A non-integer ``np.generic`` ``in_size`` fails the type assertion."""
        mod = Module()
        with self.assertRaises(AssertionError):
            mod.in_size = np.float64(3.0)

    def test_out_size_non_integer_zero_d_ndarray_raises(self):
        """A non-integer 0-d ``np.ndarray`` ``out_size`` fails the type assertion."""
        mod = Module()
        with self.assertRaises(AssertionError):
            mod.out_size = np.array(3.0)


class _UnsizedModule(Module):
    """Module without declared in/out sizes, used for Sequential size tests."""

    def update(self, x):
        """Return the input unchanged."""
        return x


class TestSequentialSizeInferenceEdges(unittest.TestCase):
    """Cover Sequential construction/mutation when sizes are undefined."""

    def test_first_without_in_size(self):
        """A Sequential whose first layer has no in_size leaves in_size unset."""
        seq = brainstate.nn.Sequential(_UnsizedModule(), _UnsizedModule())
        self.assertIsNone(seq.in_size)

    def test_insert_module_into_empty_sets_sizes(self):
        """Inserting a sized module into an empty Sequential sets in/out sizes."""
        seq = Sequential.__new__(Sequential)
        Module.__init__(seq)
        seq.layers = []
        seq.insert(0, brainstate.nn.Linear(3, 7))
        self.assertEqual(seq.in_size, (3,))
        self.assertEqual(seq.out_size, (7,))

    def test_insert_module_into_empty_without_sizes(self):
        """Inserting an unsized module into an empty Sequential leaves sizes None."""
        seq = Sequential.__new__(Sequential)
        Module.__init__(seq)
        seq.layers = []
        seq.insert(0, _UnsizedModule())
        self.assertIsNone(seq.in_size)
        self.assertIsNone(seq.out_size)

    def test_insert_at_beginning_updates_in_size(self):
        """Inserting a sized module at index 0 updates the Sequential in_size."""
        seq = brainstate.nn.Sequential(brainstate.nn.Linear(5, 3))
        seq.insert(0, brainstate.nn.Linear(8, 5))
        self.assertEqual(seq.in_size, (8,))

    def test_insert_middle_unsized_keeps_none_out_size(self):
        """Inserting into an unsized chain leaves the out_size as None."""
        seq = brainstate.nn.Sequential(_UnsizedModule(), _UnsizedModule())
        seq.insert(1, _UnsizedModule())
        self.assertEqual(len(seq.layers), 3)
        self.assertIsNone(seq.out_size)

    def test_extend_with_unsized_modules(self):
        """Extending with unsized modules leaves the out_size unchanged."""
        seq = brainstate.nn.Sequential(_UnsizedModule())
        seq.extend([_UnsizedModule()])
        self.assertEqual(len(seq.layers), 2)
        self.assertIsNone(seq.out_size)

    def test_describer_without_in_size_raises(self):
        """A describer first layer with no in_size raises ``BrainStateError``."""
        seq = brainstate.nn.Sequential(_UnsizedModule())
        with self.assertRaises(BrainStateError):
            seq.append(brainstate.nn.Linear.desc(out_size=4))

    def test_elementwise_block_passthrough(self):
        """An ``ElementWiseBlock`` layer passes the size through unchanged."""

        class _EW(ElementWiseBlock):
            """Identity element-wise block."""

            def update(self, x):
                """Return the input unchanged."""
                return x

        seq = brainstate.nn.Sequential(brainstate.nn.Linear(6, 4), _EW())
        self.assertEqual(seq.out_size, (4,))


class TestModuleTrainEval(unittest.TestCase):
    """Cover train/eval behavior toggled via ``environ.context(fit=...)``."""

    def test_fit_flag_changes_behavior(self):
        """A module can branch on the ``fit`` environment flag."""

        class _FitAware(Module):
            """Module that doubles its input only during fitting."""

            def update(self, x):
                """Return ``x * 2`` when fitting, else ``x``."""
                if brainstate.environ.get('fit', desc='fit flag'):
                    return x * 2
                return x

        mod = _FitAware()
        with brainstate.environ.context(fit=True):
            _testing.assert_allclose(mod(jnp.ones(3)), jnp.ones(3) * 2)
        with brainstate.environ.context(fit=False):
            _testing.assert_allclose(mod(jnp.ones(3)), jnp.ones(3))

    def test_dropout_eval_is_identity(self):
        """Dropout passes inputs through unchanged when not fitting."""
        drop = brainstate.nn.Dropout(0.5)
        x = brainstate.random.randn(64)
        with brainstate.environ.context(fit=False):
            out = drop(x)
        _testing.assert_allclose(out, x)


if __name__ == '__main__':
    unittest.main()
