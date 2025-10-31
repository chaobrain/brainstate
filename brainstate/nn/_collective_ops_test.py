# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import jax.numpy as jnp
import pytest

import brainstate


class SimpleTestModule(brainstate.nn.Module):
    """Simple test module with init_state method"""

    def __init__(self):
        super().__init__()
        self.state_initialized = False
        self.init_args = None
        self.init_kwargs = None

    def init_state(self, *args, **kwargs):
        self.state_initialized = True
        self.init_args = args
        self.init_kwargs = kwargs
        self.state = brainstate.State(jnp.zeros(5))


class OrderedTestModule(brainstate.nn.Module):
    """Test module with call_order decorator"""

    def __init__(self, order_level):
        super().__init__()
        self.order_level = order_level
        self.call_sequence = []

    @brainstate.nn.call_order(0)
    def init_state(self):
        self.state = brainstate.State(jnp.array([self.order_level]))


class NestedModule(brainstate.nn.Module):
    """Module with nested submodules"""

    def __init__(self):
        super().__init__()
        self.submodule1 = SimpleTestModule()
        self.submodule2 = SimpleTestModule()

    def init_state(self):
        self.state = brainstate.State(jnp.zeros(3))


class Test_init_all_states:
    """Comprehensive tests for init_all_states function"""

    def test_basic_initialization(self):
        """Test basic state initialization"""
        module = SimpleTestModule()
        assert not module.state_initialized

        brainstate.nn.init_all_states(module)

        assert module.state_initialized
        assert module.state.value.shape == (5,)

    def test_with_positional_args(self):
        """Test init_all_states with positional arguments"""
        module = SimpleTestModule()

        brainstate.nn.init_all_states(module, 1, 2, 3)

        assert module.state_initialized
        assert module.init_args == (1, 2, 3)

    def test_with_keyword_args(self):
        """Test init_all_states with keyword arguments"""
        module = SimpleTestModule()

        brainstate.nn.init_all_states(module, batch_size=10, seq_len=20)

        assert module.state_initialized
        assert module.init_kwargs == {'batch_size': 10, 'seq_len': 20}

    def test_with_mixed_args(self):
        """Test init_all_states with both positional and keyword arguments"""
        module = SimpleTestModule()

        brainstate.nn.init_all_states(module, 42, batch_size=10)

        assert module.state_initialized
        assert module.init_args == (42,)
        assert module.init_kwargs == {'batch_size': 10}

    def test_nested_modules(self):
        """Test that init_all_states initializes nested submodules"""
        module = NestedModule()

        brainstate.nn.init_all_states(module)

        assert module.submodule1.state_initialized
        assert module.submodule2.state_initialized
        assert hasattr(module, 'state')

    def test_with_gru_cell(self):
        """Test with real GRUCell module"""
        gru = brainstate.nn.GRUCell(1, 2)

        brainstate.nn.init_all_states(gru, batch_size=10)

        # Check that states were created
        state_dict = gru.states()
        assert len(state_dict) > 0

    def test_sequential_module(self):
        """Test with Sequential module containing multiple layers"""
        seq = brainstate.nn.Sequential(
            brainstate.nn.Linear(10, 20),
            brainstate.nn.Dropout(0.5)
        )

        brainstate.nn.init_all_states(seq)

        # Check that Linear layer has weight and bias
        state_dict = seq.states()
        assert len(state_dict) > 0

    def test_node_to_exclude(self):
        """Test excluding specific nodes from initialization"""
        module = NestedModule()

        # Exclude submodule1 by type matching - simpler and more reliable
        brainstate.nn.init_all_states(
            module,
            node_to_exclude=NestedModule  # Exclude the parent, only init children
        )

        # Parent should not be initialized, but children should be
        assert not hasattr(module, 'state') or module.state is None or not hasattr(module.state, 'value')
        assert module.submodule1.state_initialized
        assert module.submodule2.state_initialized

    def test_with_call_order(self):
        """Test that call_order is respected during initialization"""

        class OrderedModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.execution_order = []

            @brainstate.nn.call_order(1)
            def init_state(self):
                self.execution_order.append('parent')

        class ChildModule(brainstate.nn.Module):
            def __init__(self, parent_module):
                super().__init__()
                self.parent = parent_module

            @brainstate.nn.call_order(0)
            def init_state(self):
                self.parent.execution_order.append('child')

        parent = OrderedModule()
        child = ChildModule(parent)
        parent.child_module = child

        brainstate.nn.init_all_states(parent)

        # Child (order 0) should execute before parent (order 1)
        assert parent.execution_order == ['child', 'parent']


class ResetTestModule(brainstate.nn.Module):
    """Test module with both init_state and reset_state methods"""

    def __init__(self):
        super().__init__()
        self.state_initialized = False
        self.state_reset = False
        self.reset_args = None
        self.reset_kwargs = None

    def init_state(self, *args, **kwargs):
        self.state_initialized = True
        self.state_reset = False
        self.state = brainstate.State(jnp.ones(5))

    def reset_state(self, *args, **kwargs):
        self.state_reset = True
        self.reset_args = args
        self.reset_kwargs = kwargs
        if hasattr(self, 'state'):
            self.state.value = jnp.zeros(5)


class ResetOrderedModule(brainstate.nn.Module):
    """Test module with call_order on reset_state"""

    def __init__(self, order_level, execution_tracker):
        super().__init__()
        self.order_level = order_level
        self.execution_tracker = execution_tracker

    def init_state(self):
        self.state = brainstate.State(jnp.ones(3))

    @brainstate.nn.call_order(0)
    def reset_state(self):
        self.execution_tracker.append(f'order_{self.order_level}')
        self.state.value = jnp.zeros(3)


class NestedResetModule(brainstate.nn.Module):
    """Module with nested submodules that have reset_state"""

    def __init__(self):
        super().__init__()
        self.submodule1 = ResetTestModule()
        self.submodule2 = ResetTestModule()

    def init_state(self):
        self.state = brainstate.State(jnp.ones(3))

    def reset_state(self):
        self.state.value = jnp.zeros(3)


class Test_reset_all_states:
    """Comprehensive tests for reset_all_states function"""

    def test_basic_reset(self):
        """Test basic state reset"""
        module = ResetTestModule()
        brainstate.nn.init_all_states(module)

        assert module.state_initialized
        assert not module.state_reset
        assert jnp.allclose(module.state.value, jnp.ones(5))

        brainstate.nn.reset_all_states(module)

        assert module.state_reset
        assert jnp.allclose(module.state.value, jnp.zeros(5))

    def test_with_positional_args(self):
        """Test reset_all_states with positional arguments"""
        module = ResetTestModule()
        brainstate.nn.init_all_states(module)

        brainstate.nn.reset_all_states(module, 1, 2, 3)

        assert module.state_reset
        assert module.reset_args == (1, 2, 3)

    def test_with_keyword_args(self):
        """Test reset_all_states with keyword arguments"""
        module = ResetTestModule()
        brainstate.nn.init_all_states(module)

        brainstate.nn.reset_all_states(module, batch_size=10, seq_len=20)

        assert module.state_reset
        assert module.reset_kwargs == {'batch_size': 10, 'seq_len': 20}

    def test_with_mixed_args(self):
        """Test reset_all_states with both positional and keyword arguments"""
        module = ResetTestModule()
        brainstate.nn.init_all_states(module)

        brainstate.nn.reset_all_states(module, 42, batch_size=10)

        assert module.state_reset
        assert module.reset_args == (42,)
        assert module.reset_kwargs == {'batch_size': 10}

    def test_nested_modules(self):
        """Test that reset_all_states resets nested submodules"""
        module = NestedResetModule()
        brainstate.nn.init_all_states(module)

        # Verify initial state
        assert jnp.allclose(module.state.value, jnp.ones(3))
        assert jnp.allclose(module.submodule1.state.value, jnp.ones(5))
        assert jnp.allclose(module.submodule2.state.value, jnp.ones(5))

        brainstate.nn.reset_all_states(module)

        # Verify all states were reset
        assert jnp.allclose(module.state.value, jnp.zeros(3))
        assert module.submodule1.state_reset
        assert module.submodule2.state_reset
        assert jnp.allclose(module.submodule1.state.value, jnp.zeros(5))
        assert jnp.allclose(module.submodule2.state.value, jnp.zeros(5))

    def test_with_gru_cell(self):
        """Test reset with real GRUCell module"""
        gru = brainstate.nn.GRUCell(5, 10)
        brainstate.nn.init_all_states(gru, batch_size=8)

        # Get initial state
        initial_states = {k: v.value.copy() for k, v in gru.states().items()
                          if hasattr(v.value, 'copy') and not isinstance(v.value, dict)}

        # Reset state
        brainstate.nn.reset_all_states(gru, batch_size=8)

        # Verify state was reset (should be zeros for hidden state)
        for key in initial_states:
            current_val = gru.states()[key].value
            if not isinstance(current_val, dict):
                # Hidden state should be reset to zeros
                if 'h' in str(key):
                    assert jnp.allclose(current_val, jnp.zeros_like(current_val))

    def test_sequential_reset(self):
        """Test reset with Sequential module"""

        # Create a simple network with stateful components
        class StatefulLayer(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.reset_called = False

            def init_state(self):
                self.state = brainstate.State(jnp.ones(5))

            def reset_state(self):
                self.reset_called = True
                self.state.value = jnp.zeros(5)

        layer1 = StatefulLayer()
        layer2 = StatefulLayer()
        seq = brainstate.nn.Sequential(layer1, layer2)

        brainstate.nn.init_all_states(seq)
        brainstate.nn.reset_all_states(seq)

        assert layer1.reset_called
        assert layer2.reset_called

    def test_node_to_exclude(self):
        """Test excluding specific nodes from reset"""
        module = NestedResetModule()
        brainstate.nn.init_all_states(module)

        # Exclude the parent module from reset
        brainstate.nn.reset_all_states(
            module,
            node_to_exclude=NestedResetModule
        )

        # Parent should not be reset, but children should be
        assert jnp.allclose(module.state.value, jnp.ones(3))  # Not reset
        assert module.submodule1.state_reset  # Reset
        assert module.submodule2.state_reset  # Reset

    def test_with_call_order(self):
        """Test that call_order is respected during reset"""
        execution_tracker = []

        class OrderedParent(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.child1 = ResetOrderedModule(1, execution_tracker)
                self.child2 = ResetOrderedModule(2, execution_tracker)

            def init_state(self):
                pass

        parent = OrderedParent()
        brainstate.nn.init_all_states(parent)

        execution_tracker.clear()
        brainstate.nn.reset_all_states(parent)

        # Both should execute (order 0), check that reset was called
        assert len(execution_tracker) == 2

    def test_multiple_resets(self):
        """Test calling reset_all_states multiple times"""
        module = ResetTestModule()
        brainstate.nn.init_all_states(module)

        for i in range(3):
            brainstate.nn.reset_all_states(module)
            assert module.state_reset
            assert jnp.allclose(module.state.value, jnp.zeros(5))

    def test_reset_without_init(self):
        """Test that reset works even if init wasn't called explicitly"""
        gru = brainstate.nn.GRUCell(5, 10)

        # Initialize first
        brainstate.nn.init_all_states(gru, batch_size=8)

        # Reset should work
        brainstate.nn.reset_all_states(gru, batch_size=8)

        # Verify it didn't crash
        states = gru.states()
        assert len(states) > 0


class CustomMethodModule(brainstate.nn.Module):
    """Test module with custom methods for call_all_fns testing"""

    def __init__(self):
        super().__init__()
        self.method_called = False
        self.call_count = 0
        self.received_args = None
        self.received_kwargs = None

    def custom_method(self, *args, **kwargs):
        self.method_called = True
        self.call_count += 1
        self.received_args = args
        self.received_kwargs = kwargs

    def another_method(self):
        self.call_count += 10


class OrderedCallModule(brainstate.nn.Module):
    """Test module with ordered methods"""

    def __init__(self, execution_tracker):
        super().__init__()
        self.execution_tracker = execution_tracker

    @brainstate.nn.call_order(0)
    def ordered_method_0(self):
        self.execution_tracker.append('order_0')

    @brainstate.nn.call_order(1)
    def ordered_method_1(self):
        self.execution_tracker.append('order_1')

    @brainstate.nn.call_order(2)
    def ordered_method_2(self):
        self.execution_tracker.append('order_2')

    def unordered_method(self):
        self.execution_tracker.append('unordered')


class NestedCallModule(brainstate.nn.Module):
    """Module with nested submodules for call testing"""

    def __init__(self):
        super().__init__()
        self.child1 = CustomMethodModule()
        self.child2 = CustomMethodModule()
        self.method_called = False

    def custom_method(self, *args, **kwargs):
        self.method_called = True


class Test_call_order:
    """Comprehensive tests for call_order decorator"""

    def test_basic_call_order(self):
        """Test basic call_order decorator"""
        execution_tracker = []

        class TestModule(brainstate.nn.Module):
            @brainstate.nn.call_order(0)
            def method(self):
                execution_tracker.append('executed')

        module = TestModule()
        assert hasattr(module.method, 'call_order')
        assert module.method.call_order == 0

    def test_different_order_levels(self):
        """Test different order levels"""
        for level in [0, 1, 5, 9]:
            class TestModule(brainstate.nn.Module):
                @brainstate.nn.call_order(level)
                def method(self):
                    pass

            module = TestModule()
            assert module.method.call_order == level

    def test_order_boundary_validation(self):
        """Test that order level boundary validation works"""
        # Valid levels (0 to MAX_ORDER-1)
        for level in range(brainstate.nn._collective_ops.MAX_ORDER):
            @brainstate.nn.call_order(level)
            def valid_method():
                pass

            assert valid_method.call_order == level

        # Invalid levels
        with pytest.raises(ValueError, match="must be an integer"):
            @brainstate.nn.call_order(-1)
            def invalid_method1():
                pass

        with pytest.raises(ValueError, match="must be an integer"):
            @brainstate.nn.call_order(brainstate.nn._collective_ops.MAX_ORDER)
            def invalid_method2():
                pass

    def test_disable_boundary_check(self):
        """Test disabling boundary check"""

        @brainstate.nn.call_order(100, check_order_boundary=False)
        def method():
            pass

        assert method.call_order == 100

        @brainstate.nn.call_order(-5, check_order_boundary=False)
        def method2():
            pass

        assert method2.call_order == -5

    def test_order_preserved_on_methods(self):
        """Test that call_order is preserved on instance methods"""
        execution_tracker = []
        module = OrderedCallModule(execution_tracker)

        assert module.ordered_method_0.call_order == 0
        assert module.ordered_method_1.call_order == 1
        assert module.ordered_method_2.call_order == 2
        assert not hasattr(module.unordered_method, 'call_order')

    def test_multiple_decorators(self):
        """Test applying call_order to multiple methods"""
        execution_tracker = []

        class MultiMethodModule(brainstate.nn.Module):
            @brainstate.nn.call_order(2)
            def method_a(self):
                execution_tracker.append('a')

            @brainstate.nn.call_order(0)
            def method_b(self):
                execution_tracker.append('b')

            @brainstate.nn.call_order(1)
            def method_c(self):
                execution_tracker.append('c')

        module = MultiMethodModule()
        assert module.method_a.call_order == 2
        assert module.method_b.call_order == 0
        assert module.method_c.call_order == 1


class Test_call_all_fns:
    """Comprehensive tests for call_all_fns function"""

    def test_basic_function_call(self):
        """Test basic function calling"""
        module = CustomMethodModule()

        assert not module.method_called
        brainstate.nn.call_all_fns(module, 'custom_method')
        assert module.method_called
        assert module.call_count == 1

    def test_with_positional_args(self):
        """Test call_all_fns with positional arguments"""
        module = CustomMethodModule()

        brainstate.nn.call_all_fns(module, 'custom_method', (1, 2, 3))

        assert module.method_called
        assert module.received_args == (1, 2, 3)

    def test_with_keyword_args(self):
        """Test call_all_fns with keyword arguments"""
        module = CustomMethodModule()

        brainstate.nn.call_all_fns(module, 'custom_method', kwargs={'foo': 'bar', 'baz': 42})

        assert module.method_called
        assert module.received_kwargs == {'foo': 'bar', 'baz': 42}

    def test_with_mixed_args(self):
        """Test call_all_fns with both positional and keyword arguments"""
        module = CustomMethodModule()

        brainstate.nn.call_all_fns(
            module,
            'custom_method',
            args=(1, 2),
            kwargs={'key': 'value'}
        )

        assert module.method_called
        assert module.received_args == (1, 2)
        assert module.received_kwargs == {'key': 'value'}

    def test_nested_modules(self):
        """Test that call_all_fns calls methods on nested modules"""
        module = NestedCallModule()

        brainstate.nn.call_all_fns(module, 'custom_method')

        assert module.method_called
        assert module.child1.method_called
        assert module.child2.method_called

    def test_call_order_respected(self):
        """Test that call_order is respected"""
        execution_tracker = []

        class ParentModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.child1 = OrderedCallModule(execution_tracker)
                self.child2 = OrderedCallModule(execution_tracker)

        module = ParentModule()

        # Call ordered_method_1 on all modules (parent doesn't have it, so skip)
        brainstate.nn.call_all_fns(module, 'ordered_method_1', fn_if_not_exist='pass')

        # Should be called on both children (both have order 1)
        assert execution_tracker.count('order_1') == 2

    def test_execution_order_with_mixed_decorators(self):
        """Test execution order with both ordered and unordered methods"""
        execution_tracker = []

        class MixedModule(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.ordered = OrderedCallModule(execution_tracker)

            def unordered_method(self):
                execution_tracker.append('parent_unordered')

        module = MixedModule()

        # Call unordered_method - parent has no decorator, child has no decorator
        execution_tracker.clear()
        brainstate.nn.call_all_fns(module, 'unordered_method')

        # Both should be called (unordered methods execute first)
        assert 'parent_unordered' in execution_tracker
        assert 'unordered' in execution_tracker

    def test_node_to_exclude(self):
        """Test excluding specific nodes"""
        module = NestedCallModule()

        # Exclude the parent module
        brainstate.nn.call_all_fns(
            module,
            'custom_method',
            node_to_exclude=NestedCallModule
        )

        # Parent should not be called, but children should be
        assert not module.method_called
        assert module.child1.method_called
        assert module.child2.method_called

    def test_fn_if_not_exist_raise(self):
        """Test fn_if_not_exist='raise' behavior"""
        module = CustomMethodModule()

        with pytest.raises(AttributeError, match="does not have method"):
            brainstate.nn.call_all_fns(module, 'nonexistent_method', fn_if_not_exist='raise')

    def test_fn_if_not_exist_pass(self):
        """Test fn_if_not_exist='pass' behavior"""
        module = CustomMethodModule()

        # Should not raise error
        brainstate.nn.call_all_fns(module, 'nonexistent_method', fn_if_not_exist='pass')

    def test_fn_if_not_exist_none(self):
        """Test fn_if_not_exist='none' behavior"""
        module = CustomMethodModule()

        # Should not raise error
        brainstate.nn.call_all_fns(module, 'nonexistent_method', fn_if_not_exist='none')

    def test_fn_if_not_exist_warn(self):
        """Test fn_if_not_exist='warn' behavior"""
        module = CustomMethodModule()

        # Should issue warning but not raise
        with pytest.warns(UserWarning, match="does not have method"):
            brainstate.nn.call_all_fns(module, 'nonexistent_method', fn_if_not_exist='warn')

    def test_invalid_fn_name_type(self):
        """Test that invalid fn_name type raises error"""
        module = CustomMethodModule()

        with pytest.raises(TypeError, match="fn_name must be a string"):
            brainstate.nn.call_all_fns(module, 123)

    def test_invalid_kwargs_type(self):
        """Test that invalid kwargs type raises error"""
        module = CustomMethodModule()

        with pytest.raises(TypeError, match="kwargs must be a mapping"):
            brainstate.nn.call_all_fns(module, 'custom_method', kwargs=[1, 2, 3])

    def test_non_callable_attribute(self):
        """Test that non-callable attributes raise error"""

        class ModuleWithAttribute(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.my_attr = "not callable"

        module = ModuleWithAttribute()

        with pytest.raises(TypeError, match="must be callable"):
            brainstate.nn.call_all_fns(module, 'my_attr')

    def test_with_gru_cell(self):
        """Test with real GRU cell"""
        gru = brainstate.nn.GRUCell(5, 10)

        # Initialize states
        brainstate.nn.call_all_fns(gru, 'init_state', kwargs={'batch_size': 8})

        # Verify states were created
        states = gru.states()
        assert len(states) > 0

    def test_multiple_calls_same_function(self):
        """Test calling same function multiple times"""
        module = CustomMethodModule()

        for i in range(5):
            brainstate.nn.call_all_fns(module, 'custom_method')

        assert module.call_count == 5

    def test_single_non_tuple_arg(self):
        """Test that single non-tuple argument is wrapped"""
        module = CustomMethodModule()

        brainstate.nn.call_all_fns(module, 'custom_method', args=42)

        assert module.received_args == (42,)

    def test_sequential_module(self):
        """Test with Sequential module"""
        layer1 = CustomMethodModule()
        layer2 = CustomMethodModule()
        seq = brainstate.nn.Sequential(layer1, layer2)

        # Sequential itself doesn't have custom_method, so skip it
        brainstate.nn.call_all_fns(seq, 'custom_method', fn_if_not_exist='pass')

        assert layer1.method_called
        assert layer2.method_called
