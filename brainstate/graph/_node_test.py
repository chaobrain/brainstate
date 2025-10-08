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
from brainstate._state import State
from brainstate.graph._node import (
    Node,
    _node_flatten,
    _node_set_key,
    _node_pop_key,
    _node_create_empty,
    _node_clear
)


class TestNode(unittest.TestCase):
    """Test suite for the Node class."""

    def test_node_creation(self):
        """Test basic node creation."""

        class SimpleNode(Node):
            def __init__(self, value):
                self.value = value

        node = SimpleNode(10)
        self.assertEqual(node.value, 10)
        self.assertIsInstance(node, Node)

    def test_node_subclass_registration(self):
        """Test that Node subclasses are automatically registered."""

        class TestNode(Node):
            pass

        # The subclass should be registered automatically via __init_subclass__
        node = TestNode()
        self.assertIsInstance(node, Node)

    def test_graph_invisible_attrs(self):
        """Test that graph_invisible_attrs works correctly."""

        class NodeWithInvisible(Node):
            graph_invisible_attrs = ('_private', '_internal')

            def __init__(self):
                self.public = 1
                self._private = 2
                self._internal = 3

        node = NodeWithInvisible()
        flattened, static = _node_flatten(node)

        # Check that only public attribute is in flattened
        keys = [k for k, v in flattened]
        self.assertIn('public', keys)
        self.assertNotIn('_private', keys)
        self.assertNotIn('_internal', keys)

    def test_deepcopy_using_treefy(self):
        """Test deep copying of nodes using treefy_split/merge."""

        class NodeWithData(Node):
            def __init__(self, data=None):
                if data is not None:
                    self.data = data
                    self.nested = {'a': 1, 'b': [2, 3]}

        original = NodeWithData([1, 2, 3])

        # Use treefy_split and treefy_merge to copy
        graphdef, state = brainstate.graph.treefy_split(original)
        # Create a new instance using treefy_merge
        copied = brainstate.graph.treefy_merge(graphdef, state)

        # Check that it's a different object
        self.assertIsNot(original, copied)

        # Check that data is present
        self.assertEqual(original.data, copied.data)

        # Modify copied data shouldn't affect original
        copied.data.append(4)
        self.assertEqual(len(original.data), 3)
        self.assertEqual(len(copied.data), 4)

    def test_node_with_state(self):
        """Test nodes containing State objects."""

        class NodeWithState(Node):
            def __init__(self):
                self.value = State(10)
                self.normal = 20

        node = NodeWithState()
        self.assertIsInstance(node.value, State)
        self.assertEqual(node.value.value, 10)
        self.assertEqual(node.normal, 20)

    def test_complex_nested_structure(self):
        """Test nodes with complex nested structures using treefy."""

        class ComplexNode(Node):
            def __init__(self):
                self.list_data = [1, 2, [3, 4]]
                self.dict_data = {'a': 1, 'b': {'c': 2}}
                self.tuple_data = (1, 2, (3, 4))

        node = ComplexNode()

        # Test using treefy_split and merge
        graphdef, state = brainstate.graph.treefy_split(node)
        copied = brainstate.graph.treefy_merge(graphdef, state)

        self.assertEqual(node.list_data, copied.list_data)
        self.assertEqual(node.dict_data, copied.dict_data)
        self.assertEqual(node.tuple_data, copied.tuple_data)


class TestNodeHelperFunctions(unittest.TestCase):
    """Test suite for node helper functions."""

    def test_node_flatten(self):
        """Test _node_flatten function."""

        class TestNode(Node):
            def __init__(self):
                self.b = 2
                self.a = 1
                self.c = 3

        node = TestNode()
        flattened, static = _node_flatten(node)

        # Check that attributes are sorted
        keys = [k for k, v in flattened]
        self.assertEqual(keys, ['a', 'b', 'c'])

        # Check values
        values = [v for k, v in flattened]
        self.assertEqual(values, [1, 2, 3])

        # Check static contains type
        self.assertEqual(static, (TestNode,))

    def test_node_flatten_with_invisible(self):
        """Test _node_flatten with invisible attributes."""

        class TestNode(Node):
            graph_invisible_attrs = ('hidden',)

            def __init__(self):
                self.visible = 1
                self.hidden = 2

        node = TestNode()
        flattened, static = _node_flatten(node)

        keys = [k for k, v in flattened]
        self.assertIn('visible', keys)
        self.assertNotIn('hidden', keys)

    def test_node_set_key_simple(self):
        """Test _node_set_key with simple values."""

        class TestNode(Node):
            pass

        node = TestNode()
        _node_set_key(node, 'attr', 10)
        self.assertEqual(node.attr, 10)

        _node_set_key(node, 'attr', 20)
        self.assertEqual(node.attr, 20)

    def test_node_set_key_with_state(self):
        """Test _node_set_key with State objects."""

        class TestNode(Node):
            def __init__(self):
                self.state_attr = State(10)

        node = TestNode()

        # Test setting a regular value
        _node_set_key(node, 'regular_attr', 30)
        self.assertEqual(node.regular_attr, 30)

        # Test updating with a TreefyState
        # We'll use the real TreefyState from graph operations
        graphdef, states = brainstate.graph.treefy_split(node)

        # The states should contain our State object wrapped as TreefyState
        # When setting with TreefyState, it should update the existing State
        initial_state = node.state_attr

        # Create a new node and try to set the TreefyState
        new_node = TestNode()
        for key, value in states.to_flat().items():
            if 'state_attr' in key:
                _node_set_key(new_node, 'state_attr', value)
                # The State object should be updated via update_from_ref
                self.assertIsInstance(new_node.state_attr, State)

    def test_node_set_key_invalid_key(self):
        """Test _node_set_key with invalid key."""

        class TestNode(Node):
            pass

        node = TestNode()

        with self.assertRaises(KeyError) as context:
            _node_set_key(node, 123, 'value')
        self.assertIn('Invalid key', str(context.exception))

    def test_node_pop_key(self):
        """Test _node_pop_key function."""

        class TestNode(Node):
            def __init__(self):
                self.attr1 = 10
                self.attr2 = 20

        node = TestNode()

        # Pop existing attribute
        value = _node_pop_key(node, 'attr1')
        self.assertEqual(value, 10)
        self.assertFalse(hasattr(node, 'attr1'))
        self.assertTrue(hasattr(node, 'attr2'))

    def test_node_pop_key_invalid(self):
        """Test _node_pop_key with invalid key."""

        class TestNode(Node):
            pass

        node = TestNode()

        # Invalid key type
        with self.assertRaises(KeyError) as context:
            _node_pop_key(node, 123)
        self.assertIn('Invalid key', str(context.exception))

        # Non-existent key
        with self.assertRaises(KeyError):
            _node_pop_key(node, 'nonexistent')

    def test_node_create_empty(self):
        """Test _node_create_empty function."""

        class TestNode(Node):
            def __init__(self, value=None):
                self.value = value
                self.initialized = True

        # Create empty node
        node = _node_create_empty((TestNode,))

        # Check it's the right type
        self.assertIsInstance(node, TestNode)

        # Check __init__ was not called
        self.assertFalse(hasattr(node, 'value'))
        self.assertFalse(hasattr(node, 'initialized'))

    def test_node_clear(self):
        """Test _node_clear function."""

        class TestNode(Node):
            def __init__(self):
                self.attr1 = 10
                self.attr2 = 20
                self.attr3 = [1, 2, 3]

        node = TestNode()

        # Verify attributes exist
        self.assertTrue(hasattr(node, 'attr1'))
        self.assertTrue(hasattr(node, 'attr2'))
        self.assertTrue(hasattr(node, 'attr3'))

        # Clear the node
        _node_clear(node)

        # Verify attributes are gone
        self.assertFalse(hasattr(node, 'attr1'))
        self.assertFalse(hasattr(node, 'attr2'))
        self.assertFalse(hasattr(node, 'attr3'))

        # Verify node still exists and is valid
        self.assertIsInstance(node, TestNode)


class TestNodeIntegration(unittest.TestCase):
    """Integration tests for Node with the graph system."""

    def test_node_with_nested_nodes(self):
        """Test nodes containing other nodes."""

        class ChildNode(Node):
            def __init__(self, value=None):
                if value is not None:
                    self.value = value

        class ParentNode(Node):
            def __init__(self):
                self.child1 = ChildNode(10)
                self.child2 = ChildNode(20)
                self.data = [1, 2, 3]

        parent = ParentNode()

        # Test using treefy_split and merge
        graphdef, state = brainstate.graph.treefy_split(parent)
        copied = brainstate.graph.treefy_merge(graphdef, state)

        self.assertIsNot(parent.child1, copied.child1)
        self.assertEqual(parent.child1.value, copied.child1.value)

    def test_node_with_list_of_nodes(self):
        """Test nodes containing lists of other nodes."""

        class ItemNode(Node):
            def __init__(self, id=None):
                if id is not None:
                    self.id = id

        class ContainerNode(Node):
            def __init__(self):
                self.items = [ItemNode(i) for i in range(3)]

        container = ContainerNode()
        graphdef, state = brainstate.graph.treefy_split(container)
        copied = brainstate.graph.treefy_merge(graphdef, state)

        self.assertEqual(len(container.items), len(copied.items))
        for orig, cp in zip(container.items, copied.items):
            self.assertIsNot(orig, cp)
            self.assertEqual(orig.id, cp.id)

    def test_node_with_dict_of_nodes(self):
        """Test nodes containing dictionaries of other nodes."""

        class ValueNode(Node):
            def __init__(self, value=None):
                if value is not None:
                    self.value = value

        class DictNode(Node):
            def __init__(self):
                self.mapping = {
                    'a': ValueNode(1),
                    'b': ValueNode(2),
                    'c': ValueNode(3)
                }

        node = DictNode()
        graphdef, state = brainstate.graph.treefy_split(node)
        copied = brainstate.graph.treefy_merge(graphdef, state)

        self.assertEqual(set(node.mapping.keys()), set(copied.mapping.keys()))
        for key in node.mapping:
            self.assertIsNot(node.mapping[key], copied.mapping[key])
            self.assertEqual(node.mapping[key].value, copied.mapping[key].value)


class TestStateRetrieve(unittest.TestCase):
    """Tests for state retrieval from nodes."""

    def test_list_of_states_1(self):
        """Test retrieving states from a list."""

        class Model(brainstate.graph.Node):
            def __init__(self):
                self.a = [1, 2, 3]
                self.b = [brainstate.State(1), brainstate.State(2), brainstate.State(3)]

        m = Model()
        graphdef, states = brainstate.graph.treefy_split(m)
        print(states.to_flat())
        self.assertTrue(len(states.to_flat()) == 3)

    def test_list_of_states_2(self):
        """Test retrieving states from nested lists."""

        class Model(brainstate.graph.Node):
            def __init__(self):
                self.a = [1, 2, 3]
                self.b = [brainstate.State(1), [brainstate.State(2), brainstate.State(3)]]

        m = Model()
        graphdef, states = brainstate.graph.treefy_split(m)
        print(states.to_flat())
        self.assertTrue(len(states.to_flat()) == 3)

    def test_list_of_node_1(self):
        """Test retrieving states from a list of nodes."""

        class Model(brainstate.graph.Node):
            def __init__(self):
                self.a = [1, 2, 3]
                self.b = [brainstate.nn.Linear(1, 2), brainstate.nn.Linear(2, 3)]

        m = Model()
        graphdef, states = brainstate.graph.treefy_split(m)
        print(states.to_flat())
        self.assertTrue(len(states.to_flat()) == 2)

    def test_list_of_node_2(self):
        """Test retrieving states from nested structures of nodes."""

        class Model(brainstate.graph.Node):
            def __init__(self):
                self.a = [1, 2, 3]
                self.b = [brainstate.nn.Linear(1, 2), [brainstate.nn.Linear(2, 3)],
                          (brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5))]

        m = Model()
        graphdef, states = brainstate.graph.treefy_split(m)
        print(states.to_flat())
        self.assertTrue(len(states.to_flat()) == 4)

    def test_mixed_states_and_nodes(self):
        """Test nodes with mixed states and sub-nodes."""

        class Model(brainstate.graph.Node):
            def __init__(self):
                self.state1 = brainstate.State(1.0)
                self.state2 = brainstate.State(2.0)
                self.linear = brainstate.nn.Linear(5, 10)
                self.data = [1, 2, 3]

        m = Model()
        graphdef, states = brainstate.graph.treefy_split(m)

        # Should have states from both direct State objects and Linear layer
        flat_states = states.to_flat()
        self.assertGreaterEqual(len(flat_states), 2)  # At least the two direct states


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_node(self):
        """Test node with no attributes."""

        class EmptyNode(Node):
            pass

        node = EmptyNode()
        flattened, static = _node_flatten(node)

        self.assertEqual(len(flattened), 0)
        self.assertEqual(static, (EmptyNode,))

    def test_node_with_none_values(self):
        """Test node with None values."""

        class NoneNode(Node):
            def __init__(self):
                self.none_val = None
                self.real_val = 10

        node = NoneNode()
        flattened, static = _node_flatten(node)

        values_dict = dict(flattened)
        self.assertIsNone(values_dict['none_val'])
        self.assertEqual(values_dict['real_val'], 10)

    def test_node_with_special_attributes(self):
        """Test node with special Python attributes."""

        class SpecialNode(Node):
            def __init__(self):
                self.__dict__['special'] = 'value'
                self.normal = 'normal'

        node = SpecialNode()
        self.assertEqual(node.special, 'value')
        self.assertEqual(node.normal, 'normal')

    def test_circular_reference(self):
        """Test handling of circular references."""

        class CircularNode(Node):
            pass

        node1 = CircularNode()
        node2 = CircularNode()
        node1.ref = node2
        node2.ref = node1

        # This should not cause infinite recursion with treefy
        try:
            graphdef, state = brainstate.graph.treefy_split(node1)
            copied = brainstate.graph.treefy_merge(graphdef, state)
            # Check that circular reference is preserved
            self.assertIs(copied.ref.ref, copied)
        except RecursionError:
            self.fail("Treefy failed with circular reference")

    def test_node_inheritance(self):
        """Test node inheritance hierarchy."""

        class BaseNode(Node):
            def __init__(self):
                self.base_attr = 'base'

        class DerivedNode(BaseNode):
            def __init__(self):
                super().__init__()
                self.derived_attr = 'derived'

        node = DerivedNode()
        self.assertEqual(node.base_attr, 'base')
        self.assertEqual(node.derived_attr, 'derived')

        flattened, static = _node_flatten(node)
        keys = [k for k, v in flattened]
        self.assertIn('base_attr', keys)
        self.assertIn('derived_attr', keys)

    def test_node_with_property(self):
        """Test node with property decorators."""

        class PropertyNode(Node):
            def __init__(self):
                self._value = 10

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, val):
                self._value = val

        node = PropertyNode()
        self.assertEqual(node.value, 10)

        node.value = 20
        self.assertEqual(node.value, 20)

        # Only _value should appear in flattened
        flattened, static = _node_flatten(node)
        keys = [k for k, v in flattened]
        self.assertIn('_value', keys)

    def test_multiple_inheritance(self):
        """Test node with multiple inheritance."""

        class Mixin:
            def mixin_method(self):
                return 'mixin'

        class MultiNode(Node, Mixin):
            def __init__(self):
                self.data = 'data'

        node = MultiNode()
        self.assertEqual(node.mixin_method(), 'mixin')
        self.assertEqual(node.data, 'data')

        # Test that it still works as a Node with treefy
        graphdef, state = brainstate.graph.treefy_split(node)
        copied = brainstate.graph.treefy_merge(graphdef, state)
        self.assertIsNot(node, copied)
        self.assertEqual(copied.data, 'data')


if __name__ == '__main__':
    unittest.main()
