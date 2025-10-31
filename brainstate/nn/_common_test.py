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

import unittest
from unittest.mock import Mock, patch

import jax.numpy as jnp

import brainstate
from brainstate import environ
from brainstate.nn import Module, EnvironContext
from brainstate.nn._common import _filter_states


class DummyModule(Module):
    """A simple module for testing purposes."""

    def __init__(self, value=0):
        super().__init__()
        self.value = value
        self.state = brainstate.State(jnp.array([1.0, 2.0, 3.0]))
        self.param = brainstate.ParamState(jnp.array([4.0, 5.0, 6.0]))

    def update(self, x):
        return x + self.value

    def __call__(self, x, y=0):
        return x + self.value + y


class TestEnvironContext(unittest.TestCase):
    """Test cases for EnvironContext class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dummy_module = DummyModule(10)

    def test_init_valid_module(self):
        """Test EnvironContext initialization with valid module."""
        context = EnvironContext(self.dummy_module, fit=True, a='test')
        self.assertEqual(context.layer, self.dummy_module)
        self.assertEqual(context.context, {'fit': True, 'a': 'test'})

    def test_init_invalid_module(self):
        """Test EnvironContext initialization with invalid module."""
        with self.assertRaises(AssertionError):
            EnvironContext("not a module", training=True)

        with self.assertRaises(AssertionError):
            EnvironContext(None, training=True)

        with self.assertRaises(AssertionError):
            EnvironContext(42, training=True)

    def test_update_with_context(self):
        """Test update method applies context correctly."""
        context = EnvironContext(self.dummy_module, fit=True)

        # Test with positional arguments
        result = context.update(5)
        self.assertEqual(result, 15)  # 5 + 10

        # Test with keyword arguments
        result = context.update(5, y=3)
        self.assertEqual(result, 18)  # 5 + 10 + 3

    def test_update_context_applied(self):
        """Test that environment context is actually applied during update."""
        with patch.object(environ, 'context') as mock_context:
            mock_context.return_value.__enter__ = Mock(return_value=None)
            mock_context.return_value.__exit__ = Mock(return_value=None)

            context = EnvironContext(self.dummy_module, fit=True, a='eval')
            context.update(5)

            mock_context.assert_called_once_with(fit=True, a='eval')

    def test_add_context(self):
        """Test add_context method updates context correctly."""
        context = EnvironContext(self.dummy_module, fit=True)
        self.assertEqual(context.context, {'fit': True})

        # Add new context
        context.add_context(a='test', debug=False)
        self.assertEqual(context.context, {'fit': True, 'a': 'test', 'debug': False})

        # Overwrite existing context
        context.add_context(fit=False)
        self.assertEqual(context.context, {'fit': False, 'a': 'test', 'debug': False})

    def test_empty_context(self):
        """Test EnvironContext with no initial context."""
        context = EnvironContext(self.dummy_module)
        self.assertEqual(context.context, {})

        result = context.update(7)
        self.assertEqual(result, 17)  # 7 + 10


class TestFilterStates(unittest.TestCase):
    """Test cases for _filter_states function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_module = Mock(spec=Module)
        self.mock_module.states = Mock()

    def test_filter_states_none(self):
        """Test _filter_states with None filters."""
        result = _filter_states(self.mock_module, None)
        self.assertIsNone(result)
        self.mock_module.states.assert_not_called()

    def test_filter_states_single_filter(self):
        """Test _filter_states with single filter (non-dict)."""
        filter_obj = lambda x: x.startswith('test')
        self.mock_module.states.return_value = ['test1', 'test2']

        result = _filter_states(self.mock_module, filter_obj)

        self.mock_module.states.assert_called_once_with(filter_obj)
        self.assertEqual(result, ['test1', 'test2'])
