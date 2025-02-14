# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

import jax
import unittest
from absl.testing import absltest

import brainstate as bst


class TestNestedMapping(absltest.TestCase):
    def test_create_state(self):
        state = bst.util.NestedDict({'a': bst.ParamState(1), 'b': {'c': bst.ParamState(2)}})

        assert state['a'].value == 1
        assert state['b']['c'].value == 2

    def test_get_attr(self):
        state = bst.util.NestedDict({'a': bst.ParamState(1), 'b': {'c': bst.ParamState(2)}})

        assert state.a.value == 1
        assert state.b['c'].value == 2

    def test_set_attr(self):
        state = bst.util.NestedDict({'a': bst.ParamState(1), 'b': {'c': bst.ParamState(2)}})

        state.a.value = 3
        state.b['c'].value = 4

        assert state['a'].value == 3
        assert state['b']['c'].value == 4

    def test_set_attr_variables(self):
        state = bst.util.NestedDict({'a': bst.ParamState(1), 'b': {'c': bst.ParamState(2)}})

        state.a.value = 3
        state.b['c'].value = 4

        assert isinstance(state.a, bst.ParamState)
        assert state.a.value == 3
        assert isinstance(state.b['c'], bst.ParamState)
        assert state.b['c'].value == 4

    def test_add_nested_attr(self):
        state = bst.util.NestedDict({'a': bst.ParamState(1), 'b': {'c': bst.ParamState(2)}})
        state.b['d'] = bst.ParamState(5)

        assert state['b']['d'].value == 5

    def test_delete_nested_attr(self):
        state = bst.util.NestedDict({'a': bst.ParamState(1), 'b': {'c': bst.ParamState(2)}})
        del state['b']['c']

        assert 'c' not in state['b']

    def test_integer_access(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [bst.nn.Linear(1, 2), bst.nn.Linear(2, 3)]

        module = Foo()
        state_refs = bst.graph.treefy_states(module)

        assert module.layers[0].weight.value['weight'].shape == (1, 2)
        assert state_refs.layers[0]['weight'].value['weight'].shape == (1, 2)
        assert module.layers[1].weight.value['weight'].shape == (2, 3)
        assert state_refs.layers[1]['weight'].value['weight'].shape == (2, 3)

    def test_pure_dict(self):
        module = bst.nn.Linear(4, 5)
        state_map = bst.graph.treefy_states(module)
        pure_dict = state_map.to_pure_dict()
        assert isinstance(pure_dict, dict)
        assert isinstance(pure_dict['weight'].value['weight'], jax.Array)
        assert isinstance(pure_dict['weight'].value['bias'], jax.Array)


class TestSplit(unittest.TestCase):
    def test_split(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = bst.nn.BatchNorm1d([10, 3])
                self.linear = bst.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        with bst.environ.context(fit=True):
            model = Model()
            x = bst.random.randn(1, 10, 3)
            y = model(x)
            self.assertEqual(y.shape, (1, 10, 4))

        state_map = bst.graph.treefy_states(model)

        with self.assertRaises(ValueError):
            params, others = state_map.split(bst.ParamState)

        params, others = state_map.split(bst.ParamState, ...)
        print()
        print(params)
        print(others)

        self.assertTrue(len(params.to_flat()) == 2)
        self.assertTrue(len(others.to_flat()) == 2)


class TestStateMap2(unittest.TestCase):
    def test1(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = bst.nn.BatchNorm1d([10, 3])
                self.linear = bst.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        with bst.environ.context(fit=True):
            model = Model()
            state_map = bst.graph.treefy_states(model).to_flat()
            state_map = bst.util.NestedDict(state_map)


class TestFlattedMapping(unittest.TestCase):
    def test1(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = bst.nn.BatchNorm1d([10, 3])
                self.linear = bst.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        model = Model()
        # print(model.states())
        # print(bst.graph.states(model))
        self.assertTrue(model.states() == bst.graph.states(model))

        print(model.nodes())
        # print(bst.graph.nodes(model))
        self.assertTrue(model.nodes() == bst.graph.nodes(model))
