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


import unittest

import jax
from absl.testing import absltest

import brainstate


class TestNestedMapping(absltest.TestCase):
    def test_create_state(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        assert state['a'].value == 1
        assert state['b']['c'].value == 2

    def test_get_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        assert state.a.value == 1
        assert state.b['c'].value == 2

    def test_set_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        state.a.value = 3
        state.b['c'].value = 4

        assert state['a'].value == 3
        assert state['b']['c'].value == 4

    def test_set_attr_variables(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})

        state.a.value = 3
        state.b['c'].value = 4

        assert isinstance(state.a, brainstate.ParamState)
        assert state.a.value == 3
        assert isinstance(state.b['c'], brainstate.ParamState)
        assert state.b['c'].value == 4

    def test_add_nested_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})
        state.b['d'] = brainstate.ParamState(5)

        assert state['b']['d'].value == 5

    def test_delete_nested_attr(self):
        state = brainstate.util.NestedDict({'a': brainstate.ParamState(1), 'b': {'c': brainstate.ParamState(2)}})
        del state['b']['c']

        assert 'c' not in state['b']

    def test_integer_access(self):
        class Foo(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [brainstate.nn.Linear(1, 2), brainstate.nn.Linear(2, 3)]

        module = Foo()
        state_refs = brainstate.graph.treefy_states(module)

        assert module.layers[0].weight.value['weight'].shape == (1, 2)
        assert state_refs.layers[0]['weight'].value['weight'].shape == (1, 2)
        assert module.layers[1].weight.value['weight'].shape == (2, 3)
        assert state_refs.layers[1]['weight'].value['weight'].shape == (2, 3)

    def test_pure_dict(self):
        module = brainstate.nn.Linear(4, 5)
        state_map = brainstate.graph.treefy_states(module)
        pure_dict = state_map.to_pure_dict()
        assert isinstance(pure_dict, dict)
        assert isinstance(pure_dict['weight'].value['weight'], jax.Array)
        assert isinstance(pure_dict['weight'].value['bias'], jax.Array)


class TestSplit(unittest.TestCase):
    def test_split(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = brainstate.nn.BatchNorm1d([10, 3])
                self.linear = brainstate.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        with brainstate.environ.context(fit=True):
            model = Model()
            x = brainstate.random.randn(1, 10, 3)
            y = model(x)
            self.assertEqual(y.shape, (1, 10, 4))

        state_map = brainstate.graph.treefy_states(model)

        with self.assertRaises(ValueError):
            params, others = state_map.split(brainstate.ParamState)

        params, others = state_map.split(brainstate.ParamState, ...)
        print()
        print(params)
        print(others)

        self.assertTrue(len(params.to_flat()) == 2)
        self.assertTrue(len(others.to_flat()) == 2)


class TestStateMap2(unittest.TestCase):
    def test1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = brainstate.nn.BatchNorm1d([10, 3])
                self.linear = brainstate.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        with brainstate.environ.context(fit=True):
            model = Model()
            state_map = brainstate.graph.treefy_states(model).to_flat()
            state_map = brainstate.util.NestedDict(state_map)


class TestFlattedMapping(unittest.TestCase):
    def test1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.batchnorm = brainstate.nn.BatchNorm1d([10, 3])
                self.linear = brainstate.nn.Linear([10, 3], [10, 4])

            def __call__(self, x):
                return self.linear(self.batchnorm(x))

        model = Model()
        # print(model.states())
        # print(brainstate.graph.states(model))
        self.assertTrue(model.states() == brainstate.graph.states(model))

        print(model.nodes())
        # print(brainstate.graph.nodes(model))
        self.assertTrue(model.nodes() == brainstate.graph.nodes(model))
