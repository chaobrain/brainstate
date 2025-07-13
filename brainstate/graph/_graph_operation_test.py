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
from collections.abc import Callable
from threading import Thread

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import brainstate


class TestIter(unittest.TestCase):
    def test1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)
                self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
                self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
                self.b.a = brainstate.nn.LIF(2)

        for path, node in brainstate.graph.iter_leaf(Model()):
            print(path, node)
        for path, node in brainstate.graph.iter_node(Model()):
            print(path, node)
        for path, node in brainstate.graph.iter_node(Model(), allowed_hierarchy=(1, 1)):
            print(path, node)
        for path, node in brainstate.graph.iter_node(Model(), allowed_hierarchy=(2, 2)):
            print(path, node)

    def test_iter_leaf_v1(self):
        class Linear(brainstate.nn.Module):
            def __init__(self, din, dout):
                super().__init__()
                self.weight = brainstate.ParamState(brainstate.random.randn(din, dout))
                self.bias = brainstate.ParamState(brainstate.random.randn(dout))
                self.a = 1

        module = Linear(3, 4)
        graph = [module, module]

        num = 0
        for path, value in brainstate.graph.iter_leaf(graph):
            print(path, type(value).__name__)
            num += 1

        assert num == 3

    def test_iter_node_v1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(1, 2)
                self.b = brainstate.nn.Linear(2, 3)
                self.c = [brainstate.nn.Linear(3, 4), brainstate.nn.Linear(4, 5)]
                self.d = {'x': brainstate.nn.Linear(5, 6), 'y': brainstate.nn.Linear(6, 7)}
                self.b.a = brainstate.nn.LIF(2)

        model = Model()

        num = 0
        for path, node in brainstate.graph.iter_node([model, model]):
            print(path, node.__class__.__name__)
            num += 1
        assert num == 8


class List(brainstate.nn.Module):
    def __init__(self, items):
        super().__init__()
        self.items = list(items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __setitem__(self, idx, value):
        self.items[idx] = value


class Dict(brainstate.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.items = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value


class StatefulLinear(brainstate.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.rand(din, dout))
        self.b = brainstate.ParamState(jnp.zeros((dout,)))
        self.count = brainstate.State(jnp.array(0, dtype=jnp.uint32))

    def increment(self):
        self.count.value += 1

    def __call__(self, x):
        self.count.value += 1
        return x @ self.w.value + self.b.value


class TestGraphUtils(absltest.TestCase):
    def test_flatten_treey_state(self):
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]

        refmap = brainstate.graph.RefMap()
        graphdef, states = brainstate.graph.flatten(g, ref_index=refmap, treefy_state=True)

        states[0]['b'].value = 2
        states[3].value = 4

        assert isinstance(states[0]['b'], brainstate.TreefyState)
        assert isinstance(states[3], brainstate.TreefyState)
        assert isinstance(states, brainstate.util.NestedDict)
        assert len(refmap) == 2
        assert a['b'] in refmap
        assert g[3] in refmap

    def test_flatten(self):
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]

        refmap = brainstate.graph.RefMap()
        graphdef, states = brainstate.graph.flatten(g, ref_index=refmap, treefy_state=False)

        states[0]['b'].value = 2
        states[3].value = 4

        assert isinstance(states[0]['b'], brainstate.State)
        assert isinstance(states[3], brainstate.State)
        assert len(refmap) == 2
        assert a['b'] in refmap
        assert g[3] in refmap

    def test_unflatten_treey_state(self):
        a = brainstate.graph.Dict(a=1, b=brainstate.ParamState(2))
        g1 = brainstate.graph.List([a, 3, a, brainstate.ParamState(4)])

        graphdef, references = brainstate.graph.flatten(g1, treefy_state=True)
        g = brainstate.graph.unflatten(graphdef, references)

        print(graphdef)
        print(references)
        assert g[0] is g[2]
        assert g1[3] is not g[3]
        assert g1[0]['b'] is not g[0]['b']

    def test_unflatten(self):
        a = brainstate.graph.Dict(a=1, b=brainstate.ParamState(2))
        g1 = brainstate.graph.List([a, 3, a, brainstate.ParamState(4)])

        graphdef, references = brainstate.graph.flatten(g1, treefy_state=False)
        g = brainstate.graph.unflatten(graphdef, references)

        print(graphdef)
        print(references)
        assert g[0] is g[2]
        assert g1[3] is g[3]
        assert g1[0]['b'] is g[0]['b']

    def test_unflatten_pytree(self):
        a = {'a': 1, 'b': brainstate.ParamState(2)}
        g = [a, 3, a, brainstate.ParamState(4)]

        graphdef, references = brainstate.graph.treefy_split(g)
        g = brainstate.graph.treefy_merge(graphdef, references)

        assert g[0] is not g[2]

    def test_unflatten_empty(self):
        a = Dict({'a': 1, 'b': brainstate.ParamState(2)})
        g = List([a, 3, a, brainstate.ParamState(4)])

        graphdef, references = brainstate.graph.treefy_split(g)

        with self.assertRaisesRegex(ValueError, 'Expected key'):
            brainstate.graph.unflatten(graphdef, brainstate.util.NestedDict({}))

    def test_module_list(self):
        ls = [
            brainstate.nn.Linear(2, 2),
            brainstate.nn.BatchNorm1d([10, 2]),
        ]
        graphdef, statetree = brainstate.graph.treefy_split(ls)

        assert statetree[0]['weight'].value['weight'].shape == (2, 2)
        assert statetree[0]['weight'].value['bias'].shape == (2,)
        assert statetree[1]['weight'].value['scale'].shape == (1, 2,)
        assert statetree[1]['weight'].value['bias'].shape == (1, 2,)
        assert statetree[1]['running_mean'].value.shape == (1, 2,)
        assert statetree[1]['running_var'].value.shape == (1, 2)

    def test_shared_variables(self):
        v = brainstate.ParamState(1)
        g = [v, v]

        graphdef, statetree = brainstate.graph.treefy_split(g)
        assert len(statetree.to_flat()) == 1

        g2 = brainstate.graph.treefy_merge(graphdef, statetree)
        assert g2[0] is g2[1]

    def test_tied_weights(self):
        class Foo(brainstate.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = brainstate.nn.Linear(2, 2)
                self.baz = brainstate.nn.Linear(2, 2)

                # tie the weights
                self.baz.weight = self.bar.weight

        node = Foo()
        graphdef, state = brainstate.graph.treefy_split(node)

        assert len(state.to_flat()) == 1

        node2 = brainstate.graph.treefy_merge(graphdef, state)

        assert node2.bar.weight is node2.baz.weight

    def test_tied_weights_example(self):
        class LinearTranspose(brainstate.nn.Module):
            def __init__(self, dout: int, din: int, ) -> None:
                super().__init__()
                self.kernel = brainstate.ParamState(brainstate.init.LecunNormal()((dout, din)))

            def __call__(self, x):
                return x @ self.kernel.value.T

        class Encoder(brainstate.nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.embed = brainstate.nn.Embedding(10, 2)
                self.linear_out = LinearTranspose(10, 2)

                # tie the weights
                self.linear_out.kernel = self.embed.weight

            def __call__(self, x):
                x = self.embed(x)
                return self.linear_out(x)

        model = Encoder()
        graphdef, state = brainstate.graph.treefy_split(model)

        assert len(state.to_flat()) == 1

        x = jax.random.randint(jax.random.key(0), (2,), 0, 10)
        y = model(x)

        assert y.shape == (2, 10)

    def test_state_variables_not_shared_with_graph(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.ParamState(1)

        m = Foo()
        graphdef, statetree = brainstate.graph.treefy_split(m)

        assert isinstance(m.a, brainstate.ParamState)
        assert issubclass(statetree.a.type, brainstate.ParamState)
        assert m.a is not statetree.a
        assert m.a.value == statetree.a.value

        m2 = brainstate.graph.treefy_merge(graphdef, statetree)

        assert isinstance(m2.a, brainstate.ParamState)
        assert issubclass(statetree.a.type, brainstate.ParamState)
        assert m2.a is not statetree.a
        assert m2.a.value == statetree.a.value

    def test_shared_state_variables_not_shared_with_graph(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                p = brainstate.ParamState(1)
                self.a = p
                self.b = p

        m = Foo()
        graphdef, state = brainstate.graph.treefy_split(m)

        assert isinstance(m.a, brainstate.ParamState)
        assert isinstance(m.b, brainstate.ParamState)
        assert issubclass(state.a.type, brainstate.ParamState)
        assert 'b' not in state
        assert m.a is not state.a
        assert m.b is not state.a
        assert m.a.value == state.a.value
        assert m.b.value == state.a.value

        m2 = brainstate.graph.treefy_merge(graphdef, state)

        assert isinstance(m2.a, brainstate.ParamState)
        assert isinstance(m2.b, brainstate.ParamState)
        assert issubclass(state.a.type, brainstate.ParamState)
        assert m2.a is not state.a
        assert m2.b is not state.a
        assert m2.a.value == state.a.value
        assert m2.b.value == state.a.value
        assert m2.a is m2.b

    def test_pytree_node(self):
        @brainstate.util.dataclass
        class Tree:
            a: brainstate.ParamState
            b: str = brainstate.util.field(pytree_node=False)

        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.tree = Tree(brainstate.ParamState(1), 'a')

        m = Foo()

        graphdef, state = brainstate.graph.treefy_split(m)

        assert 'tree' in state
        assert 'a' in state.tree
        assert graphdef.subgraphs['tree'].type.__name__ == 'PytreeType'

        m2 = brainstate.graph.treefy_merge(graphdef, state)

        assert isinstance(m2.tree, Tree)
        assert m2.tree.a.value == 1
        assert m2.tree.b == 'a'
        assert m2.tree.a is not m.tree.a
        assert m2.tree is not m.tree

    def test_call_jit_update(self):
        class Counter(brainstate.graph.Node):
            def __init__(self):
                self.count = brainstate.ParamState(jnp.zeros(()))

            def inc(self):
                self.count.value += 1
                return 1

        graph_state = brainstate.graph.treefy_split(Counter())

        @jax.jit
        def update(graph_state):
            out, graph_state = brainstate.graph.call(graph_state).inc()
            self.assertEqual(out, 1)
            return graph_state

        graph_state = update(graph_state)
        graph_state = update(graph_state)

        counter = brainstate.graph.treefy_merge(*graph_state)

        self.assertEqual(counter.count.value, 2)

    def test_stateful_linear(self):
        linear = StatefulLinear(3, 2)
        linear_state = brainstate.graph.treefy_split(linear)

        @jax.jit
        def forward(x, pure_linear):
            y, pure_linear = brainstate.graph.call(pure_linear)(x)
            return y, pure_linear

        x = jnp.ones((1, 3))
        y, linear_state = forward(x, linear_state)
        y, linear_state = forward(x, linear_state)

        self.assertEqual(linear.count.value, 0)
        new_linear = brainstate.graph.treefy_merge(*linear_state)
        self.assertEqual(new_linear.count.value, 2)

    def test_getitem(self):
        nodes = dict(
            a=StatefulLinear(3, 2),
            b=StatefulLinear(2, 1),
        )
        node_state = brainstate.graph.treefy_split(nodes)
        _, node_state = brainstate.graph.call(node_state)['b'].increment()

        nodes = brainstate.graph.treefy_merge(*node_state)

        self.assertEqual(nodes['a'].count.value, 0)
        self.assertEqual(nodes['b'].count.value, 1)


class SimpleModule(brainstate.nn.Module):
    pass


class SimplePyTreeModule(brainstate.nn.Module):
    pass


class TestThreading(parameterized.TestCase):

    @parameterized.parameters(
        (SimpleModule,),
        (SimplePyTreeModule,),
    )
    def test_threading(self, module_fn: Callable[[], brainstate.nn.Module]):
        x = module_fn()

        class MyThread(Thread):

            def run(self) -> None:
                brainstate.graph.treefy_split(x)

        thread = MyThread()
        thread.start()
        thread.join()


class TestGraphOperation(unittest.TestCase):
    def test1(self):
        class MyNode(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainstate.nn.Linear(3, 2)
                self.c = [brainstate.nn.Linear(1, 2), brainstate.nn.Linear(1, 3)]
                self.d = {'x': brainstate.nn.Linear(1, 3), 'y': brainstate.nn.Linear(1, 4)}

        graphdef, statetree = brainstate.graph.flatten(MyNode())
        # print(graphdef)
        print(statetree)
        # print(brainstate.graph.unflatten(graphdef, statetree))

    def test_split(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 2)
                self.b = brainstate.nn.BatchNorm1d([10, 2])

        node = Foo()
        graphdef, params, others = brainstate.graph.treefy_split(node, brainstate.ParamState, ...)

        print(params)
        print(jax.tree.map(jnp.shape, params))

        print(jax.tree.map(jnp.shape, others))

    def test_merge(self):
        class Foo(brainstate.graph.Node):
            def __init__(self):
                self.a = brainstate.nn.Linear(2, 2)
                self.b = brainstate.nn.BatchNorm1d([10, 2])

        node = Foo()
        graphdef, params, others = brainstate.graph.treefy_split(node, brainstate.ParamState, ...)

        new_node = brainstate.graph.treefy_merge(graphdef, params, others)

        assert isinstance(new_node, Foo)
        assert isinstance(new_node.b, brainstate.nn.BatchNorm1d)
        assert isinstance(new_node.a, brainstate.nn.Linear)

    def test_update_states(self):
        x = jnp.ones((1, 2))
        y = jnp.ones((1, 3))
        model = brainstate.nn.Linear(2, 3)

        def loss_fn(x, y):
            return jnp.mean((y - model(x)) ** 2)

        def sgd(ps, gs):
            updates = jax.tree.map(lambda p, g: p - 0.1 * g, ps.value, gs)
            ps.value = updates

        prev_loss = loss_fn(x, y)
        weights = model.states()
        grads = brainstate.augment.grad(loss_fn, weights)(x, y)
        for key, val in grads.items():
            sgd(weights[key], val)
        assert loss_fn(x, y) < prev_loss

    def test_pop_states(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.nn.Linear(2, 3)
                self.b = brainstate.nn.LIF([10, 2])

        model = Model()
        with brainstate.catch_new_states('new'):
            brainstate.nn.init_all_states(model)
        # print(model.states())
        self.assertTrue(len(model.states()) == 2)
        model_states = brainstate.graph.pop_states(model, 'new')
        print(model_states)
        self.assertTrue(len(model.states()) == 1)
        assert not hasattr(model.b, 'V')
        # print(model.states())

    def test_treefy_split(self):
        class MLP(brainstate.graph.Node):
            def __init__(self, din: int, dmid: int, dout: int, n_layer: int = 3):
                self.input = brainstate.nn.Linear(din, dmid)
                self.layers = [brainstate.nn.Linear(dmid, dmid) for _ in range(n_layer)]
                self.output = brainstate.nn.Linear(dmid, dout)

            def __call__(self, x):
                x = brainstate.functional.relu(self.input(x))
                for layer in self.layers:
                    x = brainstate.functional.relu(layer(x))
                return self.output(x)

        model = MLP(2, 1, 3)
        graph_def, treefy_states = brainstate.graph.treefy_split(model)

        print(graph_def)
        print(treefy_states)

        # states = brainstate.graph.states(model)
        # print(states)
        # nest_states = states.to_nest()
        # print(nest_states)

    def test_states(self):
        class MLP(brainstate.graph.Node):
            def __init__(self, din: int, dmid: int, dout: int, n_layer: int = 3):
                self.input = brainstate.nn.Linear(din, dmid)
                self.layers = [brainstate.nn.Linear(dmid, dmid) for _ in range(n_layer)]
                self.output = brainstate.nn.LIF(dout)

            def __call__(self, x):
                x = brainstate.functional.relu(self.input(x))
                for layer in self.layers:
                    x = brainstate.functional.relu(layer(x))
                return self.output(x)

        model = brainstate.nn.init_all_states(MLP(2, 1, 3))
        states = brainstate.graph.states(model)
        print(states)
        nest_states = states.to_nest()
        print(nest_states)

        params, others = brainstate.graph.states(model, brainstate.ParamState, brainstate.ShortTermState)
        print(params)
        print(others)


if __name__ == '__main__':
    absltest.main()
