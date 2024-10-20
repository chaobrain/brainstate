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
from functools import partial
from threading import Thread
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized

import brainstate as bst


class TestIter(unittest.TestCase):
  def test1(self):
    class Model(bst.nn.Module):
      def __init__(self):
        super().__init__()
        self.a = bst.nn.Linear(1, 2)
        self.b = bst.nn.Linear(2, 3)
        self.c = [bst.nn.Linear(3, 4), bst.nn.Linear(4, 5)]
        self.d = {'x': bst.nn.Linear(5, 6), 'y': bst.nn.Linear(6, 7)}
        self.b.a = bst.nn.LIF(2)

    for path, node in bst.graph.iter_leaf(Model()):
      print(path, node)
    for path, node in bst.graph.iter_node(Model()):
      print(path, node)
    for path, node in bst.graph.iter_node(Model(), allowed_hierarchy=(1, 1)):
      print(path, node)
    for path, node in bst.graph.iter_node(Model(), allowed_hierarchy=(2, 2)):
      print(path, node)

  def test_iter_leaf_v1(self):
    class Linear(bst.nn.Module):
      def __init__(self, din, dout):
        super().__init__()
        self.weight = bst.ParamState(bst.random.randn(din, dout))
        self.bias = bst.ParamState(bst.random.randn(dout))
        self.a = 1

    module = Linear(3, 4)
    graph = [module, module]

    num = 0
    for path, value in bst.graph.iter_leaf(graph):
      print(path, type(value).__name__)
      num += 1

    assert num == 3

  def test_iter_node_v1(self):
    class Model(bst.nn.Module):
      def __init__(self):
        super().__init__()
        self.a = bst.nn.Linear(1, 2)
        self.b = bst.nn.Linear(2, 3)
        self.c = [bst.nn.Linear(3, 4), bst.nn.Linear(4, 5)]
        self.d = {'x': bst.nn.Linear(5, 6), 'y': bst.nn.Linear(6, 7)}
        self.b.a = bst.nn.LIF(2)

    model = Model()

    num = 0
    for path, node in bst.graph.iter_node([model, model]):
      print(path, node.__class__.__name__)
      num += 1
    assert num == 8


class List(bst.nn.Module):
  def __init__(self, items):
    super().__init__()
    self.items = list(items)

  def __getitem__(self, idx):
    return self.items[idx]

  def __setitem__(self, idx, value):
    self.items[idx] = value


class Dict(bst.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.items = dict(*args, **kwargs)

  def __getitem__(self, key):
    return self.items[key]

  def __setitem__(self, key, value):
    self.items[key] = value


class StatefulLinear(bst.nn.Module):
  def __init__(self, din, dout):
    super().__init__()
    self.w = bst.ParamState(bst.random.rand(din, dout))
    self.b = bst.ParamState(jnp.zeros((dout,)))
    self.count = bst.State(jnp.array(0, dtype=jnp.uint32))

  def increment(self):
    self.count.value += 1

  def __call__(self, x):
    self.count.value += 1
    return x @ self.w.value + self.b.value


class TestGraphUtils(absltest.TestCase):
  def test_flatten(self):
    a = {'a': 1, 'b': bst.ParamState(2)}
    g = [a, 3, a, bst.ParamState(4)]

    refmap = bst.graph.RefMap()
    graphdef, state = bst.graph.flatten(g, ref_index=refmap)

    state[0]['b'].raw_value = 2
    state[3].raw_value = 4

    assert len(refmap) == 2
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_unflatten(self):
    a = bst.graph.Dict(a=1, b=bst.ParamState(2))
    g = bst.graph.List([a, 3, a, bst.ParamState(4)])

    graphdef, references = bst.graph.split(g)
    g = bst.graph.merge(graphdef, references)

    print(graphdef)
    print(references)
    assert g[0] is g[2]

  def test_unflatten_pytree(self):
    a = {'a': 1, 'b': bst.ParamState(2)}
    g = [a, 3, a, bst.ParamState(4)]

    graphdef, references = bst.graph.split(g)
    g = bst.graph.merge(graphdef, references)

    assert g[0] is not g[2]

  def test_unflatten_empty(self):
    a = Dict({'a': 1, 'b': bst.ParamState(2)})
    g = List([a, 3, a, bst.ParamState(4)])

    graphdef, references = bst.graph.split(g)

    with self.assertRaisesRegex(ValueError, 'Expected key'):
      bst.graph.unflatten(graphdef, bst.util.NestedDict({}))


  def test_module_list(self):
    ls = [
      bst.nn.Linear(2, 2),
      bst.nn.BatchNorm1d([10, 2]),
    ]
    graphdef, statetree = bst.graph.split(ls)

    assert statetree[0]['weight'].value['weight'].shape == (2, 2)
    assert statetree[0]['weight'].value['bias'].shape == (2,)
    assert statetree[1]['weight'].value['scale'].shape == (1, 2,)
    assert statetree[1]['weight'].value['bias'].shape == (1, 2,)
    assert statetree[1]['running_mean'].value.shape == (1, 2,)
    assert statetree[1]['running_var'].value.shape == (1, 2)

  def test_shared_variables(self):
    v = bst.ParamState(1)
    g = [v, v]

    graphdef, statetree = bst.graph.split(g)
    assert len(statetree.to_flat()) == 1

    g2 = bst.graph.merge(graphdef, statetree)
    assert g2[0] is g2[1]

  def test_tied_weights(self):
    class Foo(bst.nn.Module):
      def __init__(self) -> None:
        super().__init__()
        self.bar = bst.nn.Linear(2, 2)
        self.baz = bst.nn.Linear(2, 2)

        # tie the weights
        self.baz.weight = self.bar.weight

    node = Foo()
    graphdef, state = bst.graph.split(node)

    assert len(state.to_flat()) == 1

    node2 = bst.graph.merge(graphdef, state)

    assert node2.bar.weight is node2.baz.weight

  def test_tied_weights_example(self):
    class LinearTranspose(bst.nn.Module):
      def __init__(self, dout: int, din: int, ) -> None:
        super().__init__()
        self.kernel = bst.ParamState(bst.init.LecunNormal()((dout, din)))

      def __call__(self, x):
        return x @ self.kernel.value.T

    class Encoder(bst.nn.Module):
      def __init__(self, ) -> None:
        super().__init__()
        self.embed = bst.nn.Embedding(10, 2)
        self.linear_out = LinearTranspose(10, 2)

        # tie the weights
        self.linear_out.kernel = self.embed.weight

      def __call__(self, x):
        x = self.embed(x)
        return self.linear_out(x)

    model = Encoder()
    graphdef, state = bst.graph.split(model)

    assert len(state.to_flat()) == 1

    x = jax.random.randint(jax.random.key(0), (2,), 0, 10)
    y = model(x)

    assert y.shape == (2, 10)

  def test_state_variables_not_shared_with_graph(self):
    class Foo(bst.graph.Node):
      def __init__(self):
        self.a = bst.ParamState(1)

    m = Foo()
    graphdef, statetree = bst.graph.split(m)

    assert isinstance(m.a, bst.ParamState)
    assert issubclass(statetree.a.type, bst.ParamState)
    assert m.a is not statetree.a
    assert m.a.value == statetree.a.value

    m2 = bst.graph.merge(graphdef, statetree)

    assert isinstance(m2.a, bst.ParamState)
    assert issubclass(statetree.a.type, bst.ParamState)
    assert m2.a is not statetree.a
    assert m2.a.value == statetree.a.value

  def test_shared_state_variables_not_shared_with_graph(self):
    class Foo(bst.graph.Node):
      def __init__(self):
        p = bst.ParamState(1)
        self.a = p
        self.b = p

    m = Foo()
    graphdef, state = bst.graph.split(m)

    assert isinstance(m.a, bst.ParamState)
    assert isinstance(m.b, bst.ParamState)
    assert issubclass(state.a.type, bst.ParamState)
    assert 'b' not in state
    assert m.a is not state.a
    assert m.b is not state.a
    assert m.a.value == state.a.value
    assert m.b.value == state.a.value

    m2 = bst.graph.merge(graphdef, state)

    assert isinstance(m2.a, bst.ParamState)
    assert isinstance(m2.b, bst.ParamState)
    assert issubclass(state.a.type, bst.ParamState)
    assert m2.a is not state.a
    assert m2.b is not state.a
    assert m2.a.value == state.a.value
    assert m2.b.value == state.a.value
    assert m2.a is m2.b

  def test_pytree_node(self):
    @bst.util.dataclass
    class Tree:
      a: bst.ParamState
      b: str = bst.util.field(pytree_node=False)

    class Foo(bst.graph.Node):
      def __init__(self):
        self.tree = Tree(bst.ParamState(1), 'a')

    m = Foo()

    graphdef, state = bst.graph.split(m)

    assert 'tree' in state
    assert 'a' in state.tree
    assert graphdef.subgraphs['tree'].type.__name__ == 'PytreeType'

    m2 = bst.graph.merge(graphdef, state)

    assert isinstance(m2.tree, Tree)
    assert m2.tree.a.value == 1
    assert m2.tree.b == 'a'
    assert m2.tree.a is not m.tree.a
    assert m2.tree is not m.tree

  @pytest.mark.skip(reason='Not implemented')
  def test_cached_unflatten(self):
    class Foo(bst.graph.Node):
      def __init__(self, ):
        self.a = bst.nn.Linear(2, 2)
        self.b = bst.nn.BatchNorm1d([10, 2])

    def f(m: Foo):
      m.a, m.b = m.b, m.a  # type: ignore

    m = Foo()
    a = m.a
    b = m.b

    ref_out_idx_out = bst.graph.RefMap()
    graphdef: bst.graph.GraphDef[Foo]
    graphdef, state = bst.graph.flatten(m, ref_index=ref_out_idx_out)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: bst.graph.GraphDef[Foo], state):
      idx_out_ref_in: dict[int, Any] = {}
      m = bst.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      f(m)
      ref_in_idx_in = bst.graph.RefMap[Any, int]()
      graphdef, state = bst.graph.flatten(m, ref_index=ref_in_idx_in)
      idx_out_idx_in = bst.graph.compose_mapping(idx_out_ref_in, ref_in_idx_in)
      static_out = bst.graph.Static((graphdef, idx_out_idx_in))
      return state, static_out

    static_out: bst.graph.Static
    state, static_out = f_pure(graphdef, state)
    idx_out_idx_in: dict[int, int]
    graphdef, idx_out_idx_in = static_out.value
    idx_in_ref_out = bst.graph.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2 = bst.graph.unflatten(graphdef, state, index_ref_cache=idx_in_ref_out)
    assert m2 is m
    assert m2.a is b
    assert m2.b is a

  @pytest.mark.skip(reason='Not implemented')
  def test_cached_unflatten_swap_variables(self):
    class Foo(bst.graph.Node):
      def __init__(self):
        self.a = bst.ParamState(1)
        self.b = bst.ParamState(2)

    def f(m: Foo):
      m.a, m.b = m.b, m.a

    m = Foo()
    a = m.a
    b = m.b

    ref_out_idx_out = bst.graph.RefMap[Any, int]()
    graphdef: bst.graph.GraphDef[Foo]
    graphdef, state = bst.graph.flatten(m, ref_index=ref_out_idx_out)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: bst.graph.GraphDef[Foo], state):
      idx_out_ref_in: dict[int, Any] = {}
      m = bst.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      f(m)
      ref_in_idx_in = bst.graph.RefMap[Any, int]()
      graphdef, state = bst.graph.flatten(m, ref_index=ref_in_idx_in)
      idx_out_idx_in = bst.graph.compose_mapping(idx_out_ref_in, ref_in_idx_in)
      static_out = bst.graph.Static((graphdef, idx_out_idx_in))
      return state, static_out

    static_out: bst.graph.Static
    state, static_out = f_pure(graphdef, state)
    idx_out_idx_in: dict[int, int]
    graphdef, idx_out_idx_in = static_out.value
    idx_in_ref_out = bst.graph.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2 = bst.graph.unflatten(graphdef, state, index_ref_cache=idx_in_ref_out)
    assert m2 is m
    assert m2.a is b
    assert m2.b is a

  @pytest.mark.skip(reason='Not implemented')
  def test_cached_unflatten_add_self_reference(self):
    class Foo(bst.graph.Node):
      def __init__(self):
        self.ref = None

    def f(m: Foo):
      m.ref = m

    m = Foo()

    ref_out_idx_out = bst.graph.RefMap()
    graphdef: bst.graph.GraphDef[Foo]
    graphdef, state = bst.graph.flatten(m, ref_index=ref_out_idx_out)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: bst.graph.GraphDef[Foo], state):
      idx_out_ref_in: dict[int, Any] = {}
      m = bst.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      f(m)
      ref_in_idx_in = bst.graph.RefMap[Any, int]()
      graphdef, state = bst.graph.flatten(m, ref_index=ref_in_idx_in)
      idx_out_idx_in = bst.graph.compose_mapping(idx_out_ref_in, ref_in_idx_in)
      static_out = bst.graph.Static((graphdef, idx_out_idx_in))
      return state, static_out

    static_out: bst.graph.Static
    state, static_out = f_pure(graphdef, state)
    idx_out_idx_in: dict[int, int]
    graphdef, idx_out_idx_in = static_out.value
    idx_in_ref_out = bst.graph.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2 = bst.graph.unflatten(graphdef, state, index_ref_cache=idx_in_ref_out)
    assert m2 is m
    assert m2.ref is m2

  def test_call_jit_update(self):
    class Counter(bst.graph.Node):
      def __init__(self):
        self.count = bst.ParamState(jnp.zeros(()))

      def inc(self):
        self.count.value += 1
        return 1

    graph_state = bst.graph.split(Counter())

    @jax.jit
    def update(graph_state):
      out, graph_state = bst.graph.call(graph_state).inc()
      self.assertEqual(out, 1)
      return graph_state

    graph_state = update(graph_state)
    graph_state = update(graph_state)

    counter = bst.graph.merge(*graph_state)

    self.assertEqual(counter.count.value, 2)

  def test_stateful_linear(self):
    linear = StatefulLinear(3, 2)
    linear_state = bst.graph.split(linear)

    @jax.jit
    def forward(x, pure_linear):
      y, pure_linear = bst.graph.call(pure_linear)(x)
      return y, pure_linear

    x = jnp.ones((1, 3))
    y, linear_state = forward(x, linear_state)
    y, linear_state = forward(x, linear_state)

    self.assertEqual(linear.count.value, 0)
    new_linear = bst.graph.merge(*linear_state)
    self.assertEqual(new_linear.count.value, 2)

  def test_getitem(self):
    nodes = dict(
      a=StatefulLinear(3, 2),
      b=StatefulLinear(2, 1),
    )
    node_state = bst.graph.split(nodes)
    _, node_state = bst.graph.call(node_state)['b'].increment()

    nodes = bst.graph.merge(*node_state)

    self.assertEqual(nodes['a'].count.value, 0)
    self.assertEqual(nodes['b'].count.value, 1)

  def test_to_tree_simple(self):
    m = bst.nn.Linear(2, 3, )
    impure_tree = (m, 1, {'b': m})

    pure_tree = bst.graph.graph_to_tree(impure_tree)[0]

    t1 = pure_tree[0]
    t2 = pure_tree[2]['b']

    self.assertEqual(pure_tree[1], 1)
    self.assertIsInstance(t1, bst.graph.NodeStates)
    assert isinstance(t1, bst.graph.NodeStates)
    self.assertIsInstance(t2, bst.graph.NodeStates)
    assert isinstance(t2, bst.graph.NodeStates)
    self.assertIsInstance(t1.graphdef, bst.graph.NodeDef)
    self.assertIsInstance(t2.graphdef, bst.graph.NodeRef)
    self.assertLen(t1.states[0].to_flat(), 1)
    self.assertLen(t2.states[0].to_flat(), 0)

    impure_tree2 = bst.graph.tree_to_graph(pure_tree)

    m1_out = impure_tree2[0]
    m2_out = impure_tree2[2]['b']

    self.assertIs(m1_out, m2_out)
    self.assertEqual(impure_tree2[1], 1)

  def test_to_tree_consistent_prefix(self):
    m = bst.nn.Linear(2, 3, )
    impure_tree = (m, 1, {'b': m})
    prefix = (0, None, 0)
    pure_tree = bst.graph.graph_to_tree(impure_tree, prefix=prefix)

    prefix = (0, None, 1)
    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing detected'):
      bst.graph.graph_to_tree(impure_tree, prefix=prefix)


class SimpleModule(bst.nn.Module):
  pass


class SimplePyTreeModule(bst.nn.Module):
  pass


class TestThreading(parameterized.TestCase):

  @parameterized.parameters(
    (SimpleModule,),
    (SimplePyTreeModule,),
  )
  def test_threading(self, module_fn: Callable[[], bst.nn.Module]):
    x = module_fn()

    class MyThread(Thread):

      def run(self) -> None:
        bst.graph.split(x)

    thread = MyThread()
    thread.start()
    thread.join()


class TestFlatten(unittest.TestCase):
  def test1(self):
    class MyNode(bst.graph.Node):
      def __init__(self):
        self.a = bst.nn.Linear(2, 3)
        self.b = bst.nn.Linear(3, 2)
        self.c = [bst.nn.Linear(1, 2), bst.nn.Linear(1, 3)]
        self.d = {'x': bst.nn.Linear(1, 3), 'y': bst.nn.Linear(1, 4)}

    graphdef, statetree = bst.graph.flatten(MyNode())
    # print(graphdef)
    print(statetree)
    # print(bst.graph.unflatten(graphdef, statetree))

  def test_split(self):
    class Foo(bst.graph.Node):
      def __init__(self):
        self.a = bst.nn.Linear(2, 2)
        self.b = bst.nn.BatchNorm1d([10, 2])

    node = Foo()
    graphdef, params, others = bst.graph.split(node, bst.ParamState, ...)

    print(params)
    print(jax.tree.map(jnp.shape, params))

    print(jax.tree.map(jnp.shape, others))

  def test_merge(self):
    class Foo(bst.graph.Node):
      def __init__(self):
        self.a = bst.nn.Linear(2, 2)
        self.b = bst.nn.BatchNorm1d([10, 2])

    node = Foo()
    graphdef, params, others = bst.graph.split(node, bst.ParamState, ...)

    new_node = bst.graph.merge(graphdef, params, others)

    assert isinstance(new_node, Foo)
    assert isinstance(new_node.b, bst.nn.BatchNorm1d)
    assert isinstance(new_node.a, bst.nn.Linear)

  def test_update_states(self):
    x = jnp.ones((1, 2))
    y = jnp.ones((1, 3))
    model = bst.nn.Linear(2, 3)

    def loss_fn(x, y):
      return jnp.mean((y - model(x)) ** 2)

    def sgd(ps, gs):
      updates = jax.tree.map(lambda p, g: p - 0.1 * g, ps.value, gs)
      ps.value = updates

    prev_loss = loss_fn(x, y)
    weights = model.states()
    grads = bst.augment.grad(loss_fn, weights)(x, y)
    for key, val in grads.items():
      sgd(weights[key], val)
    assert loss_fn(x, y) < prev_loss

  def test_pop_states(self):
    class Model(bst.nn.Module):
      def __init__(self):
        super().__init__()
        self.a = bst.nn.Linear(2, 3)
        self.b = bst.nn.LIF([10, 2])

    model = Model()
    with bst.catch_new_states('new'):
      bst.nn.all_init_states(model)
    print(model.states())
    self.assertTrue(len(model.states()) == 2)
    model_states = bst.graph.pop_states(model, 'new')
    print(model_states)
    self.assertTrue(len(model.states()) == 1)
    assert not hasattr(model.b.V, 'st')
    # print(model.states())


if __name__ == '__main__':
  absltest.main()
