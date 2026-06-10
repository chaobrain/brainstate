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

import jax
import jax.numpy as jnp

import brainstate
from brainstate.graph import states as graph_states


class TestEvalShape:
    def test_eval_shape_simple(self):
        out = brainstate.transform.eval_shape(lambda x: x * 2.0, jnp.ones(3))
        assert out.shape == (3,)
        assert out.dtype == jnp.float32

    def test_eval_shape_with_node(self):
        model = brainstate.transform.eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))
        assert isinstance(model, brainstate.nn.LSTMCell)

    def test_eval_shape_reads_existing_state(self):
        # The originally reported bug: eval_shape over a fn that reads+writes an
        # existing global State must not raise.
        st = brainstate.State(jnp.zeros(3))

        def f(x):
            st.value = st.value + x
            return st.value * 2.0

        out = brainstate.transform.eval_shape(f, jnp.ones(3))
        assert out.shape == (3,)
        assert out.dtype == jnp.float32

    def test_eval_shape_does_not_mutate_state(self):
        st = brainstate.State(jnp.zeros((2, 3)))

        def f(x):
            st.value = st.value + x
            return st.value.sum()

        before = st.value
        brainstate.transform.eval_shape(f, jnp.ones((2, 3)))
        # value object unchanged: still concrete, same shape, all zeros.
        assert st.value.shape == (2, 3)
        assert jnp.all(st.value == before)
        assert not isinstance(st.value, jax.ShapeDtypeStruct)

    def test_eval_shape_return_state_shapes(self):
        st = brainstate.State(jnp.zeros((2, 3)))

        def f(x):
            st.value = st.value + x
            return st.value.sum()

        state_shapes, out_shapes = brainstate.transform.eval_shape(
            f, jnp.ones((2, 3)), return_state_shapes=True
        )
        # state_shapes is a dict keyed by State, valued by ShapeDtypeStruct.
        assert isinstance(state_shapes, dict)
        assert st in state_shapes
        sds = state_shapes[st]
        assert isinstance(sds, jax.ShapeDtypeStruct)
        assert sds.shape == (2, 3)
        assert sds.dtype == jnp.float32
        # out_shapes is the same value the default return would give (a scalar SDS).
        assert isinstance(out_shapes, jax.ShapeDtypeStruct)
        assert out_shapes.shape == ()

    def test_eval_shape_node_has_abstract_leaves(self):
        m = brainstate.transform.eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))
        assert isinstance(m, brainstate.nn.LSTMCell)
        leaves = []
        for state in graph_states(m).values():
            leaves.extend(jax.tree.leaves(state.value))
        assert len(leaves) > 0
        # every leaf is abstract (no concrete memory allocated)
        assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves)

    def test_eval_shape_node_is_traceable_downstream(self):
        # The abstract node must be usable as input to ANOTHER transform without
        # the tracing-leakage ValueError.
        m = brainstate.transform.eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))
        first_state = list(graph_states(m).values())[0]

        # (a) nested eval_shape reading the abstract node's state
        nested = brainstate.transform.eval_shape(lambda: first_state.value)
        nested_leaves = jax.tree.leaves(nested)
        assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in nested_leaves)

        # (b) jit().eval_shape over a fn using the abstract node
        @brainstate.transform.jit
        def use_model():
            return first_state.value

        out = use_model.eval_shape()
        assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(out))

    def test_eval_shape_node_init_all_states_runs(self):
        # init_all_states on the abstract node must run without error. (For a
        # param-only cell like LSTMCell it initializes hidden states; param leaves
        # stay abstract because they are built in __init__.)
        m = brainstate.transform.eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))
        brainstate.nn.init_all_states(m)  # must not raise

    def test_eval_shape_pytree_args_and_kwargs(self):
        def f(d, *, scale):
            return {'sum': d['a'] + d['b'], 'scaled': d['a'] * scale}

        out = brainstate.transform.eval_shape(
            f, {'a': jnp.ones((2, 3)), 'b': jnp.ones((2, 3))}, scale=jnp.ones(())
        )
        assert out['sum'].shape == (2, 3)
        assert out['scaled'].shape == (2, 3)
        assert all(
            isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(out)
        )

    def test_eval_shape_mixed_state_types(self):
        p = brainstate.ParamState(jnp.zeros((4,)))
        s = brainstate.ShortTermState(jnp.zeros((4,)))

        def f(x):
            p.value = p.value + x
            s.value = s.value + x
            return p.value + s.value

        state_shapes, out = brainstate.transform.eval_shape(
            f, jnp.ones((4,)), return_state_shapes=True
        )
        assert p in state_shapes and s in state_shapes
        assert state_shapes[p].shape == (4,)
        assert state_shapes[s].shape == (4,)
        assert out.shape == (4,)

    def test_eval_shape_with_random_state(self):
        # brainstate.random usage inside f must trace cleanly.
        def f(x):
            return x + brainstate.random.normal(size=x.shape)

        out = brainstate.transform.eval_shape(f, jnp.ones((5,)))
        assert out.shape == (5,)
        assert out.dtype == jnp.float32


class TestFailedEvalShapeUnwindsNewStates:
    """States created inside ``f`` must be unwound (value + stack level)
    even when the abstract trace fails (audit M5)."""

    def test_failure_inside_f_unwinds_created_states(self):
        import pytest

        box = {}

        def f():
            st = brainstate.ShortTermState(jnp.zeros(3))
            box['st'] = st
            raise RuntimeError('boom')

        with pytest.raises(RuntimeError):
            brainstate.transform.eval_shape(f)
        st = box['st']
        # The stack level must be unwound so the orphaned state stays usable.
        # (Its value is a staged tracer either way -- it was created inside the
        # abstract trace and never had a concrete value -- so only the level
        # bookkeeping is observable.)
        assert st.stack_level == 0
        st.value = jnp.ones(3)
        assert bool(jnp.allclose(st.value, jnp.ones(3)))
