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

import functools
from typing import Any, Callable, TypeVar

from brainstate._state import catch_new_states
from brainstate.graph import graph_to_tree, tree_to_graph
from ._make_jaxpr import StatefulFunction

__all__ = [
    'eval_shape',
]

A = TypeVar('A')


def eval_shape(
    f: Callable[..., A],
    *args: Any,
    return_state_shapes: bool = False,
    **kwargs: Any,
) -> A:
    """Compute the abstract output shape of ``f`` without executing it.

    This is the brainstate-aware analogue of :func:`jax.eval_shape`. It traces
    ``f`` abstractly (no real computation, no array allocation, no mutation of any
    :class:`~brainstate.State` value) and returns the shape/dtype structure of the
    output. It is built on :class:`~brainstate.transform.StatefulFunction`, exactly
    like :func:`~brainstate.transform.jit`, :func:`~brainstate.transform.grad`, and
    :func:`~brainstate.transform.vmap`, so it handles existing global states the same
    way every other transform does.

    Three behaviors are supported by a single abstract trace:

    1. **Plain outputs.** If ``f`` returns arrays/pytrees, the result is a pytree
       with :class:`jax.ShapeDtypeStruct` leaves.
    2. **Existing states.** If ``f`` reads or writes pre-existing global
       :class:`~brainstate.State` objects, they are traced transparently (no error).
       Their concrete values are left unchanged after the call.
    3. **New nodes.** If ``f`` constructs and returns a brainstate
       :class:`~brainstate.graph.Node` (e.g. ``lambda: brainstate.nn.LSTMCell(3, 4)``),
       a node of the same type is reconstructed with abstract
       :class:`jax.ShapeDtypeStruct` leaves (lazy/abstract model construction, no
       memory allocated). The returned node is a first-class input to subsequent
       brainstate transformations.

    Parameters
    ----------
    f : Callable
        The function to abstractly evaluate. It is never executed for real.
    *args : Any
        Example positional arguments. May contain arrays, pytrees, or brainstate
        graph nodes.
    return_state_shapes : bool, optional
        If ``True``, return ``(state_shapes, out_shapes)`` where ``state_shapes`` is
        a ``dict`` mapping each touched :class:`~brainstate.State` to the
        :class:`jax.ShapeDtypeStruct` of its value. Default is ``False`` (return only
        the output shapes).
    **kwargs : Any
        Example keyword arguments.

    Returns
    -------
    out_shapes : Any
        The abstract output of ``f`` (pytree of :class:`jax.ShapeDtypeStruct`, or a
        reconstructed abstract :class:`~brainstate.graph.Node`). When
        ``return_state_shapes=True`` this is the second element of the returned
        ``(state_shapes, out_shapes)`` tuple.
    state_shapes : dict
        Only when ``return_state_shapes=True``: ``dict`` of
        ``State -> jax.ShapeDtypeStruct`` for every state touched by ``f``. Returned
        as the first element of the ``(state_shapes, out_shapes)`` tuple.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> out = brainstate.transform.eval_shape(lambda x: x * 2.0, jnp.ones(3))
        >>> out.shape, out.dtype
        ((3,), dtype('float32'))

        >>> model = brainstate.transform.eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))
        >>> isinstance(model, brainstate.nn.LSTMCell)
        True
    """
    # Convert any graph nodes inside the example inputs into pure pytrees so the
    # StatefulFunction trace can consume them.
    (g_args, g_kwargs), _ = graph_to_tree((args, kwargs))

    # Captured states created INSIDE f (the lambda: LSTMCell(...) case) are stored
    # here so we can clean up their trace level after the abstract trace ends.
    caught_box: dict = {}

    @functools.wraps(f)
    def _wrapped(*inner_args, **inner_kwargs):
        # Rebuild graph nodes from the pure-pytree inputs before calling f.
        inner_args, inner_kwargs = tree_to_graph((inner_args, inner_kwargs))
        with catch_new_states() as catcher:
            try:
                out = f(*inner_args, **inner_kwargs)
            finally:
                # record the created states even when f fails, so the cleanup
                # below can unwind them
                caught_box['states'] = catcher.get_states()
                caught_box['values'] = catcher.get_state_values()
        # Represent a returned Node as a pure pytree so StatefulFunction can derive
        # its output shapes.
        out_tree, _ = graph_to_tree(out)
        return out_tree

    # One abstract trace via the canonical stateful wrapper.
    # ``return_only_write=False`` makes ``get_out_shapes`` carry a proper
    # ``jax.ShapeDtypeStruct`` for EVERY state the function touches (read or
    # write), aligned 1:1 with ``state_trace.states``. With the default
    # (write-only) the read-only slots come back as ``None``, which would map
    # read-only states to ``None`` in ``state_shape_map`` and break the
    # documented ``State -> ShapeDtypeStruct`` contract.
    stateful_fn = StatefulFunction(_wrapped, name='eval_shape', return_only_write=False)
    try:
        stateful_fn.make_jaxpr(*g_args, **g_kwargs)
        out_shapes, state_shapes = stateful_fn.get_out_shapes(*g_args, **g_kwargs)

        # Reconstruct a Node (if any) from the abstract output pytree. ``tree_to_graph``
        # builds fresh States bound to the current (top-level) trace, so the reconstructed
        # Node is already a first-class input to subsequent brainstate transformations
        # (nested eval_shape, jit/grad/vmap, init_all_states). For plain-array outputs
        # ``tree_to_graph`` is a no-op passthrough.
        out = tree_to_graph(out_shapes)

        # Build the State -> ShapeDtypeStruct mapping for the optional return. The
        # state_shapes tuple is aligned with state_trace.states.
        state_trace = stateful_fn.get_state_trace(*g_args, **g_kwargs)
        state_shape_map = {st: sh for st, sh in zip(state_trace.states, state_shapes)}
    finally:
        # States created INSIDE f (the ``lambda: LSTMCell(...)`` case) were elevated
        # to the abstract trace's stack level. Restore their original values and drop
        # the stack level so the trace does not leak -- on failure too. Mirrors
        # vmap_new_states.
        new_states = caught_box.get('states', [])
        new_values = caught_box.get('values', [])
        for st, val in zip(new_states, new_values):
            st.restore_value(val)
            st.decrease_stack_level()

    if return_state_shapes:
        return state_shape_map, out
    return out
