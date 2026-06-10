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

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import jax
from jax.sharding import NamedSharding, PartitionSpec

from brainstate._compatible_import import jax_shard_map, SHARD_MAP_CHECK_KW
from brainstate._state import State
from brainstate._utils import set_module_as
from ._make_jaxpr import StatefulFunction

__all__ = ['shard_map']


def _prep_spec_table(table):
    """Pre-key a {State: PartitionSpec} table by id; pass through None / single spec."""
    if isinstance(table, dict):
        return {id(k): v for k, v in table.items()}
    return table


def _resolve_state_spec(state: State, table) -> PartitionSpec:
    """Resolve the PartitionSpec for a state: replicate by default."""
    if table is None:
        return PartitionSpec()
    if isinstance(table, dict):
        return table.get(id(state), PartitionSpec())
    return table  # a single PartitionSpec applied to all states


@set_module_as("brainstate.transform")
def shard_map(
    fun: Callable,
    mesh: jax.sharding.Mesh,
    in_specs: Any,
    out_specs: Any,
    *,
    state_in_specs: Optional[Union[PartitionSpec, Dict[State, PartitionSpec]]] = None,
    state_out_specs: Optional[Union[PartitionSpec, Dict[State, PartitionSpec]]] = None,
    check_vma: bool = True,
) -> Callable:
    """Map a stateful function over shards of data across a device mesh (SPMD).

    A state-aware wrapper over :func:`jax.shard_map`. The function's
    :class:`~brainstate.State` objects are sharded or replicated across ``mesh``
    according to ``state_in_specs``/``state_out_specs`` (replicated by default),
    while positional arguments are sharded per ``in_specs``. Inputs are placed on
    the mesh automatically (via :func:`jax.device_put`), so the wrapper works
    both eagerly and under :func:`jit`.

    Parameters
    ----------
    fun : callable
        The function to shard. May read and write ``State`` objects. Keyword
        arguments are broadcast (closed over), not sharded.
    mesh : jax.sharding.Mesh
        The device mesh, e.g. ``jax.make_mesh((4,), ('x',))``.
    in_specs : PartitionSpec or tuple of PartitionSpec
        Sharding spec for each positional argument. A tuple must match the
        number of positional arguments; a single spec is applied to all.
    out_specs : PartitionSpec or PyTree of PartitionSpec
        Sharding spec for the function's output.
    state_in_specs : PartitionSpec or dict of {State: PartitionSpec}, optional
        Input sharding for states. A single spec applies to all states; a dict
        maps specific states. States not covered are replicated (``PartitionSpec()``).
    state_out_specs : PartitionSpec or dict of {State: PartitionSpec}, optional
        Output sharding for written states. Same conventions as
        ``state_in_specs``.
    check_vma : bool, default True
        Forwarded to :func:`jax.shard_map` (varying-manual-axes checking).

    Returns
    -------
    callable
        A function with the same positional signature as ``fun`` that executes
        under SPMD sharding and applies state writes.

    See Also
    --------
    jax.shard_map, vmap, pmap

    Notes
    -----
    ``jax.shard_map`` traces ``fun`` at per-shard shapes, so the wrapper re-runs
    ``fun`` (rather than replaying a global jaxpr): it discovers the touched
    states once via :class:`StatefulFunction`, injects per-shard values with
    ``State.restore_value``, runs ``fun``, and restores every state afterward
    (writes to their new values, reads to their originals).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> from jax.sharding import PartitionSpec as P
        >>>
        >>> mesh = jax.make_mesh((jax.device_count(),), ('x',))
        >>> w = brainstate.State(jnp.array(3.0))
        >>> def fun(data):
        ...     return data * w.value
        >>> f = brainstate.transform.shard_map(fun, mesh, in_specs=(P('x'),), out_specs=P('x'))
        >>> f(jnp.arange(jax.device_count() * 2, dtype=jnp.float32))  # doctest: +SKIP

    Keep a per-shard buffer by giving a state an explicit partition through
    ``state_in_specs`` / ``state_out_specs``; the buffer is read and written
    in place on each device:

    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> from jax.sharding import PartitionSpec as P
        >>> mesh = jax.make_mesh((jax.device_count(),), ('x',))
        >>> buffer = brainstate.State(jnp.zeros(jax.device_count() * 2))
        >>> def accumulate(data):
        ...     buffer.value = buffer.value + data
        ...     return data
        >>> f = brainstate.transform.shard_map(
        ...     accumulate, mesh, in_specs=(P('x'),), out_specs=P('x'),
        ...     state_in_specs={buffer: P('x')}, state_out_specs={buffer: P('x')})
        >>> _ = f(jnp.ones(jax.device_count() * 2))  # doctest: +SKIP
        >>> buffer.value  # doctest: +SKIP

    Communicate across shards with collectives such as :func:`jax.lax.psum`,
    referring to the mesh axis by name. Here each device contributes a partial
    sum and ``psum`` reduces them to the global total (replicated back):

    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> from jax.sharding import PartitionSpec as P
        >>> mesh = jax.make_mesh((jax.device_count(),), ('x',))
        >>> def global_sum(data):
        ...     return jax.lax.psum(jnp.sum(data, keepdims=True), axis_name='x')
        >>> f = brainstate.transform.shard_map(
        ...     global_sum, mesh, in_specs=(P('x'),), out_specs=P())
        >>> f(jnp.arange(jax.device_count() * 2, dtype=jnp.float32))  # doctest: +SKIP

    ``shard_map`` re-traces ``fun`` on each call to discover its state usage;
    wrap it in :func:`jax.jit` to amortise that on the hot path:

    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> from jax.sharding import PartitionSpec as P
        >>> mesh = jax.make_mesh((jax.device_count(),), ('x',))
        >>> bias = brainstate.State(jnp.array(5.0))
        >>> f = brainstate.transform.shard_map(
        ...     lambda data: data + bias.value, mesh,
        ...     in_specs=(P('x'),), out_specs=P('x'))
        >>> jit_f = jax.jit(f)
        >>> jit_f(jnp.arange(jax.device_count() * 2, dtype=jnp.float32))  # doctest: +SKIP

    Multi-axis meshes express combined data- and model-parallel shardings by
    naming each axis in the :class:`~jax.sharding.PartitionSpec`:

    .. code-block:: python

        >>> import jax, jax.numpy as jnp
        >>> import brainstate
        >>> from jax.sharding import PartitionSpec as P
        >>> n = jax.device_count()
        >>> mesh2d = jax.make_mesh((n // 2, 2), ('data', 'model'))  # needs n >= 2
        >>> f = brainstate.transform.shard_map(
        ...     lambda data: data + 1.0, mesh2d,
        ...     in_specs=(P('data', 'model'),), out_specs=P('data', 'model'))
        >>> f(jnp.ones((n // 2, 2)))  # doctest: +SKIP
    """
    arg_specs = tuple(in_specs) if isinstance(in_specs, (tuple, list)) else None
    sin = _prep_spec_table(state_in_specs)
    sout = _prep_spec_table(state_out_specs)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # 1. Discover the states the function touches (shape-independent).
        #    Pass the mesh axis environment so collectives such as
        #    ``jax.lax.psum`` inside ``fun`` can be traced during state
        #    discovery without raising an "unbound axis name" error.
        sf = StatefulFunction(fun, name='shard_map', axis_env=tuple(mesh.shape.items()))
        sf.make_jaxpr(*args, **kwargs)
        cache = sf.get_arg_cache_key(*args, **kwargs)
        trace = sf.get_state_trace_by_cache(cache)
        all_states = tuple(trace.states)
        write_states = tuple(sf.get_write_states(*args, **kwargs))

        # 2. Resolve per-argument and per-state specs.
        if arg_specs is None:
            local_arg_specs = tuple(in_specs for _ in args)
        else:
            if len(arg_specs) != len(args):
                raise ValueError(
                    f"in_specs has {len(arg_specs)} entries but {len(args)} "
                    "positional arguments were given."
                )
            local_arg_specs = arg_specs
        in_state_specs = tuple(_resolve_state_spec(s, sin) for s in all_states)
        out_state_specs = tuple(_resolve_state_spec(s, sout) for s in write_states)
        orig_vals = tuple(s.value for s in all_states)

        # 3. Build the re-runnable pure function (trace fresh at shard shapes).
        def pure(state_vals, mapped_args):
            for st, v in zip(all_states, state_vals):
                st.restore_value(v)
            out = fun(*mapped_args, **kwargs)
            return out, tuple(st.value for st in write_states)

        # 4. Place inputs on the mesh per their specs (required: no auto-reshard).
        in_state_vals = tuple(
            jax.device_put(v, NamedSharding(mesh, sp))
            for v, sp in zip(orig_vals, in_state_specs)
        )
        sharded_args = tuple(
            jax.device_put(a, NamedSharding(mesh, sp))
            for a, sp in zip(args, local_arg_specs)
        )

        # 5. Run under jax.shard_map (flag name varies across jax versions).
        sm_kwargs = dict(
            mesh=mesh,
            in_specs=(in_state_specs, local_arg_specs),
            out_specs=(out_specs, out_state_specs),
        )
        if SHARD_MAP_CHECK_KW is not None:
            sm_kwargs[SHARD_MAP_CHECK_KW] = check_vma
        sharded = jax_shard_map(pure, **sm_kwargs)
        try:
            out, write_vals = sharded(in_state_vals, sharded_args)
        except Exception:
            # a failure mid-trace must not leave shard tracers in the states
            for st, ov in zip(all_states, orig_vals):
                st.restore_value(ov)
            raise

        # 6. Restore ALL states: writes -> new values, reads -> originals
        #    (prevents shard tracers from leaking into global State objects).
        wv_by_id = {id(s): v for s, v in zip(write_states, write_vals)}
        for st, ov in zip(all_states, orig_vals):
            st.restore_value(wv_by_id[id(st)] if id(st) in wv_by_id else ov)

        return out

    return wrapped
