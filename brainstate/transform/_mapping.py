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

import functools
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, Dict, Optional, Tuple, Union, TypeVar

import jax
from jax._src import source_info_util

from brainstate._compatible_import import Device, make_iota, to_elt, BatchTracer, BatchTrace
from brainstate._error import BatchAxisError
from brainstate._state import State, StateTraceStack
from brainstate._utils import set_module_as
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict
from brainstate.util.filter import to_predicate
from ._loop_collect_return import scan
from ._make_jaxpr import StatefulFunction, BoundedCache

__all__ = [
    'StatefulMapping',
    'vmap2',
    'pmap',
    'map',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
_rand = None


def _import_rand_state():
    global _rand
    if _rand is None:
        from brainstate.random import RandomState
        _rand = RandomState
    return _rand


class StatefulMapping(StatefulFunction):
    """
    Vectorized wrapper that preserves BrainState state semantics during mapping.

    ``StatefulMapping`` extends JAX mapping transforms (such as :func:`jax.vmap`
    and :func:`jax.pmap`) with awareness of :class:`~brainstate.State`
    instances. It tracks state reads and writes across the mapped axis,
    ensures deterministic random-number handling, and restores side effects
    after each batched execution. The helper is typically constructed by
    :func:`brainstate.transform.vmap` or :func:`brainstate.transform.pmap`, but
    it can also be instantiated directly for custom mapping primitives.

    Parameters
    ----------
    fun : callable
        Stateless callable to be wrapped. The callable may close over
        :class:`~brainstate.State` objects that should be tracked during the
        mapping transform.
    in_axes : int, tuple of int, or None, default 0
        Alignment of the mapped axis per positional argument, following the
        semantics of :func:`jax.vmap`. Arguments mapped with ``None`` are treated
        as static.
    out_axes : int, tuple of int, or None, default 0
        Placement of the mapped axis in the return value, consistent with JAX
        mapping primitives.
    state_in_axes : dict[AxisName, Filter] or Filter, optional
        Specification of input states that participate in the mapped axis. A
        dictionary maps axis identifiers to :mod:`brainstate.util.filter`
        predicates; passing a single filter applies it to axis ``0``. Values are
        normalized via :func:`brainstate.util.filter.to_predicate`.
    state_out_axes : dict[AxisName, Filter] or Filter, optional
        Specification of state outputs to scatter back along the mapped axis.
        Uses the same semantics and normalization as ``state_in_axes``.
    unexpected_out_state_mapping : {'raise', 'warn', 'ignore'}, default 'raise'
        Strategy for handling states written during the mapped call that are not
        captured by ``state_out_axes``.
    axis_size : int, optional
        Explicit size of the mapped axis. When omitted, the size is inferred
        from the mapped arguments.
    axis_name : hashable, optional
        Name for the mapped axis so that collective primitives can target it.
    name : str, optional
        Human-readable identifier for diagnostics and debugging.
    mapping_fn : callable, default ``jax.vmap``
        Mapping primitive that executes ``fun``. The callable must accept the
        ``in_axes`` and ``out_axes`` keyword arguments used by :func:`jax.vmap`.

    Attributes
    ----------
    origin_fun : callable
        Original Python callable wrapped by the mapping helper.
    in_axes : int, tuple of int, or None
        Mapping specification for positional arguments.
    out_axes : int, tuple of int, or None
        Mapping specification for the return value.
    state_in_axes : dict[AxisName, Predicate]
        Normalized predicates describing which states to batch on input.
    state_out_axes : dict[AxisName, Predicate]
        Normalized predicates describing which states to batch on output.
    axis_size : int or None
        Size of the mapped axis, if explicitly provided.
    axis_name : hashable or None
        Axis identifier forwarded to collective primitives.
    mapping_fn : callable
        Mapping primitive responsible for executing ``fun``.

    Raises
    ------
    TypeError
        If ``in_axes`` has an unsupported type.
    ValueError
        If batch dimensions are inconsistent or cannot be inferred.
    RuntimeError
        If tracing or executing the mapped function fails.

    Notes
    -----
    Random states (for example :class:`~brainstate.RandomState`) encountered
    during execution are automatically split along the mapped axis and restored
    afterwards; this behaviour cannot be disabled. The wrapper caches inferred
    state placements, batch sizes, and trace stacks keyed by abstract argument
    signatures so repeated calls with the same structure avoid re-tracing.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> counter = brainstate.ShortTermState(jnp.array(0.0))
        >>>
        >>> def accumulate(x):
        ...     counter.value = counter.value + x
        ...     return counter.value
        >>>
        >>> batched_accumulate = brainstate.transform.StatefulMapping(
        ...     accumulate,
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(brainstate.ShortTermState)},
        ...     state_out_axes={0: OfType(brainstate.ShortTermState)},
        ...     name="batched_accumulate",
        ... )
        >>>
        >>> xs = jnp.ones((3,))
        >>> batched_accumulate(xs)
        Array([1., 2., 3.], dtype=float32)
        >>> counter.value
        Array(3., dtype=float32)

    See Also
    --------
    brainstate.transform.vmap : Convenience API returning a ``StatefulMapping``.
    brainstate.transform.pmap : Device-mapped variant aware of BrainState states.
    """
    __module__ = "brainstate.transform"

    def __init__(
        self,
        fun: Callable,
        in_axes: Union[int, Tuple[int, ...], None] = 0,
        out_axes: Union[int, Tuple[int, ...], None] = 0,
        state_in_axes: Optional[Union[Dict[AxisName, Filter], Filter]] = None,
        state_out_axes: Optional[Union[Dict[AxisName, Filter], Filter]] = None,
        unexpected_out_state_mapping: str = 'raise',
        # JIT specific parameters
        static_argnums: Union[int, Iterable[int]] = (),
        static_argnames: Union[str, Iterable[str]] = (),
        axis_env: Optional[Sequence[tuple[Hashable, int]]] = None,
        return_only_write: bool = True,
        # mapping specific parameters
        axis_size: Optional[int] = None,
        axis_name: AxisName | None = None,
        name: Optional[str] = None,
        # mapping function
        mapping_fn: Callable = jax.vmap,
    ):
        super().__init__(
            self.__wrapped_fun,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            axis_env=axis_env,
            return_only_write=return_only_write,
        )
        self.origin_fun = fun
        self.traced_fn = StatefulFunction(
            fun,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            axis_env=axis_env,
            return_only_write=return_only_write,
        )

        self.name = name
        self.in_axes = in_axes
        self.out_axes = out_axes
        if state_in_axes is None:
            state_in_axes = dict()
        elif not isinstance(state_in_axes, dict):
            state_in_axes = {0: to_predicate(state_in_axes)}
        state_in_axes = {k: to_predicate(v) for k, v in state_in_axes.items()}  # type: ignore
        self.state_in_axes = state_in_axes

        if state_out_axes is None:
            state_out_axes = dict()
        elif not isinstance(state_out_axes, dict):
            state_out_axes = {0: to_predicate(state_out_axes)}
        state_out_axes = {k: to_predicate(v) for k, v in state_out_axes.items()}  # type: ignore
        self.state_out_axes = state_out_axes

        self.axis_size = axis_size
        self.axis_name = axis_name
        self.mapping_fn = mapping_fn
        self.unexpected_out_state_mapping = unexpected_out_state_mapping

        # Cache for discovered state-to-axis mappings
        self._cached_map_dim_to_in_states = BoundedCache(maxsize=128)
        self._cached_map_dim_to_out_states = BoundedCache(maxsize=128)
        self._cached_map_state_trace = BoundedCache(maxsize=128)
        self._cached_map_batch_size = BoundedCache(maxsize=128)

    def __infer_batch_size(self, args, in_axes):
        def get_batch_size_from_arg(arg_, axis_):
            if axis_ is None:
                return None

            def _get_size(arr):
                if not hasattr(arr, 'shape'):
                    return None
                if arr.ndim == 0:
                    return None
                ax = axis_ if axis_ >= 0 else arr.ndim + axis_
                if ax < 0 or ax >= arr.ndim:
                    raise IndexError(f"Axis {ax} is out of bounds for array of shape {arr.shape}")
                return arr.shape[ax]

            # Get all sizes from the pytree
            sizes = [s for s in jax.tree.leaves(jax.tree.map(_get_size, arg_)) if s is not None]
            return sizes[0] if sizes else None

        batch_sizes = []
        if isinstance(in_axes, int):
            # All args batched along the same axis
            for arg in args:
                size = get_batch_size_from_arg(arg, in_axes)
                if size is not None:
                    batch_sizes.append(size)
        elif isinstance(in_axes, (tuple, list)):
            # Different axes for different args
            if len(in_axes) != len(args):
                raise ValueError(
                    f"Length of in_axes ({len(in_axes)}) must match number of arguments ({len(args)})"
                )
            for arg, axis in zip(args, in_axes):
                size = get_batch_size_from_arg(arg, axis)
                if size is not None:
                    batch_sizes.append(size)
        elif in_axes is None:
            pass
        else:
            raise TypeError(f"Unsupported in_axes type: {type(in_axes)}")

        if not batch_sizes:
            if self.axis_size is None:
                raise ValueError("Cannot infer batch size when axis_size is None")
            batch_sizes.append(self.axis_size)

        # Check all batch sizes are consistent
        if not all(s == batch_sizes[0] for s in batch_sizes):
            raise ValueError(
                f"Inconsistent batch sizes found: {batch_sizes}. "
                f"All batched arguments must have the same size along their batch axes."
            )

        return batch_sizes[0]

    def __new_batch_arg(self, trace, batch_size: int, dim_to_states: dict):
        RandomState = _import_rand_state()

        def wrapper(x):
            if isinstance(x, RandomState):
                idx = lambda: BatchTracer(trace, make_iota(batch_size), 0, source_info_util.current())
                dim_to_states['random'].append(x)
                return to_elt(trace, idx, self._rand_value, 0)
            for dim, filter_ in self.state_in_axes.items():
                idx = lambda: BatchTracer(trace, make_iota(batch_size), dim, source_info_util.current())
                if filter_(tuple(), x):
                    dim_to_states[dim].append(x)
                    return jax.tree.map(lambda xx: to_elt(trace, idx, xx, dim), x._value)
            return x._value

        return wrapper

    def __find_batch_dim(self, st):
        leaves = jax.tree.leaves(st._value)
        batch_dims = set([leaf.batch_dim if isinstance(leaf, BatchTracer) else None for leaf in leaves])
        if len(batch_dims) != 1:
            raise ValueError(
                f"State {st} has inconsistent batch dimensions in its leaves: {batch_dims}. "
                "All leaves must have the same batch dimension."
            )
        dim = batch_dims.pop()
        return dim

    def __fn_to_eval(self, cache_key, *new_args, **new_kwargs):
        RandomState = _import_rand_state()
        if len(new_kwargs):
            raise NotImplementedError(
                'StatefulMapping currently does not support keyword arguments.'
            )

        # state trace
        trace = jax.core.trace_ctx.trace
        assert isinstance(trace, BatchTrace), f"Expected to be called within a BatchTrace context, but got {trace}"
        dim_to_in_states = defaultdict(list)
        state_trace = StateTraceStack(name=self.name)
        state_trace.set_new_arg(
            self.__new_batch_arg(trace, self._cached_map_batch_size.get(cache_key), dim_to_in_states)
        )
        self._cached_map_state_trace.set(cache_key, state_trace)

        # call functions
        with state_trace:
            out_ = self.traced_fn(*new_args)

        # cache vmapped in states
        self._cached_map_dim_to_in_states.set(cache_key, dim_to_in_states.copy())
        mapped_in_states = set([id(v) for vv in dim_to_in_states.values() for v in vv])

        # vmapped out states
        out_states = defaultdict(list)
        out_states['random'] = [st for st in state_trace.states if isinstance(st, RandomState)]
        for st in state_trace.states:
            if isinstance(st, RandomState):
                continue
            find = False
            for dim, filter_ in self.state_out_axes.items():
                if filter_(tuple(), st):
                    out_states[dim].append(st)
                    find = True
                    break
            if find:
                continue
            dim = self.__find_batch_dim(st)
            if dim is None or id(st) in mapped_in_states:
                out_states[dim].append(st)
            else:
                if self.unexpected_out_state_mapping == 'raise':
                    st.raise_error_with_source_info(
                        BatchAxisError(
                            f'State\n {st} \n was not expected to be batched on output. '
                            'Please adjust state_out_axes or set unexpected_out_state_mapping to "warn" or "ignore".'
                        )
                    )
                elif self.unexpected_out_state_mapping == 'warn':
                    warnings.warn(
                        f'State\n {st} \n was not expected to be batched on output. '
                        f'Please adjust state_out_axes or set unexpected_out_state_mapping to "ignore".',
                        UserWarning,
                    )
                    out_states[dim].append(st)
                elif self.unexpected_out_state_mapping == 'ignore':
                    out_states[dim].append(st)
                else:
                    raise ValueError(
                        'Invalid value for unexpected_out_state_mapping: '
                        f'{self.unexpected_out_state_mapping}. Must be "raise", "warn", or "ignore".'
                    )
        self._cached_map_dim_to_out_states.set(cache_key, out_states)

    def __eval(self, cache_key, *args, **kwargs):
        try:
            jax.vmap(
                functools.partial(self.__fn_to_eval, cache_key),
                in_axes=self.in_axes,
                axis_name=self.axis_name,
                axis_size=self.axis_size
            )(*args, **kwargs)
            self._cached_map_state_trace.get(cache_key).recovery_original_values()
        except Exception as e:
            if cache_key in self._cached_map_state_trace:
                self._cached_map_state_trace.get(cache_key).recovery_original_values()
            self._cached_map_state_trace.pop(cache_key, None)
            self._cached_map_dim_to_in_states.pop(cache_key, None)
            self._cached_map_dim_to_out_states.pop(cache_key, None)
            self._cached_map_batch_size.pop(cache_key, None)
            raise e

    def __assign_vals_from_in_states(self, cache_key, rand_st, *other_st):
        RandomState = _import_rand_state()
        in_states = self._cached_map_dim_to_in_states.get(cache_key)
        for st, val in zip(in_states['random'], rand_st):
            assert isinstance(st, RandomState)
            st.restore_value(val)
        for group, group_vals in zip([in_states[dim] for dim in in_states.keys() if dim != 'random'], other_st):
            for st, val in zip(group, group_vals):
                st.restore_value(val)

    def __assign_vals_from_out_states(self, cache_key, rand_st, *other_st):
        RandomState = _import_rand_state()
        out_states = self._cached_map_dim_to_out_states.get(cache_key)
        for st, val in zip(out_states['random'], rand_st):
            assert isinstance(st, RandomState)
            st.restore_value(val)
        for group, group_vals in zip([out_states[dim] for dim in out_states.keys() if dim != 'random'], other_st):
            for st, val in zip(group, group_vals):
                st.restore_value(val)

    def __get_in_state_vals(self, cache_key: Hashable):
        in_states = self._cached_map_dim_to_in_states.get(cache_key)
        in_axes = []
        in_values = []
        for dim, states in in_states.items():
            if dim == 'random':
                continue
            in_axes.append(dim)
            in_values.append([st.value for st in states])
        return tuple(in_axes), in_values

    def __get_out_state_vals(self, cache_key: Hashable):
        out_states = self._cached_map_dim_to_out_states.get(cache_key)
        out_axes = []
        out_values = []
        for dim, state in out_states.items():
            if dim == 'random':
                continue
            out_axes.append(dim)
            out_values.append([st.value for st in state])
        return tuple(out_axes), out_values

    def __get_rand_state_vals(self, cache_key: Hashable):
        RandomState = _import_rand_state()
        in_states = self._cached_map_dim_to_in_states.get(cache_key)
        batch_size = self._cached_map_batch_size.get(cache_key)
        rand_vals, rand_recover_vals = [], []
        for st in in_states['random']:
            assert isinstance(st, RandomState)
            rand_vals.append(st.split_key(batch_size))
            rand_recover_vals.append(st.value)
        return tuple(rand_vals), tuple(rand_recover_vals)

    def __wrapped_fun(self, *args, **kwargs) -> Tuple[Any, Tuple[State, ...]]:
        RandomState = _import_rand_state()
        if len(kwargs):
            raise NotImplementedError(
                'StatefulMapping currently does not support keyword arguments.'
            )

        batch_size = self.__infer_batch_size(args, self.in_axes)
        cache_key = self.get_arg_cache_key(*args, **kwargs)
        if cache_key not in self._cached_map_state_trace:
            self._rand_value = RandomState._batch_keys(batch_size)
            self._cached_map_batch_size.set(cache_key, batch_size)
            self.__eval(cache_key, *args, **kwargs)

        def fn_to_map(origin_args, rand_st, *non_rand_st):
            self.__assign_vals_from_in_states(cache_key, rand_st, *non_rand_st)
            out = self.traced_fn(*origin_args)
            return out, *self.__get_out_state_vals(cache_key)[1]

        in_axes, in_state_vals = self.__get_in_state_vals(cache_key)
        out_axes, out_state_vals = self.__get_out_state_vals(cache_key)
        rand_vals, rand_recover_vals = self.__get_rand_state_vals(cache_key)
        mapped_fn = self.mapping_fn(
            fn_to_map,
            in_axes=(self.in_axes, 0 if len(rand_vals) else None) + in_axes,
            out_axes=(self.out_axes,) + out_axes,
            axis_size=self.axis_size,
            axis_name=self.axis_name,
        )
        out_, *out_state_vals = mapped_fn(args, rand_vals, *in_state_vals)
        self.__assign_vals_from_out_states(cache_key, rand_recover_vals, *out_state_vals)
        return out_


@set_module_as('brainstate.transform')
def vmap2(
    fn: F | Missing = Missing(),
    *,
    # --- normal jax.vmap arguments --- #
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # --- brainstate specific arguments --- #
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'raise',
) -> StatefulMapping | Callable[[F], StatefulMapping]:
    """
    Vectorize a callable while preserving BrainState state semantics.

    This helper mirrors :func:`jax.vmap` but routes execution through
    :class:`~brainstate.transform.StatefulMapping` so that reads and writes to
    :class:`~brainstate.State` instances (including newly created random states)
    are tracked correctly across the mapped axis. The returned object can be used
    directly or as a decorator when ``fn`` is omitted.

    Parameters
    ----------
    fn : callable, optional
        Function to be vectorised. If omitted, the function acts as a decorator.
    in_axes : int | None | sequence, default 0
        Mapping specification for positional arguments, following the semantics
        of :func:`jax.vmap`.
    out_axes : any, default 0
        Placement of the mapped axis in the result. Must broadcast with the
        structure of the outputs.
    axis_name : hashable, optional
        Name for the mapped axis so that collective primitives (e.g. ``lax.psum``)
        can target it.
    axis_size : int, optional
        Explicit size of the mapped axis. If omitted, the size is inferred from
        the arguments.
    spmd_axis_name : hashable or tuple[hashable], optional
        Axis labels used when the transformed function is itself executed inside
        another SPMD transform (e.g. nested :func:`vmap` or :func:`pmap`).
    state_in_axes : dict[AxisName, Filter] or Filter, optional
        Filters identifying which :class:`State` objects should be batched on
        input. Passing a single filter is shorthand for ``{0: filter}``. Filters
        are converted with :func:`brainstate.util.filter.to_predicate`.
    state_out_axes : dict[AxisName, Filter] or Filter, optional
        Filters describing how written states are scattered back across the
        mapped axis. Semantics mirror ``state_in_axes``.
    unexpected_out_state_mapping : {'raise', 'warn', 'ignore'}, default 'raise'
        Policy when a state is written during the mapped call but not matched by
        ``state_out_axes``. ``'raise'`` propagates a :class:`BatchAxisError`,
        ``'warn'`` emits a warning, and ``'ignore'`` silently accepts the state.

    Returns
    -------
    StatefulMapping or callable
        If ``fn`` is supplied, returns a :class:`StatefulMapping` instance that
        behaves like ``fn`` but with batch semantics. Otherwise a decorator is
        returned.

    Raises
    ------
    ValueError
        If axis sizes are inconsistent or cannot be inferred.
    BatchAxisError
        If a state write violates ``state_out_axes`` and the policy is ``'raise'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> counter = brainstate.ShortTermState(jnp.array(0.0))
        >>>
        >>> @brainstate.transform.vmap(
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(brainstate.ShortTermState)},
        ...     state_out_axes={0: OfType(brainstate.ShortTermState)},
        ... )
        ... def accumulate(x):
        ...     counter.value = counter.value + x
        ...     return counter.value
        >>>
        >>> xs = jnp.arange(3.0)
        >>> accumulate(xs)
        Array([0., 1., 3.], dtype=float32)
        >>> counter.value
        Array(3., dtype=float32)

    See Also
    --------
    brainstate.transform.StatefulMapping : Underlying state-aware mapping helper.
    pmap : Parallel mapping variant for multiple devices.
    vmap_new_states : Vectorize newly created states within ``fn``.
    """

    if isinstance(fn, Missing):
        return functools.partial(
            vmap2,
            in_axes=in_axes,
            out_axes=out_axes,
            state_in_axes=state_in_axes,
            state_out_axes=state_out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )  # type: ignore[return-value]

    return StatefulMapping(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        unexpected_out_state_mapping=unexpected_out_state_mapping,
        mapping_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
        name='vmap2'
    )


@set_module_as('brainstate.transform')
def pmap(
    fn: Callable[[NestedDict, ...], Any] | Missing = Missing(),
    axis_name: Optional[AxisName] = None,
    *,
    in_axes: Any = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Optional[Sequence[Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: int | Iterable[int] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    # --- brainstate specific arguments --- #
    state_in_axes: Union[Dict[AxisName, Filter], Filter] = None,
    state_out_axes: Union[Dict[AxisName, Filter], Filter] = None,
    unexpected_out_state_mapping: str = 'raise',
) -> Callable[[F], F] | F:
    """
    Parallel mapping with state-aware semantics across devices.

    This function mirrors :func:`jax.pmap` but integrates with
    :class:`~brainstate.transform.StatefulMapping` so that
    :class:`~brainstate.State` objects (including random states) are replicated
    and restored correctly on every device. When ``fn`` is omitted the function
    can be used as a decorator.

    Parameters
    ----------
    fn : callable, optional
        Function to execute in SPMD style. If omitted, a decorator is returned.
    axis_name : hashable, optional
        Name for the mapped axis used by collective primitives.
    in_axes : any, default 0
        Axis mapping for positional arguments, identical to :func:`jax.pmap`.
    out_axes : any, default 0
        Placement of the mapped axis in the outputs.
    static_broadcasted_argnums : int or iterable[int], default ()
        Indices of positional arguments to treat as compile-time constants.
    devices : sequence[Device], optional
        Explicit device list to map over. Must be identical on every host in
        multi-host setups.
    backend : str, optional
        Backend identifier (``'cpu'``, ``'gpu'``, or ``'tpu'``).
    axis_size : int, optional
        Size of the mapped axis. Defaults to ``len(devices)`` or the local device
        count when ``devices`` is ``None``.
    donate_argnums : int or iterable[int], default ()
        Positional arguments whose buffers may be donated to the computation.
    global_arg_shapes : tuple[tuple[int, ...], ...], optional
        Shapes for globally distributed arguments (i.e. arguments not replicated
        across devices).
    state_in_axes : dict[AxisName, Filter] or Filter, optional
        Filters indicating which states should be treated as device-mapped inputs.
    state_out_axes : dict[AxisName, Filter] or Filter, optional
        Filters describing how state writes are scattered back to devices.
    unexpected_out_state_mapping : {'raise', 'warn', 'ignore'}, default 'raise'
        Policy applied when a state write is not covered by ``state_out_axes``.

    Returns
    -------
    StatefulMapping or callable
        If ``fn`` is provided, returns a :class:`StatefulMapping` executing ``fn``
        over devices. Otherwise returns a decorator that produces such an object.

    Raises
    ------
    ValueError
        If ``axis_size`` or argument shapes are inconsistent.
    BatchAxisError
        If an unexpected state write occurs and the policy is ``'raise'``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> from brainstate.util.filter import OfType
        >>>
        >>> weights = brainstate.ParamState(jnp.ones((4,)))
        >>>
        >>> @brainstate.transform.pmap(
        ...     axis_name='devices',
        ...     in_axes=0,
        ...     out_axes=0,
        ...     state_in_axes={0: OfType(brainstate.ParamState)},
        ...     state_out_axes={0: OfType(brainstate.ParamState)},
        ... )
        ... def update(delta):
        ...     weights.value = weights.value + delta
        ...     return weights.value
        >>>
        >>> deltas = jnp.arange(jax.local_device_count() * 4.).reshape(
        ...     jax.local_device_count(), 4
        ... )
        >>> updated = update(deltas)
        >>> updated.shape
        (jax.local_device_count(), 4)

    See Also
    --------
    jax.pmap : Underlying JAX primitive.
    vmap : Single-host vectorisation with the same state semantics.
    """

    if isinstance(fn, Missing):
        return functools.partial(
            pmap,
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices,
            backend=backend,
            axis_size=axis_size,
            donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )  # type: ignore[return-value]

    return StatefulMapping(
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        state_in_axes=state_in_axes,
        state_out_axes=state_out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        mapping_fn=functools.partial(
            jax.pmap,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices,
            backend=backend,
            donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
        ),
        unexpected_out_state_mapping=unexpected_out_state_mapping,
        name='pmap'
    )


def _batch_and_remainder(x, batch_size: int):
    leaves, tree_def = jax.tree.flatten(x)

    scan_leaves = []
    remainder_leaves = []

    length = None
    for leaf in leaves:
        if length is None:
            length = leaf.shape[0]
        if length != leaf.shape[0]:
            raise ValueError(f"All inputs must have the same length. Got {length} and {leaf.shape[0]}.")

    num_batches, num_remainder = divmod(length, batch_size)
    for leaf in leaves:
        total_batch_elems = num_batches * batch_size
        scan_leaves.append(leaf[:total_batch_elems].reshape(num_batches, batch_size, *leaf.shape[1:]))
        if num_remainder:
            remainder_leaves.append(leaf[total_batch_elems:])

    scan_tree = tree_def.unflatten(scan_leaves)
    if num_remainder:
        remainder_tree = tree_def.unflatten(remainder_leaves)
        return scan_tree, remainder_tree
    else:
        return scan_tree, None


@set_module_as('brainstate.transform')
def map(
    f,
    *xs,
    batch_size: int | None = None,
):
    """
    Apply a Python function over the leading axis of one or more pytrees.

    Compared with :func:`jax.vmap`, this helper executes sequentially by default
    (via :func:`jax.lax.scan`), making it useful when auto-vectorisation is
    impractical or when memory usage must be reduced. Providing ``batch_size``
    enables chunked evaluation that internally leverages :func:`vmap` to improve
    throughput while keeping peak memory bounded.

    Parameters
    ----------
    f : callable
        Function applied element-wise across the leading dimension. Its return
        value must be a pytree whose leaves can be stacked along axis ``0``.
    *xs : Any
        Positional pytrees sharing the same length along their leading axis.
    batch_size : int, optional
        Size of vectorised blocks. When given, ``map`` first processes full
        batches using :func:`vmap` then handles any remainder sequentially.

    Returns
    -------
    Any
        PyTree matching the structure of ``f``'s outputs with results stacked
        along the leading dimension.

    Raises
    ------
    ValueError
        If the inputs do not share the same leading length.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainstate.transform import map
        >>>
        >>> xs = jnp.arange(6).reshape(6, 1)
        >>>
        >>> def normalize(row):
        ...     return row / (1.0 + jnp.linalg.norm(row))
        >>>
        >>> stacked = map(normalize, xs, batch_size=2)
        >>> stacked.shape
        (6, 1)

    See Also
    --------
    vmap : Vectorised mapping with automatic batching.
    jax.lax.scan : Primitive used for the sequential fallback.
    """
    if batch_size is not None:
        from ._mapping_old import vmap
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
        g = lambda _, x: ((), vmap(f)(*x))
        _, scan_ys = scan(g, (), scan_xs)
        if remainder_xs is None:
            ys = jax.tree.map(lambda x: _flatten(x), scan_ys)
        else:
            remainder_ys = vmap(f)(*remainder_xs)
            ys = jax.tree.map(
                lambda x, y: jax.lax.concatenate([_flatten(x), y], dimension=0),
                scan_ys,
                remainder_ys,
            )
    else:
        g = lambda _, x: ((), f(*x))
        _, ys = scan(g, (), xs)
    return ys


def _flatten(x):
    return x.reshape(-1, *x.shape[2:])
