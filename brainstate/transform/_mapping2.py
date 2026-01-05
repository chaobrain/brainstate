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
from brainstate._state import State, StateTraceStack, NonBatchState, catch_new_states
from brainstate._utils import set_module_as
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict
from brainstate.util import filter
from ._loop_collect_return import scan
from ._make_jaxpr import StatefulFunction, BoundedCache, get_arg_cache_key

__all__ = [
    'StatefulMapping',
    'vmap2',
    'vmap2_new_states',
    'pmap2',
    'pmap2_new_states',
    'map',
]

F = TypeVar("F", bound=Callable)
AxisName = Hashable
_rand = None
INIT_NO_BATCHING = 'INIT_NO_BATCHING'


def _import_rand_state():
    global _rand
    if _rand is None:
        from brainstate.random import RandomState
        _rand = RandomState
    return _rand


class StatefulMapping:
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
        mapping_kwargs: Dict = None
    ):
        self.static_argnums = static_argnums,
        self.static_argnames = static_argnames,
        self.axis_env = axis_env,
        self.return_only_write = return_only_write,
        self.origin_fun = fun
        self.traced_fn = StatefulFunction(
            fun,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            axis_env=axis_env,
            return_only_write=return_only_write,
            name='vmap2_eval'
        )

        self.name = name
        self.in_axes = in_axes
        self.out_axes = out_axes
        if state_in_axes is None:
            state_in_axes = dict()
        elif not isinstance(state_in_axes, dict):
            state_in_axes = {0: filter.to_predicate(state_in_axes)}
        state_in_axes = {
            k: filter.to_predicate(v)
            for k, v in state_in_axes.items()
        }  # type: ignore
        self.state_in_axes = state_in_axes

        if state_out_axes is None:
            state_out_axes = dict()
        elif not isinstance(state_out_axes, dict):
            state_out_axes = {0: filter.to_predicate(state_out_axes)}
        state_out_axes = {k: filter.to_predicate(v) for k, v in state_out_axes.items()}  # type: ignore
        self.state_out_axes = state_out_axes

        self.axis_size = axis_size
        self.axis_name = axis_name
        self.mapping_fn = mapping_fn
        self.mapping_kwargs = dict() if mapping_kwargs is None else mapping_kwargs
        self.unexpected_out_state_mapping = unexpected_out_state_mapping

        # Cache for discovered state-to-axis mappings
        self._cached_map_dim_to_in_states = BoundedCache(maxsize=128)
        self._cached_map_dim_to_out_states = BoundedCache(maxsize=128)
        self._cached_map_state_trace = BoundedCache(maxsize=128)
        self._cached_map_batch_size = BoundedCache(maxsize=128)

    def __infer_batch_size(self, args, in_axes):
        """Infer the batch size from arguments and their mapping axes.

        Parameters
        ----------
        args : tuple
            Positional arguments to be mapped.
        in_axes : int, tuple, list, or None
            Axis specification for each argument.

        Returns
        -------
        int
            Inferred batch size.

        Raises
        ------
        ValueError
            If batch sizes are inconsistent across arguments or cannot be inferred.
        TypeError
            If in_axes has an unsupported type.
        """

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
        """Create a wrapper that handles batching of state arguments.

        Parameters
        ----------
        trace : BatchTrace
            JAX batch trace context.
        batch_size : int
            Size of the batch dimension.
        dim_to_states : dict
            Dictionary mapping dimensions to lists of states.

        Returns
        -------
        callable
            Wrapper function that processes state values for batching.
        """
        RandomState = _import_rand_state()

        def wrapper(x):
            if isinstance(x, RandomState):
                idx = lambda: BatchTracer(trace, make_iota(batch_size), 0, source_info_util.current())
                dim_to_states['random'].append(x)
                return to_elt(trace, idx, x._numpy_keys(batch_size), 0)
            for dim, filter_ in self.state_in_axes.items():
                idx = lambda: BatchTracer(trace, make_iota(batch_size), dim, source_info_util.current())
                if filter_(tuple(), x):
                    dim_to_states[dim].append(x)
                    return jax.tree.map(lambda xx: to_elt(trace, idx, xx, dim), x._value)
            return x._value

        return wrapper

    def __find_batch_dim(self, st):
        """Find the batch dimension of a state by examining its leaves.

        Parameters
        ----------
        st : State
            State object to analyze.

        Returns
        -------
        int or None
            The batch dimension if all leaves agree, otherwise None.

        Raises
        ------
        ValueError
            If the state has inconsistent batch dimensions across its leaves.
        """
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
            find_dim = self.__find_batch_dim(st)
            find = False
            for dim, filter_ in self.state_out_axes.items():
                if filter_(tuple(), st):
                    out_states[dim].append(st)
                    if dim is None and find_dim is not None:
                        raise BatchAxisError(
                            f''
                        )
                    find = True
                    break
            if find:
                continue
            if find_dim is None or id(st) in mapped_in_states:
                out_states[find_dim].append(st)
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
                    out_states[find_dim].append(st)
                elif self.unexpected_out_state_mapping == 'ignore':
                    out_states[find_dim].append(st)
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

    def __call__(self, *args, **kwargs):
        """Execute the stateful mapping on the given arguments.

        Parameters
        ----------
        *args
            Positional arguments to be mapped over.
        **kwargs
            Keyword arguments (currently not supported).

        Returns
        -------
        Any
            Result of the mapped computation with state updates applied.

        Raises
        ------
        NotImplementedError
            If keyword arguments are provided.
        ValueError
            If batch sizes cannot be inferred or are inconsistent.
        """
        if len(kwargs):
            raise NotImplementedError(
                'StatefulMapping currently does not support keyword arguments.'
            )

        batch_size = self.__infer_batch_size(args, self.in_axes)
        cache_key = get_arg_cache_key(self.static_argnums, self.static_argnames, args, kwargs)
        if cache_key not in self._cached_map_state_trace:
            self._cached_map_batch_size.set(cache_key, batch_size)
            self.__eval(cache_key, *args, **kwargs)

        def fn_to_map(origin_args, rand_st, *non_rand_st):
            self.__assign_vals_from_in_states(cache_key, rand_st, *non_rand_st)
            out = self.traced_fn(*origin_args)
            state_outs = self.__get_out_state_vals(cache_key)[1]
            return out, *state_outs

        in_axes, in_state_vals = self.__get_in_state_vals(cache_key)
        out_axes, out_state_vals = self.__get_out_state_vals(cache_key)
        rand_vals, rand_recover_vals = self.__get_rand_state_vals(cache_key)
        mapped_fn = self.mapping_fn(
            fn_to_map,
            in_axes=(self.in_axes, 0 if len(rand_vals) else None) + in_axes,
            out_axes=(self.out_axes,) + out_axes,
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            **self.mapping_kwargs
        )
        out_, *out_state_vals = mapped_fn(args, rand_vals, *in_state_vals)
        self.__assign_vals_from_out_states(cache_key, rand_recover_vals, *out_state_vals)
        return out_


@set_module_as('brainstate.transform')
def vmap2(
    fn: F | Missing = Missing(),
    *,
    # --- normal jax.vmap arguments --- #
    in_axes: Optional[int | Sequence[Any]] = 0,
    out_axes: Any = 0,
    axis_name: Optional[AxisName] = None,
    axis_size: Optional[int] = None,
    spmd_axis_name: Optional[AxisName | Tuple[AxisName, ...]] = None,
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
        >>> @brainstate.transform.vmap2(
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
def pmap2(
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
        >>> @brainstate.transform.pmap2(
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
            pmap2,
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
    """Split a pytree into batches and remainder.

    Parameters
    ----------
    x : Any
        PyTree to split into batches.
    batch_size : int
        Size of each batch.

    Returns
    -------
    tuple
        A tuple of (batched_tree, remainder_tree). The batched_tree has shape
        (num_batches, batch_size, ...) and remainder_tree contains leftover
        elements, or None if there's no remainder.

    Raises
    ------
    ValueError
        If inputs have inconsistent lengths along the leading dimension.
    """
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


def _flatten(x):
    """Flatten the first two dimensions of an array.

    Parameters
    ----------
    x : array
        Array with at least 2 dimensions.

    Returns
    -------
    array
        Array with first two dimensions flattened into one.
    """
    return x.reshape(-1, *x.shape[2:])


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
    scan : Primitive used for the sequential fallback.
    """
    if batch_size is not None:
        scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
        g = lambda _, x: ((), vmap2(f)(*x))
        _, scan_ys = scan(g, (), scan_xs)
        if remainder_xs is None:
            ys = jax.tree.map(lambda x: _flatten(x), scan_ys)
        else:
            remainder_ys = vmap2(f)(*remainder_xs)
            ys = jax.tree.map(
                lambda x, y: jax.lax.concatenate([_flatten(x), y], dimension=0),
                scan_ys,
                remainder_ys,
            )
    else:
        g = lambda _, x: ((), f(*x))
        _, ys = scan(g, (), xs)
    return ys


@set_module_as('brainstate.transform')
def _map_new_states(
    map_fn: Callable,
    module: 'Module',
    init_kwargs: Dict,
    state_tag: str = None,
    axis_size: int = None,
    state_out_axes: Dict[int, Filter] = None,
):
    if state_out_axes is None:
        state_out_axes = dict()
    if not isinstance(state_out_axes, dict):
        state_out_axes = {0: state_out_axes}
    # convert filters to predicates
    state_out_axes = {k: filter.to_predicate(v) for k, v in state_out_axes.items()}
    # ensure NonBatchState goes to None axis
    if None not in state_out_axes:
        state_out_axes[None] = filter.to_predicate(NonBatchState)
    else:
        state_out_axes[None] = filter.Any(INIT_NO_BATCHING, state_out_axes[None])
    # ensure default axis 0
    if 0 not in state_out_axes:
        state_out_axes[0] = filter.to_predicate(...)

    # initialize a dictionary to store vmapped states
    dict_vmap_states = defaultdict(list)

    @map_fn(axis_size=axis_size, out_axes=tuple(state_out_axes.keys()))
    def fn_to_new_state_initialization():
        with catch_new_states() as catcher_:
            module.init_all_states(**init_kwargs)
        vmap_state_vals_ = defaultdict(list)
        for st_ in catcher_.get_states():
            for out_axis_, predicate_ in state_out_axes.items():
                if predicate_(tuple(), st_):
                    vmap_state_vals_[out_axis_].append(st_.value)
                    dict_vmap_states[out_axis_].append(st_)
                    break
            else:
                vmap_state_vals_[0].append(st_.value)
                dict_vmap_states[0].append(st_)
        outs = tuple(vmap_state_vals_.get(k, tuple()) for k in state_out_axes)
        return outs

    # restore vmapped state values
    with catch_new_states(state_tag):
        tuple_vmap_state_vals = fn_to_new_state_initialization()
    tuple_vmap_states = tuple(dict_vmap_states.get(k, tuple()) for k in state_out_axes)
    for st_vals, states in zip(tuple_vmap_state_vals, tuple_vmap_states):
        for val, st in zip(st_vals, states):
            st.restore_value(val)
            # ------------------------------------------------
            # --- this is CRUCIAL to avoid jax tracing leakage
            # ------------------------------------------------
            st.decrease_stack_level()  # 'vmap2_eval' StateStackTrace
            st.decrease_stack_level()  # 'vmap2' StateStackTrace

    return dict_vmap_states


@set_module_as('brainstate.transform')
def vmap2_new_states(
    module: 'Module',
    init_kwargs: Dict,
    state_tag: str = None,
    axis_size: int = None,
    state_out_axes: Dict[int, Filter] = None,
):
    """
    Initialize and vectorize newly created states within a module.

    This function creates vectorized versions of all states that are initialized
    when calling ``module.init_all_states(**init_kwargs)``. It uses :func:`vmap2`
    to create multiple copies of each state along specified axes, enabling
    efficient batched operations on modules with stateful components.

    The vectorization process wraps the module's initialization in a :func:`vmap2`
    transform, executes it in parallel across ``axis_size`` instances, and then
    restores the vectorized state values back to the original state objects. This
    allows subsequent operations on the module to work with batched states
    transparently.

    Parameters
    ----------
    module : Module
        Module whose states should be vectorized. Must have an ``init_all_states``
        method that creates the states to be vectorized.
    init_kwargs : dict
        Keyword arguments forwarded to ``module.init_all_states(**init_kwargs)``
        during the vectorized initialization. These arguments are passed to each
        parallel initialization call.
    state_tag : str, optional
        Tag for identifying and grouping the newly created states. Used by
        BrainState's state tracking system. Defaults to ``None``.
    axis_size : int, optional
        Size of the vectorization axis. Determines how many copies of each state
        will be created along the mapped axis. If ``None``, the size must be
        inferrable from the vectorized function's execution context.
    state_out_axes : Dict[int, Filter] or Filter, optional
        Specification for how to map output states along different axes. Can be:

        - A dictionary mapping axis indices (int) to :mod:`brainstate.util.filter`
          predicates that identify which states belong to which axis
        - A single filter (treated as ``{0: filter}`` for convenience)
        - ``None`` (default: all states assigned to axis 0, except
          :class:`~brainstate.NonBatchState` which goes to axis ``None``)

        Filters are converted to predicates via
        :func:`brainstate.util.filter.to_predicate`. States matching
        :class:`~brainstate.NonBatchState` are automatically assigned to axis
        ``None`` (unbatched) regardless of other specifications.

    Returns
    -------
    dict[int, list[State]]
        Dictionary mapping axis indices to lists of vectorized states. Keys are
        the axis indices specified in ``state_out_axes`` (plus ``None`` for
        non-batched states), and values are lists of :class:`~brainstate.State`
        objects with their ``.value`` attributes set to the vectorized arrays.

    Raises
    ------
    ValueError
        If state assignment is ambiguous or if ``axis_size`` cannot be inferred.

    Notes
    -----
    **Initialization Process:**

    1. Wraps ``module.init_all_states`` in a :func:`vmap2` transform
    2. Executes the initialization ``axis_size`` times in parallel
    3. Captures all newly created states using :func:`catch_new_states`
    4. Assigns states to axes based on ``state_out_axes`` predicates
    5. Restores vectorized values to the actual state objects
    6. Adjusts state stack levels to prevent JAX tracing leakage

    **State Axis Assignment:**

    States are assigned to axes in priority order:

    - First, :class:`~brainstate.NonBatchState` → axis ``None`` (unbatched)
    - Then, states matching custom filters in ``state_out_axes``
    - Finally, remaining states → axis 0 (default batch axis)

    **Critical Implementation Detail:**

    After restoring values, the function decreases the stack level twice on each
    state to prevent JAX tracing leakage. This is necessary because the
    :func:`vmap2` transform creates two nested state trace contexts
    (``'vmap2_eval'`` and ``'vmap2'``) that must be unwound.

    Examples
    --------
    **Basic vectorization with default axis:**

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class Counter(brainstate.nn.Module):
        ...     def init_state(self):
        ...         self.count = brainstate.ShortTermState(jnp.array(0))
        >>>
        >>> module = Counter()
        >>> vmap_states = brainstate.transform.vmap2_new_states(
        ...     module,
        ...     init_kwargs={},
        ...     axis_size=5
        ... )
        >>> module.count.value.shape
        (5,)

    **Custom axis assignment with filters:**

    .. code-block:: python

        >>> from brainstate.util.filter import OfType
        >>>
        >>> class MyModule(brainstate.nn.Module):
        ...     def init_state(self, size):
        ...         self.weight = brainstate.ParamState(jnp.zeros(size))
        ...         self.counter = brainstate.ShortTermState(0)
        >>>
        >>> module = MyModule()
        >>> vmap_states = brainstate.transform.vmap2_new_states(
        ...     module,
        ...     init_kwargs={'size': 10},
        ...     axis_size=5,
        ...     state_out_axes={
        ...         1: OfType(brainstate.ParamState),  # weights on axis 1
        ...         0: OfType(brainstate.ShortTermState),  # counter on axis 0
        ...     }
        ... )
        >>> module.weight.value.shape  # (size, axis_size)
        (10, 5)
        >>> module.counter.value.shape  # (axis_size,)
        (5,)

    **Non-batched states:**

    .. code-block:: python

        >>> class MixedModule(brainstate.nn.Module):
        ...     def init_state(self):
        ...         self.batched = brainstate.ShortTermState(0)
        ...         self.shared = brainstate.NonBatchState(jnp.array([1, 2, 3]))
        >>>
        >>> module = MixedModule()
        >>> vmap_states = brainstate.transform.vmap2_new_states(
        ...     module,
        ...     init_kwargs={},
        ...     axis_size=5
        ... )
        >>> module.batched.value.shape  # batched across 5 instances
        (5,)
        >>> module.shared.value.shape  # not batched
        (3,)

    See Also
    --------
    vmap2 : Vectorize a callable with state semantics.
    pmap2_new_states : Parallel version for multi-device initialization.
    brainstate.State : Base class for stateful objects.
    brainstate.NonBatchState : Marker for states that should not be batched.
    catch_new_states : Context manager for capturing newly created states.
    """
    return _map_new_states(
        vmap2,
        module,
        init_kwargs,
        state_tag=state_tag,
        axis_size=axis_size,
        state_out_axes=state_out_axes,
    )


@set_module_as('brainstate.transform')
def pmap2_new_states(
    module: 'Module',
    init_kwargs: Dict,
    state_tag: str = None,
    axis_size: int = None,
    state_out_axes: Dict[int, Filter] = None,
):
    """
    Initialize and parallelize newly created states across multiple devices.

    This function creates device-replicated or device-sharded versions of all
    states initialized by ``module.init_all_states(**init_kwargs)``. It uses
    :func:`pmap2` to distribute state initialization across multiple devices,
    enabling efficient multi-device parallelism for modules with stateful
    components.

    The parallelization process wraps the module's initialization in a
    :func:`pmap2` transform, executes it in parallel across ``axis_size`` devices,
    and then restores the device-distributed state values back to the original
    state objects. This allows subsequent operations on the module to work with
    device-parallelized states transparently.

    Parameters
    ----------
    module : Module
        Module whose states should be parallelized across devices. Must have an
        ``init_all_states`` method that creates the states to be distributed.
    init_kwargs : dict
        Keyword arguments forwarded to ``module.init_all_states(**init_kwargs)``
        during the parallel initialization. These arguments are passed to each
        device's initialization call.
    state_tag : str, optional
        Tag for identifying and grouping the newly created states. Used by
        BrainState's state tracking system. Defaults to ``None``.
    axis_size : int, optional
        Size of the parallel axis, typically the number of devices to map over.
        If ``None``, defaults to the number of available devices (e.g.,
        ``jax.local_device_count()``).
    state_out_axes : Dict[int, Filter] or Filter, optional
        Specification for how to distribute output states across devices and axes.
        Can be:

        - A dictionary mapping axis indices (int) to :mod:`brainstate.util.filter`
          predicates that identify which states are distributed along which axis
        - A single filter (treated as ``{0: filter}`` for convenience)
        - ``None`` (default: all states distributed along axis 0, except
          :class:`~brainstate.NonBatchState` which is replicated)

        Filters are converted to predicates via
        :func:`brainstate.util.filter.to_predicate`. States matching
        :class:`~brainstate.NonBatchState` are automatically replicated
        (axis ``None``) across all devices regardless of other specifications.

    Returns
    -------
    dict[int, list[State]]
        Dictionary mapping axis indices to lists of parallelized states. Keys are
        the axis indices specified in ``state_out_axes`` (plus ``None`` for
        replicated states), and values are lists of :class:`~brainstate.State`
        objects with their ``.value`` attributes set to device-distributed arrays.

    Raises
    ------
    ValueError
        If state assignment is ambiguous, if ``axis_size`` exceeds available
        devices, or if device configuration is invalid.

    Notes
    -----
    **Initialization Process:**

    1. Wraps ``module.init_all_states`` in a :func:`pmap2` transform
    2. Executes the initialization on ``axis_size`` devices in parallel
    3. Captures all newly created states using :func:`catch_new_states`
    4. Assigns states to axes based on ``state_out_axes`` predicates
    5. Restores device-distributed values to the actual state objects
    6. Adjusts state stack levels to prevent JAX tracing leakage

    **State Device Distribution:**

    States are assigned to axes (and thus distributed across devices) in priority
    order:

    - First, :class:`~brainstate.NonBatchState` → axis ``None`` (replicated)
    - Then, states matching custom filters in ``state_out_axes``
    - Finally, remaining states → axis 0 (default device-parallel axis)

    **Device Semantics:**

    - Axis 0 (default): States are sharded across devices along the first dimension
    - Axis ``None``: States are replicated identically on all devices
    - Custom axes: States can be sharded along different dimensions based on filters

    **Multi-Host Considerations:**

    In multi-host setups, ``axis_size`` typically corresponds to the local device
    count. The devices must be specified consistently across all hosts when using
    explicit device lists with :func:`pmap2`.

    Examples
    --------
    **Basic parallel initialization:**

    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> class ParallelCounter(brainstate.nn.Module):
        ...     def init_state(self):
        ...         self.count = brainstate.ShortTermState(jnp.array(0))
        >>>
        >>> module = ParallelCounter()
        >>> pmap_states = brainstate.transform.pmap2_new_states(
        ...     module,
        ...     init_kwargs={},
        ...     axis_size=jax.local_device_count()
        ... )
        >>> module.count.value.shape
        (jax.local_device_count(),)

    **Parallel model with device-sharded parameters:**

    .. code-block:: python

        >>> from brainstate.util.filter import OfType
        >>>
        >>> class ParallelModel(brainstate.nn.Module):
        ...     def init_state(self, layer_size):
        ...         self.weight = brainstate.ParamState(
        ...             jax.random.normal(jax.random.PRNGKey(0), (layer_size,))
        ...         )
        ...         self.bias = brainstate.ParamState(jnp.zeros(layer_size))
        >>>
        >>> model = ParallelModel()
        >>> n_devices = jax.local_device_count()
        >>> pmap_states = brainstate.transform.pmap2_new_states(
        ...     model,
        ...     init_kwargs={'layer_size': 128},
        ...     axis_size=n_devices,
        ...     state_out_axes={0: OfType(brainstate.ParamState)}
        ... )
        >>> # Parameters are sharded across devices
        >>> model.weight.value.shape
        (n_devices, 128)

    **Mixed replicated and sharded states:**

    .. code-block:: python

        >>> class MixedParallelModule(brainstate.nn.Module):
        ...     def init_state(self):
        ...         # Sharded state (different on each device)
        ...         self.local_state = brainstate.ShortTermState(jnp.array(0))
        ...         # Replicated state (same on all devices)
        ...         self.global_config = brainstate.NonBatchState(
        ...             jnp.array([1.0, 2.0, 3.0])
        ...         )
        >>>
        >>> module = MixedParallelModule()
        >>> pmap_states = brainstate.transform.pmap2_new_states(
        ...     module,
        ...     init_kwargs={},
        ...     axis_size=jax.local_device_count()
        ... )
        >>> module.local_state.value.shape  # sharded
        (jax.local_device_count(),)
        >>> module.global_config.value.shape  # replicated (not sharded)
        (3,)

    **Using with ModuleMapper for data parallelism:**

    .. code-block:: python

        >>> from brainstate.nn import ModuleMapper
        >>>
        >>> model = ParallelModel()
        >>> pmapper = ModuleMapper(
        ...     model,
        ...     init_map_size=jax.local_device_count(),
        ...     behavior='pmap',
        ...     axis_name='devices'
        ... )
        >>> pmapper.init_all_states(layer_size=128)
        >>> # Data-parallel training across devices
        >>> batch_per_device = inputs.shape[0] // jax.local_device_count()
        >>> sharded_inputs = inputs.reshape(
        ...     jax.local_device_count(), batch_per_device, -1
        ... )
        >>> outputs = pmapper.update(sharded_inputs)

    See Also
    --------
    pmap2 : Parallel mapping across devices with state semantics.
    vmap2_new_states : Vectorized version for single-device batching.
    brainstate.State : Base class for stateful objects.
    brainstate.NonBatchState : Marker for states that should be replicated.
    jax.pmap : Underlying JAX parallel mapping primitive.
    catch_new_states : Context manager for capturing newly created states.
    """
    return _map_new_states(
        pmap2,
        module,
        init_kwargs,
        state_tag=state_tag,
        axis_size=axis_size,
        state_out_axes=state_out_axes,
    )
