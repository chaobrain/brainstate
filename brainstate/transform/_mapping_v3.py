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

import dataclasses
import functools
from typing import Any, TypeVar, Callable, Hashable, Sequence, Iterable, Mapping, Tuple, Union, Optional

import jax
from brainstate._compatible_import import BatchTracer

from brainstate.graph._convert import NodeStates, graph_to_tree, tree_to_graph
from brainstate.typing import Missing, Filter
from brainstate.util import NestedDict
from ._make_jaxpr import StatefulFunction

__all__ = [
    'StateAxes',
    'model_vmap',
    'model_pmap',
]

AxisName = Hashable
F = TypeVar("F", bound=Callable)
X = TypeVar("X")
Y = TypeVar("Y")
Index = int
Carry = TypeVar("Carry")


class StateAxes:
    """
    A class to represent the axes of a state.
    """

    def __init__(
        self,
        filter_axes: Union[
            Mapping[Filter, Index | Carry | None],
            Iterable[Tuple[Filter, Index | Carry | None]]
        ],
    ):
        iterable = filter_axes.items() if isinstance(filter_axes, Mapping) else filter_axes
        self._filters = tuple(filter_ for filter_, _ in iterable)
        self._axes = tuple(axis for _, axis in iterable)

    @property
    def filters(self) -> Tuple[Filter, ...]:
        return self._filters

    @property
    def axes(self) -> Tuple[Index | Carry | None, ...]:
        return self._axes

    def __repr__(self):
        return f'StateAxes({dict(self.items())})'

    def items(self):
        return zip(self.filters, self.axes)

    def __eq__(self, other):
        return isinstance(other, StateAxes) and self.filters == other.filters and self.axes == other.axes

    def __hash__(self):
        return hash((self.filters, self.axes))


def _map_split_fn(ctx, path, prefix, x):
    if isinstance(prefix, StateAxes):
        return NodeStates.from_split(*ctx.treefy_split(x, *prefix.filters), metadata=prefix)
    return NodeStates.from_split(*ctx.treefy_split(x), metadata=prefix)


@dataclasses.dataclass(eq=False)
class MapFn:
    f: Callable[..., Any]
    in_axes: Any
    out_axes: Any
    mapping_size: int = None

    def __post_init__(self):
        functools.update_wrapper(self, self.f)

    def _find_batch_tracer(self, x):
        if isinstance(x, BatchTracer) and self.mapping_size is None:
            self.mapping_size = x.val.shape[x.batch_dim]

    def __call__(self, *pure_args: Tuple[Any, ...]):
        jax.tree.map(self._find_batch_tracer, pure_args)

        # pytree to graph
        args = tree_to_graph(pure_args)
        # call the function
        out = self.f(*args)
        # graph to pytree
        pure_out, _ = graph_to_tree(out, prefix=self.out_axes, split_fn=_map_split_fn)
        return pure_out


def _map_transform(
    transform,
    f: F,
    *,
    in_axes: Optional[int | Sequence[Any]] = 0,
    out_axes: Any = 0,
    **transform_kwargs,
):
    # jax in axes
    jax_in_axes = jax.tree.map(
        lambda x: NodeStates.from_prefixes(x.axes, metadata=x) if isinstance(x, StateAxes) else x,
        in_axes,
    )

    # jax out axes
    jax_out_axes = jax.tree.map(
        lambda x: NodeStates.from_prefixes(x.axes, metadata=x) if isinstance(x, StateAxes) else x,
        out_axes,
    )

    # mapped function
    map_fn = MapFn(f, in_axes, out_axes)
    mapped_fn = transform(
        map_fn,
        in_axes=jax_in_axes,
        out_axes=jax_out_axes,
        **transform_kwargs
    )

    @functools.wraps(f)
    def fn2call(*args):
        # graph to pytree
        pure_args, rng_backup = graph_to_tree(args, prefix=in_axes, split_fn=_map_split_fn)

        # vmap with pytree
        pure_out = mapped_fn(*pure_args)

        # pytree to graph
        return tree_to_graph(pure_out)

    staful_fn = StatefulFunction(fn2call)

    @functools.wraps(f)
    def map_wrapper(*args):
        cache_key = staful_fn.get_arg_cache_key(*args, compile_if_miss=True)

    return fn2call  # type: ignore


def model_vmap(
    fn: F | Missing = Missing(),
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # specific to 'brainstate'
    rng_splits: int = 0,
    rng_restore: bool = True,
) -> F | Callable[[F], F]:
    if isinstance(fn, Missing):
        return functools.partial(
            model_vmap,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
        )  # type: ignore[return-value]

    return _map_transform(
        jax.vmap,
        fn,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
    )


def model_pmap(
    fn: Callable[[NestedDict, ...], Any] | Missing = Missing(),
    axis_name: Optional[AxisName] = None,
    *,
    in_axes: Any = 0,
    out_axes: Any = 0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Optional[Sequence[jax.Device]] = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: int | Iterable[int] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    rng_splits: int = 0,
    rng_restore: bool = True,
) -> Callable[[F], F] | F:
    if isinstance(fn, Missing):
        return functools.partial(
            model_pmap,
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices,
            backend=backend,
            axis_size=axis_size,
            donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
            rng_splits=rng_splits,
            rng_restore=rng_restore,
        )  # type: ignore[return-value]

    return _map_transform(jax.pmap, fn,
                          in_axes=in_axes,
                          out_axes=out_axes,
                          axis_name=axis_name,
                          static_broadcasted_argnums=static_broadcasted_argnums,
                          devices=devices,
                          backend=backend,
                          axis_size=axis_size,
                          donate_argnums=donate_argnums,
                          global_arg_shapes=global_arg_shapes,
                          rng_splits=rng_splits,
                          rng_restore=rng_restore, )
