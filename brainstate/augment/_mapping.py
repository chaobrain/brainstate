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

import dataclasses
import functools
from typing import Any, TypeVar, Callable, Hashable, Sequence, Iterable, Mapping, Tuple, Union, Optional

import jax

from brainstate.compile._loop_collect_return import for_loop
from brainstate.graph._graph_convert import (NodeStates, graph_to_tree, tree_to_graph)
from brainstate.typing import Missing, Filter
from brainstate.util import NestedMapping

__all__ = [
  'StateAxes',
  'vmap',
  'pmap',
  'mini_vmap',
  'mini_pmap',
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
      filter_axes: Union[Mapping[Filter, Index | Carry | None], Iterable[Tuple[Filter, Index | Carry | None]]],
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
    return NodeStates.from_split(*ctx.split(x, *prefix.filters), metadata=prefix)
  return NodeStates.from_split(*ctx.split(x), metadata=prefix)


@dataclasses.dataclass(eq=False)
class MapFn:
  f: Callable[..., Any]
  in_axes: Any
  out_axes: Any

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args: Tuple[Any, ...]):
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
    # specific to 'brainstate'
    rng_splits: int = 0,
    rng_restore: bool = True,
    # transform kwargs
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
  mapped_fn = transform(MapFn(f, in_axes, out_axes), in_axes=jax_in_axes, out_axes=jax_out_axes, **transform_kwargs)

  @functools.wraps(f)
  def map_wrapper(*args):
    # graph to pytree
    pure_args, rng_backup = graph_to_tree(args, prefix=in_axes, split_fn=_map_split_fn, rng_splits=rng_splits)

    # vmap with pytree
    pure_out = mapped_fn(*pure_args)

    # pytree to graph
    return tree_to_graph(pure_out, rng_backup=rng_backup if (rng_restore and rng_splits > 0) else dict())

  return map_wrapper  # type: ignore


def vmap(
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
  """Reference-aware version of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__.

  Args:
    f: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__). In
      addition to integers and None, :class:`StateAxes`  can be used to control
      how graph nodes like Modules are vectorized by specifying the axes to be
      applied to substates of the graph node given a `Filter
      <https://flax.readthedocs.io/en/latest/nnx/filters_guide.html>`__.
    out_axes: An integer, None, or pytree indicating where the mapped axis
      should appear in the output (see `jax.vmap
      <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__).
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``f`` with arguments that correspond to
    those of ``f``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``f``, but
    with extra array axes at positions indicated by ``out_axes``.

  Example::

    >>> from flax import nnx
    >>> from jax import random, numpy as jnp
    ...
    >>> model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((5, 2))
    ...
    >>> @nnx.vmap(in_axes=(None, 0), out_axes=0)
    ... def forward(model, x):
    ...   return model(x)
    ...
    >>> y = forward(model, x)
    >>> y.shape
    (5, 3)

  >>> class LinearEnsemble(nnx.Module):
  ...   def __init__(self, num, rngs):
  ...     self.w = nnx.Param(jax.random.uniform(rngs(), (num, 2, 3)))
  ...
  >>> model = LinearEnsemble(5, rngs=nnx.Rngs(0))
  >>> x = jnp.ones((2,))
  ...
  >>> @nnx.vmap(in_axes=(0, None), out_axes=0)
  ... def forward(model, x):
  ...   return jnp.dot(x, model.w.value)
  ...
  >>> y = forward(model, x)
  >>> y.shape
  (5, 3)

  To control control how graph node substates are vectorized, ``StateAxes``
  can be passed to ``in_axes`` and ``out_axes`` specifying the axes to be
  applied to each substate given a filter. The following example shows how to
  share the parameters between the ensemble members which keeping different
  batch statistics and dropout random state::

    >>> class Foo(nnx.Module):
    ...   def __init__(self):
    ...     self.a = nnx.Param(jnp.arange(4))
    ...     self.b = nnx.BatchStat(jnp.arange(4))
    ...
    >>> state_axes = nnx.StateAxes({nnx.Param: 0, nnx.BatchStat: None})
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=0)
    ... def mul(foo):
    ...   return foo.a * foo.b
    ...
    >>> foo = Foo()
    >>> y = mul(foo)
    >>> y
    Array([[0, 0, 0, 0],
           [0, 1, 2, 3],
           [0, 2, 4, 6],
           [0, 3, 6, 9]], dtype=int32)
  """
  if isinstance(fn, Missing):
    return functools.partial(
      vmap,
      in_axes=in_axes,
      out_axes=out_axes,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      rng_splits=rng_splits,
      rng_restore=rng_restore,
    )  # type: ignore[return-value]

  return _map_transform(jax.vmap, fn,
                        in_axes=in_axes,
                        out_axes=out_axes,
                        axis_name=axis_name,
                        axis_size=axis_size,
                        spmd_axis_name=spmd_axis_name,
                        rng_splits=rng_splits,
                        rng_restore=rng_restore, )


def pmap(
    fn: Callable[[NestedMapping, ...], Any] | Missing = Missing(),
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
  """
  Parallel map with support for collective operations.

  The purpose of :py:func:`pmap` is to express single-program multiple-data
  (SPMD) programs. Applying :py:func:`pmap` to a function will compile the
  function with XLA (similarly to :py:func:`jit`), then execute it in parallel
  on XLA devices, such as multiple GPUs or multiple TPU cores. Semantically it
  is comparable to :py:func:`vmap` because both transformations map a function
  over array axes, but where :py:func:`vmap` vectorizes functions by pushing the
  mapped axis down into primitive operations, :py:func:`pmap` instead replicates
  the function and executes each replica on its own XLA device in parallel.

  The mapped axis size must be less than or equal to the number of local XLA
  devices available, as returned by :py:func:`jax.local_device_count()` (unless
  ``devices`` is specified, see below). For nested :py:func:`pmap` calls, the
  product of the mapped axis sizes must be less than or equal to the number of
  XLA devices.

  .. note::
    :py:func:`pmap` compiles ``fun``, so while it can be combined with
    :py:func:`jit`, it's usually unnecessary.

  :py:func:`pmap` requires that all of the participating devices are identical.
  For example, it is not possible to use :py:func:`pmap` to parallelize a
  computation across two different models of GPU. It is currently an error for
  the same device to participate twice in the same `pmap`.

  **Multi-process platforms:** On multi-process platforms such as TPU pods,
  :py:func:`pmap` is designed to be used in SPMD Python programs, where every
  process is running the same Python code such that all processes run the same
  pmapped function in the same order. Each process should still call the pmapped
  function with mapped axis size equal to the number of *local* devices (unless
  ``devices`` is specified, see below), and an array of the same leading axis
  size will be returned as usual. However, any collective operations in ``fun``
  will be computed over *all* participating devices, including those on other
  processes, via device-to-device communication.  Conceptually, this can be
  thought of as running a pmap over a single array sharded across processes,
  where each process "sees" only its local shard of the input and output. The
  SPMD model requires that the same multi-process pmaps must be run in the same
  order on all devices, but they can be interspersed with arbitrary operations
  running in a single process.

  Args:
    fun: Function to be mapped over argument axes. Its arguments and return
      value should be arrays, scalars, or (nested) standard Python containers
      (tuple/list/dict) thereof. Positional arguments indicated by
      ``static_broadcasted_argnums`` can be anything at all, provided they are
      hashable and have an equality operation defined.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    in_axes: A non-negative integer, None, or nested Python container thereof
      that specifies which axes of positional arguments to map over. Arguments
      passed as keywords are always mapped over their leading axis (i.e. axis
      index 0). See :py:func:`vmap` for details.
    out_axes: A non-negative integer, None, or nested Python container thereof
      indicating where the mapped axis should appear in the output. All outputs
      with a mapped axis must have a non-None ``out_axes`` specification
      (see :py:func:`vmap`).
    static_broadcasted_argnums: An int or collection of ints specifying which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded.
      Calling the pmapped function with different values for these constants
      will trigger recompilation. If the pmapped function is called with fewer
      positional arguments than indicated by ``static_broadcasted_argnums`` then
      an error is raised. Each of the static arguments will be broadcasted to
      all devices. Arguments that are not arrays or containers thereof must be
      marked as static. Defaults to ().

      Static arguments must be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and should be immutable.

    devices: This is an experimental feature and the API is likely to change.
      Optional, a sequence of Devices to map over. (Available devices can be
      retrieved via jax.devices()). Must be given identically for each process
      in multi-process settings (and will therefore include devices across
      processes). If specified, the size of the mapped axis must be equal to
      the number of devices in the sequence local to the given process. Nested
      :py:func:`pmap` s with ``devices`` specified in either the inner or outer
      :py:func:`pmap` are not yet supported.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend. 'cpu', 'gpu', or 'tpu'.
    axis_size: Optional; the size of the mapped axis.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer need
      them once the computation has finished. In some cases XLA can make use of
      donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to.
      Note that donate_argnums only work for positional arguments, and keyword
      arguments will not be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    global_arg_shapes: Optional; a tuple of tuples of integers representing the
      shapes of the global arguments. These are arguments that are not replicated
      across devices, but are broadcasted to all devices. The tuple should have
      the same length as the number of global arguments, and each inner tuple
      should have the same length as the corresponding argument. The shapes of
      the global arguments must be the same on all devices.

  Returns:
    A parallelized version of ``fun`` with arguments that correspond to those of
    ``fun`` but with extra array axes at positions indicated by ``in_axes`` and
    with output that has an additional leading array axis (with the same size).

  For example, assuming 8 XLA devices are available, :py:func:`pmap` can be used
  as a map along a leading array axis:

  >>> import jax.numpy as jnp
  >>>
  >>> out = pmap(lambda x: x ** 2)(jnp.arange(8))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [0, 1, 4, 9, 16, 25, 36, 49]

  When the leading dimension is smaller than the number of available devices JAX
  will simply run on a subset of devices:

  >>> x = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2))
  >>> y = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2
  >>> out = pmap(jnp.dot)(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [[[    4.     9.]
    [   12.    29.]]
   [[  244.   345.]
    [  348.   493.]]
   [[ 1412.  1737.]
    [ 1740.  2141.]]]

  If your leading dimension is larger than the number of available devices you
  will get an error:

  >>> pmap(lambda x: x ** 2)(jnp.arange(9))  # doctest: +SKIP
  ValueError: ... requires 9 replicas, but only 8 XLA devices are available

  As with :py:func:`vmap`, using ``None`` in ``in_axes`` indicates that an
  argument doesn't have an extra axis and should be broadcasted, rather than
  mapped, across the replicas:

  >>> x, y = jnp.arange(2.), 4.
  >>> out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  ([4., 5.], [8., 8.])

  Note that :py:func:`pmap` always returns values mapped over their leading axis,
  equivalent to using ``out_axes=0`` in :py:func:`vmap`.

  In addition to expressing pure maps, :py:func:`pmap` can also be used to express
  parallel single-program multiple-data (SPMD) programs that communicate via
  collective operations. For example:

  >>> f = lambda x: x / jax.lax.psum(x, axis_name='i')
  >>> out = pmap(f, axis_name='i')(jnp.arange(4.))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [ 0.          0.16666667  0.33333334  0.5       ]
  >>> print(out.sum())  # doctest: +SKIP
  1.0

  In this example, ``axis_name`` is a string, but it can be any Python object
  with ``__hash__`` and ``__eq__`` defined.

  The argument ``axis_name`` to :py:func:`pmap` names the mapped axis so that
  collective operations, like :func:`jax.lax.psum`, can refer to it. Axis names
  are important particularly in the case of nested :py:func:`pmap` functions,
  where collective operations can operate over distinct axes:

  >>> from functools import partial
  >>> import jax
  >>>
  >>> @partial(pmap, axis_name='rows')
  ... @partial(pmap, axis_name='cols')
  ... def normalize(x):
  ...   row_normed = x / jax.lax.psum(x, 'rows')
  ...   col_normed = x / jax.lax.psum(x, 'cols')
  ...   doubly_normed = x / jax.lax.psum(x, ('rows', 'cols'))
  ...   return row_normed, col_normed, doubly_normed
  >>>
  >>> x = jnp.arange(8.).reshape((4, 2))
  >>> row_normed, col_normed, doubly_normed = normalize(x)  # doctest: +SKIP
  >>> print(row_normed.sum(0))  # doctest: +SKIP
  [ 1.  1.]
  >>> print(col_normed.sum(1))  # doctest: +SKIP
  [ 1.  1.  1.  1.]
  >>> print(doubly_normed.sum((0, 1)))  # doctest: +SKIP
  1.0

  On multi-process platforms, collective operations operate over all devices,
  including those on other processes. For example, assuming the following code
  runs on two processes with 4 XLA devices each:

  >>> f = lambda x: x + jax.lax.psum(x, axis_name='i')
  >>> data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
  >>> out = pmap(f, axis_name='i')(data)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [28 29 30 31] # on process 0
  [32 33 34 35] # on process 1

  Each process passes in a different length-4 array, corresponding to its 4
  local devices, and the psum operates over all 8 values. Conceptually, the two
  length-4 arrays can be thought of as a sharded length-8 array (in this example
  equivalent to jnp.arange(8)) that is mapped over, with the length-8 mapped
  axis given name 'i'. The pmap call on each process then returns the
  corresponding length-4 output shard.

  The ``devices`` argument can be used to specify exactly which devices are used
  to run the parallel computation. For example, again assuming a single process
  with 8 devices, the following code defines two parallel computations, one
  which runs on the first six devices and one on the remaining two:

  >>> from functools import partial
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[:6])
  ... def f1(x):
  ...   return x / jax.lax.psum(x, axis_name='i')
  >>>
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[-2:])
  ... def f2(x):
  ...   return jax.lax.psum(x ** 2, axis_name='i')
  >>>
  >>> print(f1(jnp.arange(6.)))  # doctest: +SKIP
  [0.         0.06666667 0.13333333 0.2        0.26666667 0.33333333]
  >>> print(f2(jnp.array([2., 3.])))  # doctest: +SKIP
  [ 13.  13.]
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


def _flatten(x):
  return x.reshape(-1, *x.shape[2:])


def _batch_and_remainder(x, batch_size: int):
  leaves, treedef = jax.tree.flatten(x)

  scan_leaves = []
  remainder_leaves = []
  for leaf in leaves:
    num_batches, _ = divmod(leaf.shape[0], batch_size)
    total_batch_elems = num_batches * batch_size
    scan_leaves.append(leaf[:total_batch_elems].reshape(num_batches, batch_size, *leaf.shape[1:]))
    remainder_leaves.append(leaf[total_batch_elems:])

  scan_tree = treedef.unflatten(scan_leaves)
  remainder_tree = treedef.unflatten(remainder_leaves)
  return scan_tree, remainder_tree


def mini_vmap(
    f: Callable[[NestedMapping, ...], Any],
    *xs: X,
    batch_size: int,
) -> Y:
  """
  A memory-efficient version of :func:`vmap`, map a function over leading array axes.

  Like Python's builtin map, except inputs and outputs are in the form of
  stacked arrays. Consider using the :func:`~jax.vmap` compile instead, unless you
  need to apply a function element by element for reduced memory usage or
  heterogeneous computation with other control flow primitives.

  When ``xs`` is an array type, the semantics of :func:`~map` are given by this
  Python implementation::

    def map(f, xs):
      return np.stack([f(x) for x in xs])

  Like :func:`~scan`, :func:`~map` is implemented in terms of JAX primitives so
  many of the same advantages over a Python loop apply: ``xs`` may be an
  arbitrary nested pytree type, and the mapped computation is compiled only
  once.

  If ``batch_size`` is provided, the computation is executed in batches of that size
  and parallelized using :func:`~jax.vmap`. This can be used as either a more performant
  version of ``map`` or as a memory-efficient version of ``vmap``. If the axis is not
  divisible by the batch size, the remainder is processed in a separate ``vmap`` and
  concatenated to the result.

    >>> x = jax.numpy.ones((10, 3, 4))
    >>> def f(x):
    ...   print('inner shape:', x.shape)
    ...   return x + 1
    >>> y = mini_vmap(f, x, batch_size=3)
    inner shape: (3, 4)
    inner shape: (3, 4)
    >>> y.shape
    (10, 3, 4)

  In the example above, "inner shape" is printed twice, once while tracing the batched
  computation and once while tracing the remainder computation.

  Args:
    f: a Python function to apply element-wise over the first axis or axes of ``xs``.
    xs: values over which to map along the leading axis.
    batch_size: (optional) integer specifying the size of the batch for each step to execute in parallel.

  Returns:
    Mapped values.
  """
  scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
  scan_ys = for_loop(vmap(f), scan_xs)
  remainder_ys = vmap(f)(remainder_xs)
  ys = jax.tree.map(lambda x, y: jax.numpy.concatenate([_flatten(x), y], axis=0), scan_ys, remainder_ys)
  return ys


def mini_pmap(f, *xs, batch_size: int, **kwargs):
  """
  A memory-efficient version of :func:`pmap`, map a function over leading array axes on multiple devices.
  """
  scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
  scan_ys = for_loop(pmap(f, **kwargs), scan_xs)
  remainder_ys = pmap(f, **kwargs)(remainder_xs)
  ys = jax.tree.map(lambda x, y: jax.numpy.concatenate([_flatten(x), y], axis=0), scan_ys, remainder_ys)
  return ys
