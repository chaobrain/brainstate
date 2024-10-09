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

import jax

from brainstate.compile._loop_collect_return import scan

__all__ = [
  'map',
]


def vmap():
  pass


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


def map(
    f,
    xs,
    *,
    batch_size: int | None = None,
):
  """Map a function over leading array axes.

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
    >>> y = map(f, x, batch_size=3)
    inner shape: (3, 4)
    inner shape: (3, 4)
    >>> y.shape
    (10, 3, 4)

  In the example above, "inner shape" is printed twice, once while tracing the batched
  computation and once while tracing the remainder computation.

  Args:
    f: a Python function to apply element-wise over the first axis or axes of
      ``xs``.
    xs: values over which to map along the leading axis.
    batch_size: (optional) integer specifying the size of the batch for each step to execute
      in parallel.

  Returns:
    Mapped values.
  """
  if batch_size is not None:
    scan_xs, remainder_xs = _batch_and_remainder(xs, batch_size)
    g = lambda _, x: ((), jax.vmap(f)(x))
    _, scan_ys = scan(g, (), scan_xs)
    remainder_ys = jax.vmap(f)(remainder_xs)
    flatten = lambda x: x.reshape(-1, *x.shape[2:])
    ys = jax.tree.map(
      lambda x, y: jax.numpy.concatenate([flatten(x), y], axis=0), scan_ys, remainder_ys,
    )
  else:
    g = lambda _, x: ((), f(x))
    _, ys = scan(g, (), xs)
  return ys
