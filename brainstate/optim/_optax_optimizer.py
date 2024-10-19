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

import importlib.util
from typing import Any

import jax.numpy as jnp

from brainstate._state import ShortTermState, ParamState
from brainstate.graph import Node

optax_installed = importlib.util.find_spec('optax') is not None

__all__ = [
  'OptaxOptimizer',
]


class OptaxState(ShortTermState):
  """Wrapper class for Optimizer Variables."""
  pass


class OptaxOptimizer(Node):
  """Simple train state for the common case with a single Optax optimizer.

  Example usage::

    >>> import jax, jax.numpy as jnp
    >>> import brainstate as bst
    >>> from brainstate import nn
    >>> import optax
    ...
    >>> class Model(bst.nn.Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.linear1 = bst.nn.Linear(2, 3)
    ...     self.linear2 = bst.nn.Linear(3, 4)
    ...   def __call__(self, x):
    ...     return self.linear2(self.linear1(x))
    ...
    >>> x = jax.random.normal(jax.random.key(0), (1, 2))
    >>> y = jnp.ones((1, 4))
    ...
    >>> model = Model()
    >>> tx = optax.adam(1e-3)
    >>> state = bst.optim.OptaxOptimizer(model, tx)
    ...
    >>> loss_fn = lambda model: ((model(x) - y) ** 2).mean()
    >>> loss_fn(model)
    Array(1.7055722, dtype=float32)
    >>> grads = bst.transform.grad(loss_fn)(state.model)
    >>> state.update(grads)
    >>> loss_fn(model)
    Array(1.6925814, dtype=float32)

  Note that you can easily extend this class by subclassing it for storing
  additional data (e.g. adding metrics).

  Example usage::

    >>> class TrainState(nnx.Optimizer):
    ...   def __init__(self, model, tx, metrics):
    ...     self.metrics = metrics
    ...     super().__init__(model, tx)
    ...   def update(self, *, grads, **updates):
    ...     self.metrics.update(**updates)
    ...     super().update(grads)
    ...
    >>> metrics = nnx.metrics.Average()
    >>> state = TrainState(model, tx, metrics)
    ...
    >>> grads = nnx.grad(loss_fn)(state.model)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    Array(1.6925814, dtype=float32)
    >>> state.update(grads=grads, values=loss_fn(state.model))
    >>> state.metrics.compute()
    Array(1.68612, dtype=float32)

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Attributes:
    step: An ``OptaxState`` :class:`State` that tracks the step count.
    model: The wrapped :class:`Module`.
    tx: An Optax gradient transformation.
    opt_state: The Optax optimizer state.
  """

  def __init__(
      self,
      model: Node,
      tx: 'optax.GradientTransformation',
      wrt: Any = ParamState,
  ):
    """
    Instantiate the class and wrap the :class:`Module` and Optax gradient
    transformation. Instantiate the optimizer state to keep track of
    :class:`State` types specified in ``wrt``. Set the step count to 0.

    Args:
      model: An NNX Module.
      tx: An Optax gradient transformation.
      wrt: optional argument to filter for which :class:`State`'s to keep
        track of in the optimizer state. These should be the :class:`State`'s
        that you plan on updating; i.e. this argument value should match the
        ``wrt``  argument passed to the ``nnx.grad`` call that will generate the
        gradients that will be passed into the ``grads`` argument of the
        :func:`update` method.
    """

    # tx must be an instance of optax.GradientTransformation
    import optax  # type: ignore[import-not-found,import-untyped]
    if not isinstance(tx, optax.GradientTransformation):
      raise TypeError(f"tx must be an instance of optax.GradientTransformation, got {tx}")
    self.tx = tx

    # model
    if not callable(model):
      raise TypeError(f"model must be a callable, got {model}")
    self.model = model

    # wrt
    self.opt_state = tx.init(nnx.states_as_trees(model, wrt))
    self.wrt = wrt

  def update(self, grads):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.
    The ``grads`` must be derived from ``nnx.grad(..., wrt=self.wrt)``, where the
    gradients are with respect to the same :class:`State` types as defined in
    ``self.wrt`` during instantiation of this ``Optimizer``. For example::

      >>> from flax import nnx
      >>> import jax, jax.numpy as jnp
      >>> import optax

      >>> class CustomVariable(nnx.State):
      ...   pass

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
      ...     self.custom_variable = CustomVariable(jnp.ones((1, 3)))
      ...   def __call__(self, x):
      ...     return self.linear(x) + self.custom_variable
      >>> model = Model(rngs=nnx.Rngs(0))
      >>> jax.tree.map(jnp.shape, nnx.states_as_trees(model))
      State({
        'custom_variable': StateAsPyTree(
          type=CustomVariable,
          value=(1, 3)
        ),
        'linear': {
          'bias': StateAsPyTree(
            type=Param,
            value=(3,)
          ),
          'kernel': StateAsPyTree(
            type=Param,
            value=(2, 3)
          )
        }
      })

      >>> # update:
      >>> # - only Linear layer parameters
      >>> # - only CustomVariable parameters
      >>> # - both Linear layer and CustomVariable parameters
      >>> loss_fn = lambda model, x, y: ((model(x) - y) ** 2).mean()
      >>> for variable in (nnx.Param, CustomVariable, (nnx.Param, CustomVariable)):
      ...   # make sure `wrt` arguments match for `nnx.Optimizer` and `nnx.grad`
      ...   state = nnx.Optimizer(model, optax.adam(1e-3), wrt=variable)
      ...   grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, variable))(
      ...     state.model, jnp.ones((1, 2)), jnp.ones((1, 3))
      ...   )
      ...   state.update(grads=grads)

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: the gradients derived from ``nnx.grad``.
    """
    import optax  # type: ignore[import-not-found,import-untyped]
    state = nnx.states_as_trees(self.model, self.wrt)

    updates, new_opt_state = self.tx.update(grads, self.opt_state, state)
    new_params = optax.apply_updates(state, updates)
    assert isinstance(new_params, nnx.NestedMapping)

    nnx.update(self.model, new_params)
    self.opt_state = new_opt_state
