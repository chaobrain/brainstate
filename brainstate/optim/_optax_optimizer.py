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
from typing import Hashable, Dict

from brainstate._state import ShortTermState, ParamState, State
from brainstate.graph import Node
from brainstate.typing import PyTree
from brainstate.util import FlattedDict

optax_installed = importlib.util.find_spec('optax') is not None

__all__ = [
  'OptaxOptimizer',
]


class OptaxOptimizer(Node):
  """Simple train state for the common case with a single Optax optimizer.

  Example usage::

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import brainstate as bst
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
    >>> x = bst.random.randn(1, 2)
    >>> y = jnp.ones((1, 4))
    ...
    >>> model = Model()
    >>> tx = optax.adam(1e-3)
    >>> optimizer = bst.optim.OptaxOptimizer(model.states(bst.ParamState), tx)
    ...
    >>> loss_fn = lambda: ((model(x) - y) ** 2).mean()
    >>> loss_fn()
    Array(1.7055722, dtype=float32)
    >>> grads = bst.augment.grad(loss_fn, model.states(bst.ParamState))()
    >>> optimizer.update(grads)
    >>> loss_fn()
    Array(1.6925814, dtype=float32)

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Attributes:
    states: The parameter states to update.
    tx: An Optax gradient transformation.
    opt_state: The Optax optimizer state.
  """

  def __init__(
      self,
      states: FlattedDict[Hashable, ParamState],
      tx: 'optax.GradientTransformation',
  ):
    """
    Instantiate the class and wrap the :class:`FlattedDict` and Optax gradient
    transformation. Instantiate the optimizer state to keep track of
    :class:`State`.

    Args:
      states: A module.
      tx: An Optax gradient transformation.
    """

    # tx must be an instance of optax.GradientTransformation
    import optax  # type: ignore[import-not-found,import-untyped]
    if not isinstance(tx, optax.GradientTransformation):
      raise TypeError(f"tx must be an instance of optax.GradientTransformation, got {tx}")
    self.tx = tx

    # model
    if not isinstance(states, dict):
      raise TypeError(f"states must be a dict, got {states}")
    for k, v in states.items():
      if not isinstance(v, State):
        raise TypeError(f"states values must be ParamState, got {v}")
    self.states = states

    # wrt
    self.opt_state = ShortTermState(tx.init({k: v.value for k, v in states.items()}))

  def update(self, grads: Dict[Hashable, PyTree]):
    """Update the model states with the gradients.

    Args:
      grads: the gradients derived from ``brainstate.augment.grad``.
    """
    import optax  # type: ignore[import-not-found,import-untyped]
    grads = {k: grads[k] for k in self.states.keys()}
    states = {k: v.value for k, v in self.states.items()}

    # compute updates
    updates, new_opt_state = self.tx.update(grads, self.opt_state.value, states)
    new_params = optax.apply_updates(states, updates)

    # update model states and optimizer states
    for k, v in self.states.items():
      v.value = new_params[k]
    self.opt_state.value = new_opt_state
