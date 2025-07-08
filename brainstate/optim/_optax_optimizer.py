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


import importlib.util
from typing import Hashable, Dict, Optional

from brainstate._state import ShortTermState, State, StateDictManager
from brainstate.typing import PyTree
from ._base import Optimizer

optax_installed = importlib.util.find_spec('optax') is not None

__all__ = [
    'OptaxOptimizer',
    'LBFGS',
]


class OptaxOptimizer(Optimizer):
    """Simple train state for the common case with a single Optax optimizer.

    Example usage::

      >>> import jax
      >>> import jax.numpy as jnp
      >>> import brainstate as brainstate
      >>> import optax
      ...
      >>> class Model(brainstate.nn.Module):
      ...   def __init__(self):
      ...     super().__init__()
      ...     self.linear1 = brainstate.nn.Linear(2, 3)
      ...     self.linear2 = brainstate.nn.Linear(3, 4)
      ...   def __call__(self, x):
      ...     return self.linear2(self.linear1(x))
      ...
      >>> x = brainstate.random.randn(1, 2)
      >>> y = jnp.ones((1, 4))
      ...
      >>> model = Model()
      >>> tx = optax.adam(1e-3)
      >>> optimizer = brainstate.optim.OptaxOptimizer(tx)
      >>> optimizer.register_trainable_weights(model.states(brainstate.ParamState))
      ...
      >>> loss_fn = lambda: ((model(x) - y) ** 2).mean()
      >>> loss_fn()
      Array(1.7055722, dtype=float32)
      >>> grads = brainstate.augment.grad(loss_fn, model.states(brainstate.ParamState))()
      >>> optimizer.update(grads)
      >>> loss_fn()
      Array(1.6925814, dtype=float32)

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Attributes:
      param_states: The parameter states to update.
      tx: An Optax gradient transformation.
    """

    param_states: StateDictManager
    opt_state: Optional[ShortTermState]

    def __init__(
        self,
        tx: 'optax.GradientTransformation',
    ):
        """
        Instantiate the class and wrap the :class:`FlattedDict` and Optax gradient
        transformation. Instantiate the optimizer state to keep track of
        :class:`State`.

        Args:
          tx: An Optax gradient transformation.
        """
        super().__init__()

        # tx must be an instance of optax.GradientTransformation
        import optax  # type: ignore[import-not-found,import-untyped]
        if not isinstance(tx, optax.GradientTransformation):
            raise TypeError(f"tx must be an instance of optax.GradientTransformation, got {tx}")
        self.tx = tx

        # optimizer state
        self.opt_state = None

    def register_trainable_weights(self, param_states: Dict[Hashable, State]):
        # model
        if not isinstance(param_states, dict):
            raise TypeError(f"states must be a dict, got {param_states}")
        for k, v in param_states.items():
            if not isinstance(v, State):
                raise TypeError(f"states values must be ParamState, got {v}")
        self.param_states.update(param_states)
        self.param_states.unique_()

        # wrt
        self.opt_state = ShortTermState(self.tx.init({k: v.value for k, v in self.param_states.items()}))
        return self

    def update(self, grads: Dict[Hashable, PyTree]):
        """Update the model states with the gradients.

        Args:
          grads: the gradients derived from ``brainstate.augment.grad``.
        """
        if self.opt_state is None:
            raise ValueError("register_trainable_weights must be called before update.")

        import optax  # type: ignore[import-not-found,import-untyped]
        grads = {k: grads[k] for k in self.param_states.keys()}
        states = {k: v.value for k, v in self.param_states.items()}

        # compute updates
        updates, new_opt_state = self.tx.update(grads, self.opt_state.value, states)
        new_params = optax.apply_updates(states, updates)

        # update model states and optimizer states
        for k, v in self.param_states.items():
            v.value = new_params[k]
        self.opt_state.value = new_opt_state


class LBFGS(OptaxOptimizer):
    def __init__(
        self,
        lr: float,
        memory_size: int = 10,
        scale_init_precond: bool = True,
    ):
        import optax  # type: ignore[import-not-found,import-untyped]
        super().__init__(
            optax.lbfgs(
                lr,
                memory_size=memory_size,
                scale_init_precond=scale_init_precond,
                linesearch=None,
            )
        )
