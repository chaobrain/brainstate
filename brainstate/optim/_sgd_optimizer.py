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

# -*- coding: utf-8 -*-

import functools
from typing import Union, Dict, Optional, Tuple, Any, TypeVar

import brainunit as u
import jax
import jax.numpy as jnp

from brainstate import environ
from brainstate._state import State, LongTermState, StateDictManager
from ._base import Optimizer
from ._lr_scheduler import make_schedule, LearningRateScheduler

__all__ = [
    'to_same_dict_tree',

    # new class of brainstate.State for optimizer
    'OptimState',

    # commonly used optimizers
    'SGDOptimizer',
    'SGD',
    'Momentum',
    'MomentumNesterov',
    'Adagrad',
    'Adadelta',
    'RMSProp',
    'Adam',
    'LARS',
    'Adan',
    'AdamW',
]

T = TypeVar('T')


def cast(value: Any, dtype: Any) -> jax.Array:
    if isinstance(value, jax.Array):
        return value.astype(dtype)
    return jnp.asarray(value, dtype=dtype)


def fcast(value: T, dtype: Any = None) -> jax.Array:
    return cast(value, dtype=dtype or environ.dftype())


def _to_dict_value(old_dict: Dict) -> Dict:
    new_dict = dict()
    for k, v in old_dict.items():
        if isinstance(v, State):
            new_dict[k] = v.value
        else:
            new_dict[k] = v
    return new_dict


def to_same_dict_tree(*dicts: Dict):
    """
    Convert multiple dictionaries to the same tree structure.

    Parameters
    ----------
    *dicts: dict
      The dictionaries to be converted.

    Returns
    -------
    dict
      The converted dictionary.
    """
    if len(dicts):
        # all keys
        all_keys = tuple(set(d.keys()) for d in dicts)
        for keys in all_keys[1:]:
            if len(all_keys[0].difference(keys)) > 0:
                raise ValueError('Dictionary does not match.')

        # flatten to normal python dict
        r = [_to_dict_value(d) for d in dicts]

        if len(dicts) == 1:
            return r[0]
        else:
            return tuple(r)


def _sgd(prev_weight, gradient, weight_decay, lr=None):
    """
    The update function for SGD learning.

    Parameters
    ----------
    prev_weight: jax.Array
      The previous weight.
    gradient: jax.Array
      The gradient.
    weight_decay: float
      The weight decay.
    lr: float
      The learning rate.
    """
    if weight_decay is None:
        if lr is None:
            return prev_weight - gradient
        else:
            return prev_weight - lr * gradient
    else:
        if lr is None:
            return (1 - weight_decay) * prev_weight - gradient
        else:
            return (1 - weight_decay) * prev_weight - lr * gradient


class OptimState(LongTermState):
    """
    The state for optimizer.
    """
    pass


class SGDOptimizer(Optimizer):
    """
    Base Optimizer Class.

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.
    """

    lr: LearningRateScheduler  # learning rate

    def __init__(
        self, lr: Union[float, LearningRateScheduler, State],
    ):
        super().__init__()
        self.lr: LearningRateScheduler = make_schedule(lr)


class _WeightDecayOptimizer(SGDOptimizer):
    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        weight_decay: Optional[float] = None,
    ):
        super().__init__(lr=lr)
        self.lr: LearningRateScheduler = make_schedule(lr)
        assert weight_decay is None or 0. <= weight_decay <= 1., 'weight_decay must be in [0, 1].'
        self.weight_decay = (fcast(weight_decay) if weight_decay is not None else None)


class SGD(_WeightDecayOptimizer):
    r"""
    Stochastic gradient descent optimizer.

    SGD performs a parameter update for training examples :math:`x` and label
    :math:`y`:

    .. math::

        \theta = \theta - \eta \cdot \nabla_\theta J(\theta; x; y)


    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        weight_decay: Optional[float] = None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

    def register_trainable_weights(self, states: Optional[Dict[str, State]] = None):
        states = dict() if states is None else states
        assert isinstance(states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            self.param_states.add_unique_value(k, v)

    def update(self, grads: dict):
        lr = self.lr()
        weight_values, grad_values = to_same_dict_tree(self.param_states, grads)
        updates = jax.tree.map(
            functools.partial(_sgd, lr=lr, weight_decay=self.weight_decay),
            weight_values,
            grad_values
        )
        self.param_states.assign_values(updates)
        self.lr.step_call()


class Momentum(_WeightDecayOptimizer):
    r"""
    Momentum optimizer.

    Momentum [1]_ is a method that helps accelerate SGD in the relevant direction
    and dampens oscillations. It does this by adding a fraction :math:`\gamma`
    of the update vector of the past time step to the current update vector:

    .. math::

      \begin{align}
      \begin{split}
      v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\
      \theta &= \theta - v_t
      \end{split}
      \end{align}

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.

    References
    ----------

    .. [1] Qian, N. (1999). On the momentum term in gradient descent learning
           algorithms. Neural Networks : The Official Journal of the International
           Neural Network Society, 12(1), 145–151. http://doi.org/10.1016/S0893-6080(98)00116-6

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        momentum: float = 0.9,
        weight_decay: Optional[float] = None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.momentum = fcast(momentum)
        self.momentum_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'

        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.momentum_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        states_values, grad_values, momentum_values = to_same_dict_tree(
            self.param_states, grads, self.momentum_states
        )
        momentum_values = jax.tree.map(
            lambda vv, gg: self.momentum * vv - lr * gg,
            momentum_values,
            grad_values
        )
        new_weight_values = jax.tree.map(
            functools.partial(_sgd, lr=lr, weight_decay=self.weight_decay),
            states_values,
            momentum_values
        )
        self.momentum_states.assign_values(momentum_values)
        self.param_states.assign_values(new_weight_values)
        self.lr.step_call()


class MomentumNesterov(_WeightDecayOptimizer):
    r"""
    Nesterov accelerated gradient optimizer [2]_.

    .. math::

        \begin{align}
        \begin{split}
        v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\
        \theta &= \theta - v_t
        \end{split}
        \end{align}

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.

    References
    ----------
    .. [2] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        weight_decay: Optional[float] = None,
        momentum: float = 0.9,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.momentum = fcast(momentum)
        self.momentum_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.momentum_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        states_values, grad_values, momentum_values = to_same_dict_tree(self.param_states, grads, self.momentum_states)
        momentum_values = jax.tree.map(
            lambda mv, gv: self.momentum * mv - lr * gv,
            momentum_values,
            grad_values
        )
        weight_values = jax.tree.map(
            functools.partial(_sgd, lr=lr, weight_decay=self.weight_decay),
            states_values,
            momentum_values
        )
        self.param_states.assign_values(weight_values)
        self.momentum_states.assign_values(momentum_values)
        self.lr.step_call()


class Adagrad(_WeightDecayOptimizer):
    r"""
    Optimizer that implements the Adagrad algorithm.

    Adagrad [3]_ is an optimizer with parameter-specific learning rates, which are
    adapted relative to how frequently a parameter gets updated during training.
    The more updates a parameter receives, the smaller the updates.

    .. math::

        \theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}

    where :math:`G(t)` contains the sum of the squares of the past gradients

    One of Adagrad's main benefits is that it eliminates the need to manually tune
    the learning rate. Most implementations use a default value of 0.01 and leave it at that.
    Adagrad's main weakness is its accumulation of the squared gradients in the denominator:
    Since every added term is positive, the accumulated sum keeps growing during training.
    This in turn causes the learning rate to shrink and eventually become infinitesimally
    small, at which point the algorithm is no longer able to acquire additional knowledge.

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.

    References
    ----------
    .. [3] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        weight_decay: Optional[float] = None,
        epsilon: float = 1e-6,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.epsilon = fcast(epsilon)
        self.cache_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.cache_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        cache_values, grad_values, weight_values = to_same_dict_tree(self.cache_states, grads, self.param_states)
        cache_values = jax.tree.map(
            lambda cv, gv: cv + gv ** 2,
            cache_values,
            grad_values
        )
        updates = jax.tree.map(
            lambda cv, gv: lr * gv / jnp.sqrt(cv + self.epsilon),
            cache_values,
            grad_values
        )
        weight_values = jax.tree.map(
            functools.partial(_sgd, weight_decay=self.weight_decay),
            weight_values,
            updates
        )
        self.cache_states.assign_values(cache_values)
        self.param_states.assign_values(weight_values)
        self.lr.step_call()


class Adadelta(_WeightDecayOptimizer):
    r"""
    Optimizer that implements the Adadelta algorithm.

    Adadelta [4]_ optimization is a stochastic gradient descent method that is based
    on adaptive learning rate per dimension to address two drawbacks:

    - The continual decay of learning rates throughout training.
    - The need for a manually selected global learning rate.

    Adadelta is a more robust extension of Adagrad that adapts learning rates based on
    a moving window of gradient updates, instead of accumulating all past gradients.
    This way, Adadelta continues learning even when many updates have been done. Compared
    to Adagrad, in the original version of Adadelta you don't have to set an initial
    learning rate.

    .. math::

      \boldsymbol{s}_t \leftarrow \rho \boldsymbol{s}_{t-1} + (1 - \rho) \boldsymbol{g}_t \odot \boldsymbol{g}_t, \\
      \boldsymbol{g}_t' \leftarrow \sqrt{\frac{\Delta\boldsymbol{x}_{t-1} + \epsilon}{\boldsymbol{s}_t + \epsilon}}   \odot \boldsymbol{g}_t, \\
      \boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}'_t, \\
      \Delta\boldsymbol{x}_t \leftarrow \rho \Delta\boldsymbol{x}_{t-1} + (1 - \rho) \boldsymbol{g}'_t \odot \boldsymbol{g}'_t.

    :math:`\rho` should be between 0 and 1. A value of rho close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average fast.

    :math:`\rho` = 0.95 and :math:`\epsilon`=1e-6 are suggested in the paper and reported
    to work for multiple datasets (MNIST, speech).

    In the paper, no learning rate is considered (so learning_rate=1.0). Probably best to
    keep it at this value. epsilon is important for the very first update (so the
    numerator does not become 0).

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.

    References
    ----------
    .. [4] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State] = 0.01,
        weight_decay: Optional[float] = None,
        epsilon: float = 1e-6,
        rho: float = 0.95,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.epsilon = fcast(epsilon)
        self.rho = fcast(rho)
        self.cache_states = StateDictManager()
        self.delta_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.cache_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                self.delta_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        weight_values, grad_values, cache_values, delta_values = to_same_dict_tree(
            self.param_states, grads, self.cache_states, self.delta_states)
        cache_values = jax.tree.map(lambda cv, gv: self.rho * cv + (1 - self.rho) * gv ** 2, cache_values, grad_values)
        updates = jax.tree.map(lambda gv, dv, cv: gv * jnp.sqrt(dv + self.epsilon) / jnp.sqrt(cv + self.epsilon),
                               grad_values, delta_values, cache_values)
        delta_values = jax.tree.map(lambda dv, upd: self.rho * dv + (1 - self.rho) * upd ** 2, delta_values, updates)
        weight_values = jax.tree.map(functools.partial(_sgd, weight_decay=self.weight_decay),
                                     weight_values,
                                     updates)
        self.param_states.assign_values(weight_values)
        self.delta_states.assign_values(delta_values)
        self.cache_states.assign_values(cache_values)
        self.lr.step_call()


class RMSProp(_WeightDecayOptimizer):
    r"""
    Optimizer that implements the RMSprop algorithm.

    RMSprop [5]_ and Adadelta have both been developed independently around the same time
    stemming from the need to resolve Adagrad's radically diminishing learning rates.

    The gist of RMSprop is to:

    - Maintain a moving (discounted) average of the square of gradients
    - Divide the gradient by the root of this average

    .. math::

      \begin{split}c_t &= \rho c_{t-1} + (1-\rho)*g^2\\
      p_t &= \frac{\eta}{\sqrt{c_t + \epsilon}} * g \end{split}

    The centered version additionally maintains a moving average of the gradients,
    and uses that average to estimate the variance.

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.

    References
    ----------
    .. [5] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        weight_decay: Optional[float] = None,
        epsilon: float = 1e-6,
        rho: float = 0.9,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.epsilon = fcast(epsilon)
        self.rho = fcast(rho)
        self.cache_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.cache_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        weight_values, grad_values, cache_values = to_same_dict_tree(self.param_states, grads, self.cache_states)
        cache_values = jax.tree.map(lambda cv, gv: self.rho * cv + (1 - self.rho) * gv ** 2, cache_values, grad_values)
        update = jax.tree.map(lambda gv, cv: lr * gv / jnp.sqrt(cv + self.epsilon), grad_values, cache_values)
        weight_values = jax.tree.map(functools.partial(_sgd, weight_decay=self.weight_decay),
                                     weight_values,
                                     update)
        self.param_states.assign_values(weight_values)
        self.cache_states.assign_values(cache_values)
        self.lr.step_call()


class Adam(_WeightDecayOptimizer):
    """
    Optimizer that implements the Adam algorithm.

    Adam [6]_ - a stochastic gradient descent method (SGD) that computes
    individual adaptive learning rates for different parameters from estimates of
    first- and second-order moments of the gradients.

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.
    beta1: optional, float
      A positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    beta2: optional, float
      A positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, float
      A positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

    References
    ----------
    .. [6] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
    """

    def __init__(
        self,
        lr: Union[float, State, LearningRateScheduler],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: Optional[float] = None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        self.beta1 = fcast(beta1)
        self.beta2 = fcast(beta2)
        self.eps = fcast(eps)
        self.m1_states = StateDictManager()
        self.m2_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'

        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.m1_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                self.m2_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        lr = lr / (1 - self.beta1 ** (self.lr.last_epoch.value + 2))
        lr = lr * jnp.sqrt(1 - self.beta2 ** (self.lr.last_epoch.value + 2))
        weight_values, grad_values, m1_values, m2_values = to_same_dict_tree(
            self.param_states, grads, self.m1_states, self.m2_states
        )
        m1_values = jax.tree.map(
            lambda m1, gv: self.beta1 * m1 + (1 - self.beta1) * gv,
            m1_values,
            grad_values
        )
        m2_values = jax.tree.map(
            lambda m2, gv: self.beta2 * m2 + (1 - self.beta2) * gv ** 2,
            m2_values,
            grad_values
        )
        update = jax.tree.map(
            lambda m1, m2: lr * m1 / (jnp.sqrt(m2) + self.eps),
            m1_values,
            m2_values
        )
        weight_values = jax.tree.map(
            functools.partial(_sgd, weight_decay=self.weight_decay),
            weight_values,
            update
        )
        self.param_states.assign_values(weight_values)
        self.m1_states.assign_values(m1_values)
        self.m2_states.assign_values(m2_values)
        self.lr.step_call()


class LARS(_WeightDecayOptimizer):
    r"""
    Layer-wise adaptive rate scaling (LARS) optimizer [1]_.

    Layer-wise Adaptive Rate Scaling, or LARS, is a large batch
    optimization technique. There are two notable differences
    between LARS and other adaptive algorithms such as `Adam` or `RMSProp`:
    first, LARS uses a separate learning rate for each layer and not for
    each weight. And second, the magnitude of the update is controlled
    with respect to the weight norm for better control of training speed.

    .. math::

       m_{t} = \beta_{1}m_{t-1} + \left(1-\beta_{1}\right)\left(g_{t} + \lambda{x_{t}}\right) \\
       x_{t+1}^{\left(i\right)} = x_{t}^{\left(i\right)}  - \eta_{t}\frac{\phi\left(|| x_{t}^{\left(i\right)} ||\right)}{|| m_{t}^{\left(i\right)} || }m_{t}^{\left(i\right)}

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.
    momentum: float
      coefficient used for the moving average of the gradient.
    weight_decay: float
      weight decay coefficient.
    tc: float
      trust coefficient eta ( < 1) for trust ratio computation.
    eps: float
      epsilon used for trust ratio computation.

    References
    ----------
    .. [1] You, Yang, Igor Gitman and Boris Ginsburg. “Large Batch Training of Convolutional Networks.” arXiv: Computer Vision and Pattern Recognition (2017): n. pag.
    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        tc: float = 1e-3,
        eps: float = 1e-5,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        assert self.weight_decay is None, 'LARS does not support weight decay.'

        self.momentum = fcast(momentum)
        self.tc = fcast(tc)
        self.eps = fcast(eps)
        self.momentum_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.momentum_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        weight_values, grad_values, momentum_values = to_same_dict_tree(self.param_states, grads, self.momentum_states)

        def _lars_update(pv, gv, mv):
            p_norm = jnp.linalg.norm(pv)
            g_norm = jnp.linalg.norm(gv)
            trust_ratio = self.tc * p_norm / (g_norm + self.weight_decay * p_norm + self.eps)
            local_lr = lr * jnp.maximum(jnp.logical_or(p_norm == 0, g_norm == 0), trust_ratio)
            mv = self.momentum * mv + local_lr * (gv + self.weight_decay * pv)
            return mv

        momentum_values = jax.tree.map(_lars_update, weight_values, grad_values, momentum_values)
        weight_values = jax.tree.map(lambda pv, mv: pv - mv, weight_values, momentum_values)
        self.param_states.assign_values(weight_values)
        self.momentum_states.assign_values(momentum_values)
        self.lr.step_call()


class Adan(_WeightDecayOptimizer):
    r"""
    Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models [1]_.

    .. math::

       \begin{equation}
        \begin{aligned}
        & \mathbf{m}_k=\left(1-\beta_1\right) \mathbf{m}_{k-1}+\beta_1 \mathbf{g}_k \\
        & \mathbf{v}_k=\left(1-\beta_2\right) \mathbf{v}_{k-1}+\beta_2\left(\mathbf{g}_k-\mathbf{g}_{k-1}\right)  \\
        & \mathbf{n}_k=\left(1-\beta_3\right) \mathbf{n}_{k-1}+\beta_3\left[\mathbf{g}_k+\left(1-\beta_2\right)\left(\mathbf{g}_k-\mathbf{g}_{k-1}\right)\right]^2  \\
        & \boldsymbol{\eta}_k=\eta /\left(\sqrt{\mathbf{n}_k+\varepsilon}\right)  \\
        & \boldsymbol{\theta}_{k+1}=\left(1+\lambda_k \eta\right)^{-1}\left[\boldsymbol{\theta}_k-\boldsymbol{\eta}_k \circ\left(\mathbf{m}_k+\left(1-\beta_2\right) \mathbf{v}_k\right)\right] \\
        \end{aligned}
        \end{equation}

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate. Can be much higher than Adam, up to 5-10x. (default: 1e-3)
    betas : tuple
       Coefficients used for computing running averages of gradient and its norm. (default: (0.02, 0.08, 0.01))
    eps : float
      The term added to the denominator to improve numerical stability. (default: 1e-8)
    weight_decay : float
      decoupled weight decay (L2 penalty) (default: 0)
    no_prox: bool
      how to perform the decoupled weight decay (default: False).
      It determines the update rule of parameters with weight decay.
      By default, Adan updates the parameters in the way presented in Algorithm 1 in the paper:

      .. math::
         \boldsymbol{\theta}_{k+1} = ( 1+\lambda \eta)^{-1}\left[\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}k)\right],

      But one also can update the parameter like Adamw:

      .. math::
         \boldsymbol{\theta}_{k+1} = ( 1-\lambda \eta)\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}_k).

    References
    ----------
    .. [1] Xie, Xingyu, Pan Zhou, Huan Li, Zhouchen Lin and Shuicheng Yan.
           “Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing
           Deep Models.” ArXiv abs/2208.06677 (2022): n. pag.
    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State] = 1e-3,
        betas: Tuple[float, float, float] = (0.02, 0.08, 0.01),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        no_prox: bool = False,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        assert len(betas) == 3
        if eps < 0.:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))

        self.betas = fcast(jnp.asarray(betas))
        self.eps = fcast(eps)
        self.no_prox = no_prox
        self.exp_avg_states = StateDictManager()
        self.exp_avg_sq_states = StateDictManager()
        self.exp_avg_diff_states = StateDictManager()
        self.pre_grad_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.exp_avg_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                self.exp_avg_sq_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                self.exp_avg_diff_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                self.pre_grad_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()
        step = self.lr.last_epoch.value + 1
        correct_m = 1 / (1 - (1 - self.betas[0]) ** (step + 1))
        correct_v = 1 / (1 - (1 - self.betas[1]) ** (step + 1))
        correct_n = 1 / (1 - (1 - self.betas[2]) ** (step + 1))
        m_values, n_values, v_values, pre_g_values, weight_values, grad_values = to_same_dict_tree(
            self.exp_avg_states, self.exp_avg_diff_states, self.exp_avg_sq_states, self.pre_grad_states,
            self.param_states, grads)

        def _adan_update(m, n, v, pre_g, g, p):
            m = m * (1 - self.betas[0]) + self.betas[0] * g
            gd = g - pre_g
            v = v * (1 - self.betas[1]) + self.betas[1] * gd
            n = n * (1 - self.betas[2]) + self.betas[2] * (g + (1 - self.betas[1]) * gd) ** 2
            weighted_step_size = lr / (jnp.sqrt(n * correct_n) + self.eps)
            if self.no_prox:
                p = (p * (1 - self.weight_decay * lr) -
                     weighted_step_size * (m * correct_m + (1 - self.betas[1]) * v * correct_v))
            else:
                p = ((p - weighted_step_size * (m * correct_m + (1 - self.betas[1]) * v * correct_v)) /
                     (1 + self.weight_decay * lr))
            return m, n, v, p

        m_values, n_values, v_values, weight_values = jax.tree.map(
            _adan_update, m_values, n_values, v_values, pre_g_values, grad_values, weight_values)
        self.exp_avg_states.assign_values(m_values)
        self.exp_avg_diff_states.assign_values(n_values)
        self.exp_avg_sq_states.assign_values(v_values)
        self.param_states.assign_values(weight_values)
        self.lr.step_call()


class AdamW(_WeightDecayOptimizer):
    r"""
    Adam with weight decay regularization [1]_.

    AdamW uses weight decay to regularize learning towards small weights, as
    this leads to better generalization. In SGD you can also use L2 regularization
    to implement this as an additive loss term, however L2 regularization
    does not behave as intended for adaptive gradient algorithms such as Adam.

    .. math::

       \begin{aligned}
          &\rule{110mm}{0.4pt}                                                                 \\
          &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
              \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
              \: \epsilon \text{ (epsilon)}                                                    \\
          &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
              \: \textit{maximize}                                                             \\
          &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
              \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
          &\rule{110mm}{0.4pt}                                                                 \\
          &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

          &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
          &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
          &\hspace{5mm}\textbf{else}                                                           \\
          &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
          &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
          &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
          &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
          &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
          &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
          &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
          &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
              \widehat{v_t})                                                                   \\
          &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
              \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
          &\hspace{5mm}\textbf{else}                                                           \\
          &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
              \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
          &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
          &\bf{return} \:  \theta_t                                                     \\[-1.ex]
          &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}


    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.
    beta1: optional, float
      A positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates. Generally close to 1.
    beta2: optional, float
      A positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates. Generally close to 1.
    eps: optional, float
      A positive scalar value for epsilon, a small constant for
      numerical stability.
    weight_decay: float
      Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate.
    amsgrad: bool
      whether to use the AMSGrad variant of this algorithm
      from the paper `On the Convergence of Adam and Beyond`.

    References
    ----------
    .. [1] Loshchilov, Ilya and Frank Hutter. “Decoupled Weight Decay Regularization.” International Conference on Learning Representations (2019).

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        if eps < 0.:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))
        if weight_decay < 0.:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.beta1 = fcast(beta1)
        self.beta2 = fcast(beta2)
        self.eps = fcast(eps)
        self.amsgrad = amsgrad
        self.m1_states = StateDictManager()
        self.m2_states = StateDictManager()
        if self.amsgrad:
            self.vmax_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                self.m1_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                self.m2_states[k] = OptimState(u.math.tree_zeros_like(v.value))
                if self.amsgrad:
                    self.vmax_states[k] = OptimState(u.math.tree_zeros_like(v.value))

    def update(self, grads: dict):
        lr_old = self.lr()
        step = self.lr.last_epoch.value + 2
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        lr = lr_old * jnp.sqrt(bias_correction2) / bias_correction1

        def _adamw_update(p, m, v, g, vmax=None):
            if self.weight_decay != 0:
                p *= (1 - lr_old * self.weight_decay)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            if self.amsgrad:
                vmax = jnp.maximum(vmax, v)
                denom = jnp.sqrt(vmax) + self.eps
                return p - lr * m / denom, m, v, vmax
            else:
                denom = jnp.sqrt(v.value) + self.eps
                return p - lr * m / denom, m, v

        if self.amsgrad:
            weight_values, m1_values, m2_values, vmax_values = to_same_dict_tree(
                self.param_states, self.m1_states, self.m2_states, self.vmax_states)
            weight_values, m1_values, m2_values, vmax_values = jax.tree.map(
                _adamw_update, weight_values, m1_values, m2_values, grads, vmax_values)
            self.vmax_states.assign_values(vmax_values)
        else:
            weight_values, m1_values, m2_values = to_same_dict_tree(self.param_states, self.m1_states, self.m2_states)
            weight_values, m1_values, m2_values = jax.tree.map(
                _adamw_update, weight_values, m1_values, m2_values, grads)
        self.param_states.assign_values(weight_values)
        self.m1_states.assign_values(m1_values)
        self.m2_states.assign_values(m2_values)
        self.lr.step_call()


class SM3(_WeightDecayOptimizer):
    """
    SM3 algorithm [1]_.

    The 'Square-root of Minima of Sums of Maxima of Squared-gradients Method'
    (SM3) algorithm is a memory-efficient adaptive optimization algorithm similar
    to Adam and Adagrad with greatly reduced memory usage for history tensors.
    For an `n x m` matrix, Adam and Adagrad use `O(nm)` memory for history
    tensors, while SM3 uses `O(n+m)` due to the chosen cover. In general, a tensor
    of shape `(n_1, n_2, ..., n_k)` optimized using Adam will use `O(prod n_i)`
    memory for storage tensors, while the optimization using SM3 will use
    `O(sum n_i)` memory. Despite storing fewer parameters, this optimization
    algorithm manages to be comparably effective.

    This advantage drastically shrinks when `momentum > 0`. The momentum is
    tracked using a tensor of the same shape as the tensor being optimized. With
    momentum, SM3 will use just over half as much memory as Adam, and a bit more
    than Adagrad.

    Parameters
    ----------
    lr: float, LearningRateScheduler
      learning rate.
    momentum: float
      coefficient used to scale prior updates
      before adding. This drastically increases memory usage if
      `momentum > 0.0`. (default: 0.0)
    beta: float
      coefficient used for exponential moving averages (default: 0.0)
    eps: float
      Term added to square-root in denominator to
      improve numerical stability (default: 1e-30).

    References
    ----------
    .. [1] Anil, Rohan, Vineet Gupta, Tomer Koren and Yoram Singer. “Memory Efficient Adaptive Optimization.” Neural Information Processing Systems (2019).

    """

    def __init__(
        self,
        lr: Union[float, LearningRateScheduler, State],
        beta: float = 0.,
        momentum: float = 0.,
        eps: float = 1e-30,
        weight_decay: Optional[float] = None,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)

        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {0}".format(beta))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))

        self.eps = fcast(eps)
        self.beta = fcast(beta)
        self.momentum = fcast(momentum)
        self.memory_states = StateDictManager()

    def register_trainable_weights(self, train_states: Optional[Dict[str, State]] = None):
        train_states = dict() if train_states is None else train_states
        assert isinstance(train_states, dict), '"states" must be a dict of brainstate.State.'
        for k, v in train_states.items():
            assert isinstance(v, State), f'"{k}" must be an instance of brainstate.State.'
            if self.param_states.add_unique_value(k, v):
                rank, ndim, dtype = v.value.shape, v.value.ndim, v.value.dtype
                for i in range(ndim):
                    shape = [1] * ndim
                    shape[i] = rank[i]
                    self.memory_states[f'{k}_m{i}'] = State(jnp.zeros(shape, dtype=dtype))
                if self.momentum > 0.:
                    self.memory_states[f'{k}_mbuffer'] = State(jnp.zeros_like(v.value))

    def update(self, grads: dict):
        lr = self.lr()

        for k, p in self.param_states.items():
            g = grads[k]
            ndim = p.ndim
            update = self.memory_states[f'{k}_m0'].value
            for i in range(1, ndim):
                update = jnp.minimum(update, self.memory_states[f'{k}_m{i}'].value)
            if self.beta > 0.:
                update *= self.beta
            update += g * g * (1 - self.beta)
            # Computes max along all dimensions except the given dim.
            # If tensor is a scalar, it returns tensor.
            for i in range(ndim):
                result = update
                for j in range(ndim):
                    if i != j:
                        result = jnp.maximum(result, axis=j, keepdim=True)
                acc = self.memory_states[f'{k}_m{i}'].value
                if self.beta > 0.:
                    acc.value = jnp.maximum(acc, result)
                else:
                    # No need to compare - nu_max is bigger because of grad ** 2
                    acc.value = result
            update = g / jnp.sqrt(update + self.eps)
            if self.momentum > 0.:
                m_buffer = self.memory_states[f'{k}_mbuffer'].value
                update = update * (1. - self.momentum) + m_buffer * self.momentum
                m_buffer.value = update
            if self.weight_decay is None:
                p.value -= lr * update
            else:
                p.value = (1 - self.weight_decay) * p - lr * update
        self.lr.step_call()
