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

"""
Parameter regularization classes.

This module provides regularization classes that can be applied to parameters
during training to encourage certain properties (sparsity, smoothness, etc.)
and prevent overfitting.
"""

from abc import ABC, abstractmethod

import brainstate
import brainunit as u

from ._module import Module
from ._utils import get_size, get_value

Data = brainstate.typing.ArrayLike
Size = brainstate.typing.Size

__all__ = [
    'Regularization',
    'ChainedReg',
    'GaussianReg',
    'L1Reg',
    'L2Reg',
    'ElasticNetReg',
    'HuberReg',
    'GroupLassoReg',
    'TotalVariationReg',
    'MaxNormReg',
    'EntropyReg',
    'OrthogonalReg',
    'SpectralNormReg',
    # Prior distribution-based regularizations
    'StudentTReg',
    'CauchyReg',
    'UniformReg',
    'LogNormalReg',
    'ExponentialReg',
    'GammaReg',
    'BetaReg',
    'HorseshoeReg',
    'InverseGammaReg',
    'LogUniformReg',
    'SpikeAndSlabReg',
    'DirichletReg',
]


class Regularization(Module, ABC):
    """
    Abstract base class for parameter regularization.

    Provides the interface for implementing regularization terms that can be
    added to the training loss. Subclasses must implement ``loss``, ``sample_init``,
    and ``reset_value`` methods.

    Parameters
    ----------
    fit_hyper : bool, optional
        Whether to optimize the hyperparameters of the regularization
        as trainable parameters. Default is ``False``.

    Attributes
    ----------
    fit_hyper : bool
        Whether hyperparameters are trainable.

    Notes
    -----
    Regularization can be used with the ``Param`` class to add regularization
    terms to the training loss.
    """

    __module__ = 'brainstate.nn'

    def __init__(self, fit_hyper: bool = False):
        super().__init__()
        self.fit_hyper = fit_hyper

    @abstractmethod
    def loss(self, value: Data) -> Data:
        """
        Calculate regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values to compute regularization for.

        Returns
        -------
        array_like
            Scalar regularization loss.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_init(self, shape: Size) -> Data:
        """
        Sample initial value from the regularization's implied prior distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the parameter to initialize.

        Returns
        -------
        array_like
            Sampled initial value.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_value(self) -> Data:
        """
        Return the reset value (e.g., prior mean).

        Returns
        -------
        array_like
            Value to reset the parameter to.
        """
        raise NotImplementedError


class ChainedReg(Regularization):
    """
    Composite regularization that chains multiple regularizations together.

    Combines multiple regularization priors into a single composite regularization
    by summing their losses. This allows applying multiple constraints or priors
    simultaneously to parameters.

    Parameters
    ----------
    *regularizations : Regularization
        Variable number of regularization instances to chain together.
    weight : float, optional
        Overall regularization weight (lambda) that scales the combined loss.
        Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Attributes
    ----------
    regularizations : tuple of Regularization
        The regularizations being combined.
    weight : array_like or ParamState
        Regularization weight (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import ChainedReg, L1Reg, L2Reg, UniformReg
    >>> # Combine L1 sparsity + L2 smoothness + bounded constraint
    >>> reg = ChainedReg(
    ...     L1Reg(weight=0.01),
    ...     L2Reg(weight=0.001),
    ...     UniformReg(weight=1.0, lower=-1.0, upper=1.0),
    ...     weight=1.0
    ... )
    >>> value = jnp.array([0.5, -0.3, 0.8])
    >>> loss = reg.loss(value)

    Notes
    -----
    - The ``loss()`` method returns the sum of all component regularization losses,
      scaled by the overall weight.
    - The ``sample_init()`` and ``reset_value()`` methods use the first
      regularization in the chain, as it's typically the most interpretable prior.
    - An empty chain will return zero loss and zero for sample_init/reset_value.
    - Each regularization is stored as a submodule for proper state management.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        *regularizations,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper=fit_hyper)
        self.regularizations = regularizations
        # Register each regularization as a submodule for proper state tracking
        for i, reg in enumerate(regularizations):
            setattr(self, f'_reg_{i}', reg)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate combined regularization loss.

        Sums the losses from all component regularizations, scaled by weight.

        Parameters
        ----------
        value : array_like
            Parameter values to compute regularization for.

        Returns
        -------
        array_like
            Sum of all regularization losses, scaled by weight.
        """
        if len(self.regularizations) == 0:
            return 0.0
        total_loss = self.regularizations[0].loss(value)
        for reg in self.regularizations[1:]:
            total_loss = total_loss + reg.loss(value)
        return self.weight * total_loss

    def sample_init(self, shape: Size) -> Data:
        """
        Sample initial value from the first regularization's prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the parameter to initialize.

        Returns
        -------
        array_like
            Sampled initial value from the first regularization,
            or zeros if the chain is empty.
        """
        if len(self.regularizations) == 0:
            return u.math.zeros(get_size(shape))
        return self.regularizations[0].sample_init(shape)

    def reset_value(self) -> Data:
        """
        Return the reset value from the first regularization.

        Returns
        -------
        array_like
            Reset value from the first regularization,
            or zero if the chain is empty.
        """
        if len(self.regularizations) == 0:
            return 0.0
        return self.regularizations[0].reset_value()


class GaussianReg(Regularization):
    r"""
    Gaussian prior regularization.

    Implements regularization based on the negative log-likelihood of a
    Gaussian distribution:

    .. math::
        L = \lambda \left( \sum_i \text{precision}_i \cdot (x_i - \mu_i)^2 - \sum_i \log(\text{precision}_i) \right)

    where precision = 1/std^2 and :math:`\lambda` is the weight.

    Parameters
    ----------
    mean : array_like
        Prior mean value.
    std : array_like
        Prior standard deviation.
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize mean, precision, and weight as trainable parameters.
        Default is ``False``.

    Attributes
    ----------
    mean : array_like or ParamState
        Prior mean (trainable if ``fit_hyper=True``).
    precision : array_like or ParamState
        Prior precision (trainable if ``fit_hyper=True``).
    weight : array_like or ParamState
        Regularization weight (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import GaussianReg
    >>> reg = GaussianReg(mean=0.0, std=1.0, weight=0.01)
    >>> value = jnp.array([0.5, -0.5])
    >>> loss = reg.loss(value)
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        mean: Data,
        std: Data,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        mean_t = u.math.asarray(mean, dtype=brainstate.environ.dftype())
        std_t = u.math.asarray(std, dtype=brainstate.environ.dftype())
        precision = 1.0 / (std_t ** 2 + 1e-8)
        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())

        if fit_hyper:
            mean_t = brainstate.ParamState(mean_t)
            precision = brainstate.ParamState(precision)
        self.mean = mean_t
        self.precision = precision
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Gaussian regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Gaussian negative log-likelihood loss.
        """

        # Add lower bound for numerical stability
        prec = u.math.relu(get_value(self.precision)) + 1e-6
        loss = u.math.sum(prec * (value - get_value(self.mean)) ** 2) - u.math.sum(u.math.log(prec))
        return self.weight * loss

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Gaussian prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from N(mean, std^2).
        """
        noise = brainstate.random.randn(*get_size(shape))
        std = 1.0 / u.math.sqrt(u.math.relu(get_value(self.precision)) + 1e-8)
        return get_value(self.mean) + std * noise

    def reset_value(self) -> Data:
        """
        Return the prior mean.

        Returns
        -------
        array_like
            The mean value.
        """
        return get_value(self.mean)


class L1Reg(Regularization):
    r"""
    L1 (Lasso) regularization.

    Implements L1 regularization:

    .. math::
        L = \lambda \sum_i |x_i|

    The corresponding prior is the Laplace distribution.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Attributes
    ----------
    weight : array_like or ParamState
        Regularization weight (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import L1Reg
    >>> reg = L1Reg(weight=0.01)
    >>> value = jnp.array([1.0, -2.0, 0.5])
    >>> loss = reg.loss(value)  # Returns 0.01 * (1.0 + 2.0 + 0.5)

    Notes
    -----
    L1 regularization encourages sparsity in the parameter values.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate L1 regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            L1 loss: self.weight * sum(|value|).
        """
        return self.weight * u.math.sum(u.math.abs(value))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Laplace prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Laplace(0, 1/weight).
        """
        # L1 prior corresponds to Laplace distribution
        # Sample from Laplace(0, 1/weight)
        scale = 1.0 / (self.weight + 1e-8)
        u_ = brainstate.random.rand(*get_size(shape)) - 0.5
        return scale * u.math.sign(u_) * u.math.log(1 - 2 * u.math.abs(u_) + 1e-8)

    def reset_value(self) -> Data:
        """
        Return zero (the mode of Laplace(0, b)).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class L2Reg(Regularization):
    r"""
    L2 (Ridge) regularization.

    Implements L2 regularization:

    .. math::
        L = \lambda \sum_i x_i^2

    The corresponding prior is the Gaussian distribution with zero mean.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Attributes
    ----------
    weight : array_like or ParamState
        Regularization weight (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import L2Reg
    >>> reg = L2Reg(weight=0.01)
    >>> value = jnp.array([1.0, -2.0, 0.5])
    >>> loss = reg.loss(value)  # Returns 0.01 * (1.0 + 4.0 + 0.25)

    Notes
    -----
    L2 regularization encourages small parameter values and is more
    numerically stable than L1 regularization.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate L2 regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            L2 loss: self.weight * sum(value^2).
        """
        return brainstate.nn.relu(get_value(self.weight)) * u.math.sum(value ** 2)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Gaussian prior with zero mean.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from N(0, 1/weight).
        """
        # L2 prior corresponds to Gaussian with zero mean
        std = 1.0 / u.math.sqrt(self.weight + 1e-8)
        return std * brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero (the mean of N(0, sigma^2)).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class ElasticNetReg(Regularization):
    r"""
    Elastic Net regularization (combination of L1 and L2).

    Implements a weighted combination of L1 and L2 regularization:

    .. math::
        L = \alpha \cdot \lambda_1 \sum_i |x_i| + (1 - \alpha) \cdot \lambda_2 \sum_i x_i^2

    where :math:`\alpha \in [0, 1]` controls the mix between L1 and L2.

    Parameters
    ----------
    l1_weight : float, optional
        Weight for L1 regularization. Default is 1.0.
    l2_weight : float, optional
        Weight for L2 regularization. Default is 1.0.
    alpha : float, optional
        Mixing ratio between L1 and L2 (0 = pure L2, 1 = pure L1).
        Default is 0.5.
    fit_hyper : bool, optional
        Whether to optimize weights as trainable parameters.
        Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import ElasticNetReg
    >>> reg = ElasticNetReg(l1_weight=0.01, l2_weight=0.01, alpha=0.5)
    >>> value = jnp.array([1.0, -2.0, 0.5])
    >>> loss = reg.loss(value)

    Notes
    -----
    Elastic Net combines the sparsity-inducing property of L1 with the
    stability of L2 regularization, making it useful when there are
    correlated features.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        alpha: float = 0.5,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        l1_weight_t = u.math.asarray(l1_weight, dtype=brainstate.environ.dftype())
        l2_weight_t = u.math.asarray(l2_weight, dtype=brainstate.environ.dftype())
        alpha_t = u.math.asarray(alpha, dtype=brainstate.environ.dftype())

        if fit_hyper:
            alpha_t = brainstate.ParamState(alpha_t)
        self.l1_weight = l1_weight_t
        self.l2_weight = l2_weight_t
        self.alpha = alpha_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Elastic Net regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Combined L1 and L2 loss.
        """
        alpha = u.math.clip(get_value(self.alpha), 0.0, 1.0)

        l1_loss = self.l1_weight * u.math.sum(u.math.abs(value))
        l2_loss = self.l2_weight * u.math.sum(value ** 2)
        return alpha * l1_loss + (1.0 - alpha) * l2_loss

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from a mixture prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from the mixture distribution.
        """
        alpha = u.math.clip(get_value(self.alpha), 0.0, 1.0)
        l1_weight = u.math.relu(get_value(self.l1_weight)) + 1e-8
        l2_weight = u.math.relu(get_value(self.l2_weight)) + 1e-8

        # Sample from Gaussian (L2 prior)
        std_l2 = 1.0 / u.math.sqrt(l2_weight)
        sample_l2 = std_l2 * brainstate.random.randn(*get_size(shape))

        # Sample from Laplace (L1 prior)
        scale_l1 = 1.0 / l1_weight
        u_ = brainstate.random.rand(*get_size(shape)) - 0.5
        sample_l1 = scale_l1 * u.math.sign(u_) * u.math.log(1 - 2 * u.math.abs(u_) + 1e-8)

        # Mix samples based on alpha
        return alpha * sample_l1 + (1.0 - alpha) * sample_l2

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class HuberReg(Regularization):
    r"""
    Huber regularization (robust regularization).

    Implements regularization using the Huber loss function, which behaves
    like L2 for small values and L1 for large values:

    .. math::
        L = \lambda \sum_i \begin{cases}
            \frac{1}{2} x_i^2 & \text{if } |x_i| \leq \delta \\
            \delta (|x_i| - \frac{1}{2}\delta) & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    delta : float, optional
        Threshold for switching between L2 and L1 behavior. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import HuberReg
    >>> reg = HuberReg(weight=0.01, delta=1.0)
    >>> value = jnp.array([0.5, 2.0, -3.0])
    >>> loss = reg.loss(value)

    Notes
    -----
    Huber regularization is more robust to outliers than L2 while being
    more stable than L1 for small values.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        delta: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        delta_t = u.math.asarray(delta, dtype=brainstate.environ.dftype())

        if fit_hyper:
            delta_t = brainstate.ParamState(delta_t)
        self.weight = weight_t
        self.delta = delta_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Huber regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Huber loss.
        """

        delta = u.math.relu(get_value(self.delta)) + 1e-8

        abs_val = u.math.abs(value)
        quadratic = 0.5 * value ** 2
        linear = delta * (abs_val - 0.5 * delta)
        huber = u.math.where(abs_val <= delta, quadratic, linear)
        return self.weight * u.math.sum(huber)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Huber prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample approximately from Huber prior (using Gaussian).
        """
        # Approximate with Gaussian for simplicity
        std = 1.0 / u.math.sqrt(self.weight)
        return std * brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class GroupLassoReg(Regularization):
    r"""
    Group Lasso regularization.

    Implements Group Lasso which encourages entire groups of parameters
    to be zero together:

    .. math::
        L = \lambda \sum_g \sqrt{\sum_{i \in g} x_i^2}

    where g indexes groups of parameters.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    group_size : int, optional
        Size of each group. Default is 1 (equivalent to L1).
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import GroupLassoReg
    >>> reg = GroupLassoReg(weight=0.01, group_size=4)
    >>> value = jnp.array([1.0, 0.5, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0])
    >>> loss = reg.loss(value)

    Notes
    -----
    Group Lasso is useful when parameters naturally form groups (e.g.,
    all weights connecting to one neuron) and you want entire groups
    to be zeroed out together.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        group_size: int = 1,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t
        self.group_size = group_size

    def loss(self, value: Data) -> Data:
        """
        Calculate Group Lasso regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Group Lasso loss.
        """

        # Flatten the input
        flat = u.math.reshape(value, (-1,))
        n_elements = flat.shape[0]

        # Pad if necessary to make divisible by group_size
        remainder = n_elements % self.group_size
        if remainder != 0:
            padding = self.group_size - remainder
            flat = u.math.concatenate([flat, u.math.zeros(padding)])

        # Reshape into groups and compute L2 norm of each group
        groups = u.math.reshape(flat, (-1, self.group_size))
        group_norms = u.math.sqrt(u.math.sum(groups ** 2, axis=1) + 1e-8)

        return self.weight * u.math.sum(group_norms)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Group Lasso prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample (using Gaussian approximation).
        """
        std = 1.0 / u.math.sqrt(self.weight)
        return std * brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class TotalVariationReg(Regularization):
    r"""
    Total Variation regularization.

    Encourages smoothness by penalizing differences between adjacent values:

    .. math::
        L = \lambda \sum_i |x_{i+1} - x_i|

    For order=2 (second derivative):

    .. math::
        L = \lambda \sum_i |x_{i+2} - 2x_{i+1} + x_i|

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    order : int, optional
        Order of the difference (1 for first derivative, 2 for second).
        Default is 1.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import TotalVariationReg
    >>> reg = TotalVariationReg(weight=0.01, order=1)
    >>> value = jnp.array([1.0, 1.2, 1.1, 1.3, 1.2])
    >>> loss = reg.loss(value)

    Notes
    -----
    Total Variation is commonly used in image processing to encourage
    piecewise constant solutions while preserving edges.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        order: int = 1,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t
        self.order = order

    def loss(self, value: Data) -> Data:
        """
        Calculate Total Variation regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Total Variation loss.
        """

        # Flatten the input
        flat = u.math.reshape(value, (-1,))

        if self.order == 1:
            # First order: |x_{i+1} - x_i|
            diff = flat[1:] - flat[:-1]
        else:
            # Second order: |x_{i+2} - 2*x_{i+1} + x_i|
            diff = flat[2:] - 2 * flat[1:-1] + flat[:-2]

        return self.weight * u.math.sum(u.math.abs(diff))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample with smooth prior (correlated Gaussian).

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Smooth sample using cumulative sum of noise.
        """
        std = 1.0 / u.math.sqrt(self.weight)
        # Generate smooth samples using cumulative sum
        noise = std * brainstate.random.randn(*get_size(shape))
        return u.math.cumsum(noise.flatten()).reshape(get_size(shape)) / u.math.sqrt(
            float(u.math.prod(u.math.asarray(get_size(shape)))))

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class MaxNormReg(Regularization):
    r"""
    Max Norm regularization (soft constraint).

    Implements a soft constraint on the L2 norm of parameters:

    .. math::
        L = \lambda \cdot \max(0, \|x\|_2 - c)^2

    where c is the maximum allowed norm.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    max_value : float, optional
        Maximum allowed norm. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import MaxNormReg
    >>> reg = MaxNormReg(weight=1.0, max_value=3.0)
    >>> value = jnp.array([2.0, 2.0, 2.0])  # norm = sqrt(12) > 3
    >>> loss = reg.loss(value)  # penalty applied

    Notes
    -----
    Max Norm regularization is useful for constraining the capacity of
    neural networks without penalizing small weights.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        max_value: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        max_value_t = u.math.asarray(max_value, dtype=brainstate.environ.dftype())
        if fit_hyper:
            max_value_t = brainstate.ParamState(max_value_t)
        self.weight = weight_t
        self.max_value = max_value_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Max Norm regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Max Norm penalty (zero if norm <= max_value).
        """

        max_val = u.math.relu(get_value(self.max_value)) + 1e-8

        norm = u.math.sqrt(u.math.sum(value ** 2) + 1e-8)
        violation = u.math.relu(norm - max_val)
        return self.weight * violation ** 2

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from truncated Gaussian.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample with norm <= max_value.
        """
        max_val = u.math.relu(get_value(self.max_value)) + 1e-8

        # Sample from Gaussian and project to ball
        sample = brainstate.random.randn(*get_size(shape))
        norm = u.math.sqrt(u.math.sum(sample ** 2) + 1e-8)
        # Scale to be within the ball
        scale = max_val / (norm + 1e-8) * 0.5  # 0.5 to be safely inside
        return sample * u.math.minimum(scale, 1.0)

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class EntropyReg(Regularization):
    r"""
    Entropy regularization.

    Regularizes based on the entropy of softmax-normalized values:

    .. math::
        L = -\lambda \sum_i p_i \log(p_i)

    where :math:`p = \text{softmax}(x)`.

    When ``maximize=True``, maximizes entropy (encourages uniform distribution).
    When ``maximize=False``, minimizes entropy (encourages concentrated distribution).

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    maximize : bool, optional
        Whether to maximize entropy (True) or minimize it (False).
        Default is True (maximize entropy).
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import EntropyReg
    >>> reg = EntropyReg(weight=0.01, maximize=True)
    >>> value = jnp.array([1.0, 2.0, 1.0])
    >>> loss = reg.loss(value)

    Notes
    -----
    Entropy regularization is useful in attention mechanisms and
    reinforcement learning to encourage exploration.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        maximize: bool = True,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t
        self.maximize = maximize

    def loss(self, value: Data) -> Data:
        """
        Calculate Entropy regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (will be softmax-normalized).

        Returns
        -------
        array_like
            Entropy-based loss.
        """

        # Flatten and compute softmax
        flat = u.math.reshape(value, (-1,))
        # Numerically stable softmax
        max_val = u.math.max(flat)
        exp_val = u.math.exp(flat - max_val)
        probs = exp_val / (u.math.sum(exp_val) + 1e-8)

        # Compute entropy: -sum(p * log(p))
        entropy = -u.math.sum(probs * u.math.log(probs + 1e-8))

        if self.maximize:
            # Return negative entropy to maximize it (minimize loss)
            return -self.weight * entropy
        else:
            # Return positive entropy to minimize it
            return self.weight * entropy

    def sample_init(self, shape: Size) -> Data:
        """
        Sample initial values.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Uniform or concentrated initialization based on maximize.
        """
        if self.maximize:
            # For max entropy, initialize uniformly
            return u.math.zeros(get_size(shape))
        else:
            # For min entropy, initialize with some variation
            return brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class OrthogonalReg(Regularization):
    r"""
    Orthogonal regularization.

    Encourages weight matrices to be orthogonal by penalizing deviation
    from orthogonality:

    .. math::
        L = \lambda \|W^T W - I\|_F^2

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import OrthogonalReg
    >>> reg = OrthogonalReg(weight=0.01)
    >>> W = jnp.array([[1.0, 0.1], [0.1, 1.0]])
    >>> loss = reg.loss(W)

    Notes
    -----
    Orthogonal regularization is particularly useful for RNNs where it
    helps prevent vanishing/exploding gradients. Works best with
    2D weight matrices.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Orthogonal regularization loss.

        Parameters
        ----------
        value : array_like
            Weight matrix (2D) or flattened parameters.

        Returns
        -------
        array_like
            Orthogonality penalty.
        """

        # Ensure 2D matrix
        if len(value.shape) == 1:
            # For 1D, reshape to column vector
            n = int(u.math.sqrt(float(value.shape[0])))
            if n * n == value.shape[0]:
                W = u.math.reshape(value, (n, n))
            else:
                # Cannot form square matrix, use as column
                W = u.math.reshape(value, (-1, 1))
        elif len(value.shape) > 2:
            # Flatten to 2D
            W = u.math.reshape(value, (value.shape[0], -1))
        else:
            W = value

        # Compute W^T W
        WtW = u.math.matmul(W.T, W)

        # Identity matrix
        I = u.math.eye(WtW.shape[0])

        # Frobenius norm of (W^T W - I)
        diff = WtW - I
        return self.weight * u.math.sum(diff ** 2)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from uniform distribution on orthogonal matrices.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Orthogonal initialization using QR decomposition.
        """
        shape_tuple = get_size(shape)
        if len(shape_tuple) == 1:
            n = shape_tuple[0]
            # Generate random matrix and orthogonalize
            random_matrix = brainstate.random.randn(int(u.math.sqrt(float(n))), int(u.math.sqrt(float(n))))
            q, _ = u.math.linalg.qr(random_matrix)
            return u.math.reshape(q, (n,))[:n]
        elif len(shape_tuple) == 2:
            m, n = shape_tuple
            random_matrix = brainstate.random.randn(m, n)
            q, _ = u.math.linalg.qr(random_matrix)
            return q[:m, :n]
        else:
            # For higher dimensions, use random normal
            return brainstate.random.randn(*shape_tuple) * 0.1

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class SpectralNormReg(Regularization):
    r"""
    Spectral Norm regularization.

    Penalizes the spectral norm (largest singular value) of weight matrices:

    .. math::
        L = \lambda \cdot \max(0, \sigma_{\max}(W) - c)^2

    where :math:`\sigma_{\max}` is the largest singular value and c is the
    maximum allowed value.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    max_value : float, optional
        Maximum allowed spectral norm. Default is 1.0.
    n_power_iterations : int, optional
        Number of power iterations for estimating spectral norm.
        Default is 1.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import SpectralNormReg
    >>> reg = SpectralNormReg(weight=1.0, max_value=1.0)
    >>> W = jnp.array([[2.0, 0.0], [0.0, 0.5]])  # spectral norm = 2
    >>> loss = reg.loss(W)

    Notes
    -----
    Spectral normalization is useful for stabilizing GAN training and
    controlling the Lipschitz constant of neural networks.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        max_value: float = 1.0,
        n_power_iterations: int = 1,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        max_value_t = u.math.asarray(max_value, dtype=brainstate.environ.dftype())
        if fit_hyper:
            max_value_t = brainstate.ParamState(max_value_t)
        self.weight = weight_t
        self.max_value = max_value_t
        self.n_power_iterations = n_power_iterations

    def _estimate_spectral_norm(self, W: Data) -> Data:
        """
        Estimate spectral norm using power iteration.

        Parameters
        ----------
        W : array_like
            Weight matrix.

        Returns
        -------
        array_like
            Estimated spectral norm.
        """
        # Ensure 2D
        if len(W.shape) == 1:
            return u.math.sqrt(u.math.sum(W ** 2))
        elif len(W.shape) > 2:
            W = u.math.reshape(W, (W.shape[0], -1))

        # Power iteration
        v = brainstate.random.randn(W.shape[1])
        v = v / (u.math.sqrt(u.math.sum(v ** 2)) + 1e-8)

        for _ in range(self.n_power_iterations):
            u_ = u.math.matmul(W, v)
            u_ = u_ / (u.math.sqrt(u.math.sum(u_ ** 2)) + 1e-8)
            v = u.math.matmul(W.T, u_)
            v = v / (u.math.sqrt(u.math.sum(v ** 2)) + 1e-8)

        # Spectral norm estimate
        Wv = u.math.matmul(W, v)
        sigma = u.math.sqrt(u.math.sum(Wv ** 2) + 1e-8)
        return sigma

    def loss(self, value: Data) -> Data:
        """
        Calculate Spectral Norm regularization loss.

        Parameters
        ----------
        value : array_like
            Weight matrix.

        Returns
        -------
        array_like
            Spectral norm penalty.
        """

        max_val = u.math.relu(get_value(self.max_value)) + 1e-8

        sigma = self._estimate_spectral_norm(value)
        violation = u.math.relu(sigma - max_val)
        return self.weight * violation ** 2

    def sample_init(self, shape: Size) -> Data:
        """
        Sample with bounded spectral norm.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample with spectral norm approximately bounded.
        """
        max_val = u.math.relu(get_value(self.max_value)) + 1e-8
        shape_tuple = get_size(shape)

        sample = brainstate.random.randn(*shape_tuple)

        # Scale by max_value / estimated_norm
        if len(shape_tuple) >= 2:
            sigma = self._estimate_spectral_norm(sample)
            scale = max_val / (sigma + 1e-8) * 0.5
            return sample * scale
        else:
            norm = u.math.sqrt(u.math.sum(sample ** 2) + 1e-8)
            return sample * max_val / (norm + 1e-8) * 0.5

    def reset_value(self) -> Data:
        """
        Return zero.

        Returns
        -------
        float
            Zero.
        """
        return 0.0


# =============================================================================
# Prior Distribution-Based Regularizations
# =============================================================================


class StudentTReg(Regularization):
    r"""
    Student's t-distribution prior regularization.

    Implements regularization based on the negative log-likelihood of a
    Student's t-distribution, which has heavier tails than Gaussian:

    .. math::
        L = \lambda \sum_i \log\left(1 + \frac{(x_i / s)^2}{\nu}\right)

    where :math:`\nu` is the degrees of freedom and :math:`s` is the scale.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    df : float, optional
        Degrees of freedom (nu). Lower values give heavier tails.
        Default is 3.0.
    scale : float, optional
        Scale parameter. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import StudentTReg
    >>> reg = StudentTReg(weight=1.0, df=3.0, scale=1.0)
    >>> value = jnp.array([0.5, 2.0, -1.0])
    >>> loss = reg.loss(value)

    Notes
    -----
    Student's t prior is more robust to outliers than Gaussian. As df -> infinity,
    it approaches a Gaussian distribution. df=1 gives the Cauchy distribution.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        df: float = 3.0,
        scale: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        df_t = u.math.asarray(df, dtype=brainstate.environ.dftype())
        scale_t = u.math.asarray(scale, dtype=brainstate.environ.dftype())

        if fit_hyper:
            df_t = brainstate.ParamState(df_t)
            scale_t = brainstate.ParamState(scale_t)
        self.weight = weight_t
        self.df = df_t
        self.scale = scale_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Student's t regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Student's t negative log-likelihood loss.
        """

        df = u.math.relu(get_value(self.df)) + 1e-8
        scale = u.math.relu(get_value(self.scale)) + 1e-8

        z = value / scale
        return self.weight * u.math.sum(u.math.log(1.0 + z ** 2 / df))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Student's t distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Student's t distribution.
        """
        df = u.math.relu(get_value(self.df)) + 1e-8
        scale = u.math.relu(get_value(self.scale)) + 1e-8

        # Approximate with scaled Gaussian for simplicity
        # True Student-t would require ratio of Gaussian/Chi
        return scale * brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero (the mode of symmetric Student's t).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class CauchyReg(Regularization):
    r"""
    Cauchy prior regularization.

    Implements regularization based on the negative log-likelihood of a
    Cauchy distribution (Student's t with df=1), which has very heavy tails:

    .. math::
        L = \lambda \sum_i \log\left(1 + (x_i / s)^2\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    scale : float, optional
        Scale parameter. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import CauchyReg
    >>> reg = CauchyReg(weight=1.0, scale=1.0)
    >>> value = jnp.array([0.5, 5.0, -1.0])
    >>> loss = reg.loss(value)

    Notes
    -----
    Cauchy prior allows for very large parameter values, making it extremely
    robust but also allowing outliers. It has no defined mean or variance.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        scale: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        scale_t = u.math.asarray(scale, dtype=brainstate.environ.dftype())

        if fit_hyper:
            scale_t = brainstate.ParamState(scale_t)
        self.weight = weight_t
        self.scale = scale_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Cauchy regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Cauchy negative log-likelihood loss.
        """

        scale = u.math.relu(get_value(self.scale)) + 1e-8

        z = value / scale
        return self.weight * u.math.sum(u.math.log(1.0 + z ** 2))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Cauchy distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Cauchy distribution.
        """
        scale = u.math.relu(get_value(self.scale)) + 1e-8

        # Cauchy = tan(pi * (U - 0.5)) where U ~ Uniform(0,1)
        u_ = brainstate.random.rand(*get_size(shape))
        return scale * u.math.tan(u.math.pi * (u_ - 0.5))

    def reset_value(self) -> Data:
        """
        Return zero (the mode of symmetric Cauchy).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class UniformReg(Regularization):
    r"""
    Uniform prior regularization (soft bounded constraint).

    Implements a soft constraint that encourages parameters to stay within
    a specified interval [lower, upper]:

    .. math::
        L = \lambda \sum_i \left(\text{relu}(l - x_i)^2 + \text{relu}(x_i - u)^2\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    lower : float, optional
        Lower bound. Default is -1.0.
    upper : float, optional
        Upper bound. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import UniformReg
    >>> reg = UniformReg(weight=1.0, lower=-1.0, upper=1.0)
    >>> value = jnp.array([0.5, 1.5, -0.5])  # 1.5 is out of bounds
    >>> loss = reg.loss(value)

    Notes
    -----
    This is a soft constraint; values outside the bounds are penalized but
    not strictly prohibited.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        lower: float = -1.0,
        upper: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        lower_t = u.math.asarray(lower, dtype=brainstate.environ.dftype())
        upper_t = u.math.asarray(upper, dtype=brainstate.environ.dftype())

        if fit_hyper:
            lower_t = brainstate.ParamState(lower_t)
            upper_t = brainstate.ParamState(upper_t)
        self.weight = weight_t
        self.lower = lower_t
        self.upper = upper_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Uniform regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Penalty for values outside bounds.
        """

        lower = get_value(self.lower)
        upper = get_value(self.upper)

        below = u.math.relu(lower - value) ** 2
        above = u.math.relu(value - upper) ** 2
        return self.weight * u.math.sum(below + above)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Uniform distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Uniform(lower, upper).
        """
        lower = get_value(self.lower)
        upper = get_value(self.upper)

        return lower + (upper - lower) * brainstate.random.rand(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return the midpoint of the interval.

        Returns
        -------
        array_like
            (lower + upper) / 2.
        """
        return (get_value(self.lower) + get_value(self.upper)) / 2.0


class LogNormalReg(Regularization):
    r"""
    Log-normal prior regularization (for positive parameters).

    Implements regularization based on the negative log-likelihood of a
    log-normal distribution:

    .. math::
        L = \lambda \sum_i \left(\frac{(\log x_i - \mu)^2}{2\sigma^2} + \log x_i\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    mu : float, optional
        Mean of log(x). Default is 0.0.
    sigma : float, optional
        Standard deviation of log(x). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import LogNormalReg
    >>> reg = LogNormalReg(weight=1.0, mu=0.0, sigma=1.0)
    >>> value = jnp.array([0.5, 1.0, 2.0])  # positive values
    >>> loss = reg.loss(value)

    Notes
    -----
    Log-normal prior is appropriate for parameters that must be positive,
    such as scales or variances. Values <= 0 will produce invalid results.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        mu_t = u.math.asarray(mu, dtype=brainstate.environ.dftype())
        sigma_t = u.math.asarray(sigma, dtype=brainstate.environ.dftype())

        if fit_hyper:
            mu_t = brainstate.ParamState(mu_t)
            sigma_t = brainstate.ParamState(sigma_t)
        self.weight = weight_t
        self.mu = mu_t
        self.sigma = sigma_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Log-normal regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (should be positive).

        Returns
        -------
        array_like
            Log-normal negative log-likelihood loss.
        """

        mu = get_value(self.mu)
        sigma = u.math.relu(get_value(self.sigma)) + 1e-8

        # Ensure positive values
        x_pos = u.math.relu(value) + 1e-8
        log_x = u.math.log(x_pos)
        return self.weight * u.math.sum((log_x - mu) ** 2 / (2 * sigma ** 2) + log_x)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Log-normal distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from LogNormal(mu, sigma).
        """
        mu = get_value(self.mu)
        sigma = u.math.relu(get_value(self.sigma)) + 1e-8

        return u.math.exp(mu + sigma * brainstate.random.randn(*get_size(shape)))

    def reset_value(self) -> Data:
        """
        Return the median of log-normal (exp(mu)).

        Returns
        -------
        array_like
            exp(mu).
        """
        return u.math.exp(get_value(self.mu))


class ExponentialReg(Regularization):
    r"""
    Exponential prior regularization (for positive parameters).

    Implements regularization based on the negative log-likelihood of an
    exponential distribution:

    .. math::
        L = \lambda \cdot \text{rate} \sum_i x_i

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    rate : float, optional
        Rate parameter (1/mean). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import ExponentialReg
    >>> reg = ExponentialReg(weight=1.0, rate=1.0)
    >>> value = jnp.array([0.5, 1.0, 2.0])  # positive values
    >>> loss = reg.loss(value)

    Notes
    -----
    Exponential prior encourages small positive values and promotes sparsity.
    It's the continuous analog of L1 regularization for positive parameters.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        rate: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        rate_t = u.math.asarray(rate, dtype=brainstate.environ.dftype())

        if fit_hyper:
            rate_t = brainstate.ParamState(rate_t)
        self.weight = weight_t
        self.rate = rate_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Exponential regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (should be positive).

        Returns
        -------
        array_like
            Exponential negative log-likelihood loss.
        """

        rate = u.math.relu(get_value(self.rate)) + 1e-8

        # Penalize negative values heavily, then apply exponential loss
        x_pos = u.math.relu(value)
        return self.weight * rate * u.math.sum(x_pos)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Exponential distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Exponential(rate).
        """
        rate = u.math.relu(get_value(self.rate)) + 1e-8

        # Exponential via inverse transform: -log(U) / rate
        u_ = brainstate.random.rand(*get_size(shape))
        return -u.math.log(u_ + 1e-8) / rate

    def reset_value(self) -> Data:
        """
        Return the mode of exponential (0).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class GammaReg(Regularization):
    r"""
    Gamma prior regularization (for positive parameters).

    Implements regularization based on the negative log-likelihood of a
    Gamma distribution:

    .. math::
        L = -\lambda \sum_i \left((\alpha - 1) \log x_i - \beta x_i\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    alpha : float, optional
        Shape parameter. Default is 2.0.
    beta : float, optional
        Rate parameter. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import GammaReg
    >>> reg = GammaReg(weight=1.0, alpha=2.0, beta=1.0)
    >>> value = jnp.array([0.5, 1.0, 2.0])  # positive values
    >>> loss = reg.loss(value)

    Notes
    -----
    Gamma prior is flexible for positive parameters. alpha=1 gives
    exponential distribution. The mode is (alpha-1)/beta for alpha >= 1.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 2.0,
        beta: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        alpha_t = u.math.asarray(alpha, dtype=brainstate.environ.dftype())
        beta_t = u.math.asarray(beta, dtype=brainstate.environ.dftype())

        if fit_hyper:
            alpha_t = brainstate.ParamState(alpha_t)
            beta_t = brainstate.ParamState(beta_t)
        self.weight = weight_t
        self.alpha = alpha_t
        self.beta = beta_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Gamma regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (should be positive).

        Returns
        -------
        array_like
            Gamma negative log-likelihood loss.
        """

        alpha = u.math.relu(get_value(self.alpha)) + 1e-8
        beta = u.math.relu(get_value(self.beta)) + 1e-8

        # Ensure positive values
        x_pos = u.math.relu(value) + 1e-8
        # Negative log-likelihood (ignoring constants)
        return self.weight * u.math.sum(-(alpha - 1) * u.math.log(x_pos) + beta * x_pos)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Gamma distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Gamma(alpha, beta).
        """
        alpha = u.math.relu(get_value(self.alpha)) + 1e-8
        beta = u.math.relu(get_value(self.beta)) + 1e-8

        # Use JAX's gamma sampling: gamma(shape_param, scale, size)
        # scale = 1/beta for our parameterization
        return brainstate.random.gamma(alpha, scale=1.0 / beta, size=get_size(shape))

    def reset_value(self) -> Data:
        """
        Return the mode of Gamma ((alpha-1)/beta for alpha >= 1).

        Returns
        -------
        array_like
            Mode value.
        """
        alpha = u.math.relu(get_value(self.alpha)) + 1e-8
        beta = u.math.relu(get_value(self.beta)) + 1e-8
        return u.math.maximum((alpha - 1) / beta, 1e-8)


class BetaReg(Regularization):
    r"""
    Beta prior regularization (for parameters in [0, 1]).

    Implements regularization based on the negative log-likelihood of a
    Beta distribution:

    .. math::
        L = -\lambda \sum_i \left((a - 1) \log x_i + (b - 1) \log(1 - x_i)\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    a : float, optional
        First shape parameter. Default is 2.0.
    b : float, optional
        Second shape parameter. Default is 2.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import BetaReg
    >>> reg = BetaReg(weight=1.0, a=2.0, b=2.0)
    >>> value = jnp.array([0.3, 0.5, 0.7])  # values in [0, 1]
    >>> loss = reg.loss(value)

    Notes
    -----
    Beta prior is appropriate for probability parameters. a=b=1 gives
    uniform distribution. The mode is (a-1)/(a+b-2) for a,b > 1.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        a: float = 2.0,
        b: float = 2.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        a_t = u.math.asarray(a, dtype=brainstate.environ.dftype())
        b_t = u.math.asarray(b, dtype=brainstate.environ.dftype())

        if fit_hyper:
            a_t = brainstate.ParamState(a_t)
            b_t = brainstate.ParamState(b_t)
        self.weight = weight_t
        self.a = a_t
        self.b = b_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Beta regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (should be in [0, 1]).

        Returns
        -------
        array_like
            Beta negative log-likelihood loss.
        """

        a = u.math.relu(get_value(self.a)) + 1e-8
        b = u.math.relu(get_value(self.b)) + 1e-8

        # Clip to (0, 1) for numerical stability
        x_clip = u.math.clip(value, 1e-8, 1.0 - 1e-8)
        # Negative log-likelihood (ignoring constants)
        return self.weight * u.math.sum(-(a - 1) * u.math.log(x_clip) - (b - 1) * u.math.log(1 - x_clip))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Beta distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Beta(a, b).
        """
        a = u.math.relu(get_value(self.a)) + 1e-8
        b = u.math.relu(get_value(self.b)) + 1e-8

        return brainstate.random.beta(a, b, size=get_size(shape))

    def reset_value(self) -> Data:
        """
        Return the mode of Beta ((a-1)/(a+b-2) for a,b > 1).

        Returns
        -------
        array_like
            Mode value.
        """
        a = u.math.relu(get_value(self.a)) + 1e-8
        b = u.math.relu(get_value(self.b)) + 1e-8
        return u.math.clip((a - 1) / (a + b - 2 + 1e-8), 0.0, 1.0)


class HorseshoeReg(Regularization):
    r"""
    Horseshoe prior regularization (strong sparsity with heavy tails).

    Implements an approximation to the horseshoe prior using a
    log-penalty formulation:

    .. math::
        L = \lambda \sum_i \log\left(1 + (x_i / \tau)^2\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    tau : float, optional
        Global scale parameter. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import HorseshoeReg
    >>> reg = HorseshoeReg(weight=1.0, tau=0.1)
    >>> value = jnp.array([0.01, 0.5, 2.0])
    >>> loss = reg.loss(value)

    Notes
    -----
    The horseshoe prior provides strong shrinkage toward zero for small
    coefficients while leaving large coefficients relatively unshrunk.
    This is useful for sparse signal recovery and variable selection.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        tau: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        tau_t = u.math.asarray(tau, dtype=brainstate.environ.dftype())

        if fit_hyper:
            tau_t = brainstate.ParamState(tau_t)
        self.weight = weight_t
        self.tau = tau_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Horseshoe regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Horseshoe-like penalty.
        """

        tau = u.math.relu(get_value(self.tau)) + 1e-8

        z = value / tau
        return self.weight * u.math.sum(u.math.log(1.0 + z ** 2))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from approximate Horseshoe prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample approximating horseshoe prior.
        """
        tau = u.math.relu(get_value(self.tau)) + 1e-8

        # Approximate with Cauchy (heavy-tailed) scaled by tau
        u_ = brainstate.random.rand(*get_size(shape))
        local_scale = u.math.abs(u.math.tan(u.math.pi * (u_ - 0.5)))
        return tau * local_scale * brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero (the mode).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class InverseGammaReg(Regularization):
    r"""
    Inverse-Gamma prior regularization (for variance parameters).

    Implements regularization based on the negative log-likelihood of an
    Inverse-Gamma distribution:

    .. math::
        L = \lambda \sum_i \left((\alpha + 1) \log x_i + \frac{\beta}{x_i}\right)

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    alpha : float, optional
        Shape parameter. Default is 2.0.
    beta : float, optional
        Scale parameter. Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import InverseGammaReg
    >>> reg = InverseGammaReg(weight=1.0, alpha=2.0, beta=1.0)
    >>> value = jnp.array([0.5, 1.0, 2.0])  # positive values
    >>> loss = reg.loss(value)

    Notes
    -----
    Inverse-Gamma is commonly used as a prior for variance parameters
    in Bayesian models. The mode is beta/(alpha+1).
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 2.0,
        beta: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        alpha_t = u.math.asarray(alpha, dtype=brainstate.environ.dftype())
        beta_t = u.math.asarray(beta, dtype=brainstate.environ.dftype())

        if fit_hyper:
            alpha_t = brainstate.ParamState(alpha_t)
            beta_t = brainstate.ParamState(beta_t)
        self.weight = weight_t
        self.alpha = alpha_t
        self.beta = beta_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Inverse-Gamma regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (should be positive).

        Returns
        -------
        array_like
            Inverse-Gamma negative log-likelihood loss.
        """

        alpha = u.math.relu(get_value(self.alpha)) + 1e-8
        beta = u.math.relu(get_value(self.beta)) + 1e-8

        # Ensure positive values
        x_pos = u.math.relu(value) + 1e-8
        # Negative log-likelihood (ignoring constants)
        return self.weight * u.math.sum((alpha + 1) * u.math.log(x_pos) + beta / x_pos)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Inverse-Gamma distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from InverseGamma(alpha, beta).
        """
        alpha = u.math.relu(get_value(self.alpha)) + 1e-8
        beta = u.math.relu(get_value(self.beta)) + 1e-8

        # InverseGamma(alpha, beta) = beta / Gamma(alpha, scale=1)
        gamma_sample = brainstate.random.gamma(alpha, scale=1.0, size=get_size(shape))
        return beta / (gamma_sample + 1e-8)

    def reset_value(self) -> Data:
        """
        Return the mode of Inverse-Gamma (beta/(alpha+1)).

        Returns
        -------
        array_like
            Mode value.
        """
        alpha = u.math.relu(get_value(self.alpha)) + 1e-8
        beta = u.math.relu(get_value(self.beta)) + 1e-8
        return beta / (alpha + 1)


class LogUniformReg(Regularization):
    r"""
    Log-uniform (Jeffreys) prior regularization (scale-invariant).

    Implements regularization based on the negative log-likelihood of a
    log-uniform distribution:

    .. math::
        L = \lambda \sum_i \log x_i

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    lower : float, optional
        Lower bound. Default is 1e-3.
    upper : float, optional
        Upper bound. Default is 1e3.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import LogUniformReg
    >>> reg = LogUniformReg(weight=1.0, lower=1e-3, upper=1e3)
    >>> value = jnp.array([0.1, 1.0, 10.0])  # positive values
    >>> loss = reg.loss(value)

    Notes
    -----
    Log-uniform (Jeffreys) prior is scale-invariant and commonly used
    as a weakly informative prior for scale parameters.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        lower: float = 1e-3,
        upper: float = 1e3,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        lower_t = u.math.asarray(lower, dtype=brainstate.environ.dftype())
        upper_t = u.math.asarray(upper, dtype=brainstate.environ.dftype())

        if fit_hyper:
            lower_t = brainstate.ParamState(lower_t)
            upper_t = brainstate.ParamState(upper_t)
        self.weight = weight_t
        self.lower = lower_t
        self.upper = upper_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Log-uniform regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (should be positive).

        Returns
        -------
        array_like
            Log-uniform negative log-likelihood loss.
        """

        lower = u.math.relu(get_value(self.lower)) + 1e-8
        upper = u.math.relu(get_value(self.upper)) + 1e-8

        # Ensure positive values and clip to bounds
        x_pos = u.math.clip(u.math.relu(value) + 1e-8, lower, upper)
        # p(x) proportional to 1/x, so -log p(x) proportional to log(x)
        return self.weight * u.math.sum(u.math.log(x_pos))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Log-uniform distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from LogUniform(lower, upper).
        """
        lower = u.math.relu(get_value(self.lower)) + 1e-8
        upper = u.math.relu(get_value(self.upper)) + 1e-8

        # Sample: exp(Uniform(log(lower), log(upper)))
        log_lower = u.math.log(lower)
        log_upper = u.math.log(upper)
        log_sample = log_lower + (log_upper - log_lower) * brainstate.random.rand(*get_size(shape))
        return u.math.exp(log_sample)

    def reset_value(self) -> Data:
        """
        Return the geometric mean (sqrt(lower * upper)).

        Returns
        -------
        array_like
            Geometric mean.
        """
        lower = u.math.relu(get_value(self.lower)) + 1e-8
        upper = u.math.relu(get_value(self.upper)) + 1e-8
        return u.math.sqrt(lower * upper)


class SpikeAndSlabReg(Regularization):
    r"""
    Spike-and-slab prior regularization (variable selection).

    Implements a soft approximation to the spike-and-slab mixture prior:

    .. math::
        L = -\lambda \sum_i \log\left(\pi \cdot \text{spike}(x_i) + (1-\pi) \cdot \text{slab}(x_i)\right)

    where spike is a narrow Gaussian and slab is a wide Gaussian.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    spike_scale : float, optional
        Scale of the spike (narrow) component. Default is 0.01.
    slab_scale : float, optional
        Scale of the slab (wide) component. Default is 1.0.
    pi : float, optional
        Mixture weight (probability of spike). Default is 0.5.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import SpikeAndSlabReg
    >>> reg = SpikeAndSlabReg(weight=1.0, spike_scale=0.01, slab_scale=1.0, pi=0.5)
    >>> value = jnp.array([0.001, 0.5, -0.002])
    >>> loss = reg.loss(value)

    Notes
    -----
    Spike-and-slab priors are the gold standard for sparse Bayesian learning
    and variable selection. The spike component encourages exact sparsity
    while the slab allows for large coefficients.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        spike_scale: float = 0.01,
        slab_scale: float = 1.0,
        pi: float = 0.5,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        spike_scale_t = u.math.asarray(spike_scale, dtype=brainstate.environ.dftype())
        slab_scale_t = u.math.asarray(slab_scale, dtype=brainstate.environ.dftype())
        pi_t = u.math.asarray(pi, dtype=brainstate.environ.dftype())

        if fit_hyper:
            spike_scale_t = brainstate.ParamState(spike_scale_t)
            slab_scale_t = brainstate.ParamState(slab_scale_t)
            pi_t = brainstate.ParamState(pi_t)
        self.weight = weight_t
        self.spike_scale = spike_scale_t
        self.slab_scale = slab_scale_t
        self.pi = pi_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Spike-and-slab regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Spike-and-slab negative log-likelihood loss.
        """

        spike_scale = u.math.relu(get_value(self.spike_scale)) + 1e-8
        slab_scale = u.math.relu(get_value(self.slab_scale)) + 1e-8
        pi = u.math.clip(get_value(self.pi), 1e-8, 1.0 - 1e-8)

        # Gaussian densities (unnormalized, up to constant)
        spike_density = u.math.exp(-0.5 * (value / spike_scale) ** 2) / spike_scale
        slab_density = u.math.exp(-0.5 * (value / slab_scale) ** 2) / slab_scale

        # Mixture density
        mixture = pi * spike_density + (1 - pi) * slab_density
        return self.weight * u.math.sum(-u.math.log(mixture + 1e-8))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from Spike-and-slab mixture.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from spike-and-slab mixture.
        """
        spike_scale = u.math.relu(get_value(self.spike_scale)) + 1e-8
        slab_scale = u.math.relu(get_value(self.slab_scale)) + 1e-8
        pi = u.math.clip(get_value(self.pi), 1e-8, 1.0 - 1e-8)

        # Sample component indicators
        is_spike = brainstate.random.rand(*get_size(shape)) < pi

        # Sample from both components
        spike_samples = spike_scale * brainstate.random.randn(*get_size(shape))
        slab_samples = slab_scale * brainstate.random.randn(*get_size(shape))

        return u.math.where(is_spike, spike_samples, slab_samples)

    def reset_value(self) -> Data:
        """
        Return zero (the mode of the spike).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class DirichletReg(Regularization):
    r"""
    Dirichlet prior regularization (for probability simplexes).

    Implements regularization based on the negative log-likelihood of a
    Dirichlet distribution applied to softmax-normalized values:

    .. math::
        L = -\lambda \sum_i (\alpha_i - 1) \log p_i

    where :math:`p = \text{softmax}(x)`.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    alpha : float, optional
        Concentration parameter (same for all dimensions). Default is 1.0.
        Values < 1 encourage sparsity, values > 1 encourage uniformity.
    fit_hyper : bool, optional
        Whether to optimize hyperparameters. Default is ``False``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import DirichletReg
    >>> reg = DirichletReg(weight=1.0, alpha=1.0)  # uniform prior
    >>> value = jnp.array([1.0, 2.0, 1.0])  # logits
    >>> loss = reg.loss(value)

    Notes
    -----
    Dirichlet prior is appropriate for attention weights, mixture proportions,
    and other probability simplexes. alpha=1 is uniform, alpha<1 encourages
    sparsity, alpha>1 encourages uniformity.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        alpha_t = u.math.asarray(alpha, dtype=brainstate.environ.dftype())

        if fit_hyper:
            alpha_t = brainstate.ParamState(alpha_t)
        self.weight = weight_t
        self.alpha = alpha_t

    def loss(self, value: Data) -> Data:
        """
        Calculate Dirichlet regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values (logits, will be softmax-normalized).

        Returns
        -------
        array_like
            Dirichlet negative log-likelihood loss.
        """

        alpha = u.math.relu(get_value(self.alpha)) + 1e-8

        # Flatten and compute softmax
        flat = u.math.reshape(value, (-1,))
        # Numerically stable softmax
        max_val = u.math.max(flat)
        exp_val = u.math.exp(flat - max_val)
        probs = exp_val / (u.math.sum(exp_val) + 1e-8)

        # Dirichlet negative log-likelihood (ignoring constants)
        return self.weight * u.math.sum(-(alpha - 1) * u.math.log(probs + 1e-8))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample logits that give Dirichlet-distributed probabilities.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Logits corresponding to Dirichlet-sampled probabilities.
        """
        alpha = u.math.relu(get_value(self.alpha)) + 1e-8

        # Sample from Dirichlet via Gamma
        # For symmetric Dirichlet, sample Gamma(alpha, 1) and normalize
        gamma_samples = brainstate.random.gamma(alpha, scale=1.0, size=get_size(shape))
        probs = gamma_samples / (u.math.sum(gamma_samples) + 1e-8)

        # Convert to logits (inverse softmax)
        return u.math.log(probs + 1e-8)

    def reset_value(self) -> Data:
        """
        Return zero (gives uniform distribution under softmax).

        Returns
        -------
        float
            Zero.
        """
        return 0.0
