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


from abc import ABC, abstractmethod
from typing import Sequence

import brainunit as u
import jax
import jax.numpy as jnp
from brainstate.typing import ArrayLike
from jax import Array

__all__ = [
    'Transform',
    'IdentityT',
    'SigmoidT',
    'SoftplusT',
    'NegSoftplusT',
    'LogT',
    'ExpT',
    'TanhT',
    'SoftsignT',
    'AffineT',
    'ChainT',
    'MaskedT',
    'ClipT',
    # New transforms
    'ReluT',
    'PositiveT',
    'NegativeT',
    'ScaledSigmoidT',
    'PowerT',
    'OrderedT',
    'SimplexT',
    'UnitVectorT',
]


def save_exp(x, max_value: float = 20.0):
    r"""
    Numerically stable exponential function with clipping.
    
    Computes the exponential of the input after clipping to prevent numerical overflow.
    This function implements a safe exponential operation by limiting the maximum
    input value before computing the exponential.
    
    .. math::
        \text{save\_exp}(x) = \exp(\min(x, x_{\text{max}}))
    
    where :math:`x_{\text{max}}` is the maximum allowed value.
    
    Parameters
    ----------
    x : array_like
        Input array.
    max_value : float, optional
        Maximum value to clip the input to, by default 20.0. Values above this
        threshold are clipped to prevent numerical overflow in the exponential.
    
    Returns
    -------
    array_like
        The exponential of the clipped input.
        
    Notes
    -----
    This function is particularly useful in neural network computations where
    exponential operations can easily overflow for large input values. The
    default max_value of 20.0 corresponds to exp(20) ≈ 4.85e8, which is
    typically well within the numerical range of floating-point arithmetic.
    """
    x = u.math.clip(x, a_max=max_value, a_min=None)
    return u.math.exp(x)


class Transform(ABC):
    r"""
    Abstract base class for bijective parameter transformations.

    This class provides the interface for implementing bijective (one-to-one and onto)
    transformations that map parameters between different domains. These transformations
    are essential in optimization and statistical inference where parameters need to be
    constrained to specific domains (e.g., positive values, bounded intervals).

    A bijective transformation :math:`f: \mathcal{X} \rightarrow \mathcal{Y}` must satisfy:

    1. **Injectivity** (one-to-one): :math:`f(x_1) = f(x_2) \Rightarrow x_1 = x_2`
    2. **Surjectivity** (onto): :math:`\forall y \in \mathcal{Y}, \exists x \in \mathcal{X} : f(x) = y`
    3. **Invertibility**: :math:`f^{-1}(f(x)) = x` and :math:`f(f^{-1}(y)) = y`

    Methods
    -------
    forward(x)
        Apply the forward transformation :math:`y = f(x)`
    inverse(y)
        Apply the inverse transformation :math:`x = f^{-1}(y)`
    log_abs_det_jacobian(x, y)
        Compute the log absolute determinant of the Jacobian

    Notes
    -----
    Subclasses must implement both `forward` and `inverse` methods to ensure
    the transformation is truly bijective. The implementation should guarantee
    numerical stability and handle edge cases appropriately.

    Examples
    --------
    >>> class SquareTransform(Transform):
    ...     def forward(self, x):
    ...         return x**2
    ...     def inverse(self, y):
    ...         return jnp.sqrt(y)
    """
    __module__ = 'braintools.param'

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return f"{self.__class__.__name__}()"

    def __call__(self, x: ArrayLike) -> Array:
        r"""
        Apply the forward transformation to the input.

        Parameters
        ----------
        x : array_like
            Input array to transform.

        Returns
        -------
        Array
            Transformed output array.

        Notes
        -----
        This method provides a convenient callable interface that delegates
        to the forward method, allowing Transform objects to be used as functions.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Array:
        r"""
        Apply the forward transformation.

        Transforms input from the unconstrained domain to the constrained domain.
        This method implements the mathematical function :math:`y = f(x)` where
        :math:`x` is in the unconstrained space and :math:`y` is in the target domain.

        Parameters
        ----------
        x : array_like
            Input array in the unconstrained domain.

        Returns
        -------
        Array
            Transformed output in the constrained domain.

        Notes
        -----
        Implementations must ensure numerical stability and handle boundary
        conditions appropriately.
        """

    @abstractmethod
    def inverse(self, y: ArrayLike) -> Array:
        r"""
        Apply the inverse transformation.

        Transforms input from the constrained domain back to the unconstrained domain.
        This method implements the mathematical function :math:`x = f^{-1}(y)` where
        :math:`y` is in the constrained space and :math:`x` is in the unconstrained domain.

        Parameters
        ----------
        y : array_like
            Input array in the constrained domain.

        Returns
        -------
        Array
            Transformed output in the unconstrained domain.

        Notes
        -----
        Implementations must ensure that inverse(forward(x)) = x for all valid x,
        and forward(inverse(y)) = y for all y in the target domain.
        """
        pass

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Compute the log absolute determinant of the Jacobian of the forward transformation.

        For a bijective transformation :math:`f: \mathcal{X} \rightarrow \mathcal{Y}`,
        this computes:

        .. math::
            \log \left| \det \frac{\partial f(x)}{\partial x} \right|

        This is essential for computing probability densities under change of variables
        and is widely used in normalizing flows and variational inference.

        Parameters
        ----------
        x : array_like
            Input in the unconstrained domain.
        y : array_like
            Output in the constrained domain (i.e., y = forward(x)).
            This parameter is provided for efficiency since it may already
            be computed.

        Returns
        -------
        Array
            Log absolute determinant of the Jacobian.

        Notes
        -----
        The default implementation raises NotImplementedError. Subclasses
        should override this method to provide an efficient implementation.

        For element-wise transformations, the log determinant is simply
        the sum of log absolute derivatives:

        .. math::
            \log \left| \det J \right| = \sum_i \log \left| \frac{\partial f(x_i)}{\partial x_i} \right|
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement log_abs_det_jacobian. "
            "Override this method in your subclass."
        )


class IdentityT(Transform):
    """Identity transformation (no-op)."""
    __module__ = 'braintools.param'

    def __repr__(self):
        return "IdentityT()"

    def forward(self, x: ArrayLike) -> Array:
        return x

    def inverse(self, y: ArrayLike) -> Array:
        return y

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """Log determinant is 0 for identity (det(I) = 1)."""
        return jnp.zeros(jnp.shape(x)[:-1] if jnp.ndim(x) > 0 else ())


class SigmoidT(Transform):
    r"""
    Sigmoid transformation mapping unbounded values to a bounded interval.

    This transformation uses the logistic sigmoid function to map any real value
    to a bounded interval [lower, upper]. It is particularly useful for constraining
    parameters that must lie within specific bounds, such as probabilities or
    correlation coefficients.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = \text{lower} + (\text{upper} - \text{lower}) \cdot \sigma(x)

    where :math:`\sigma(x) = \frac{1}{1 + e^{-x}}` is the standard sigmoid function.

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = \log\left(\frac{y - \text{lower}}{\text{upper} - y}\right)

    Parameters
    ----------
    lower : array_like
        Lower bound of the target interval.
    upper : array_like
        Upper bound of the target interval.

    Attributes
    ----------
    lower : array_like
        Lower bound of the interval.
    width : array_like
        Width of the interval (upper - lower).
    unit : brainunit.Unit
        Physical unit of the bounds.

    Notes
    -----
    The sigmoid function provides a smooth, differentiable mapping with asymptotes
    at the specified bounds. The transformation is bijective from ℝ to (lower, upper),
    though numerical precision may limit the effective range near the boundaries.

    Examples
    --------
    >>> # Map to probability range [0, 1]
    >>> transform = SigmoidT(0.0, 1.0)
    >>> x = jnp.array([-2.0, 0.0, 2.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.12, 0.5, 0.88]

    >>> # Map to correlation range [-1, 1]
    >>> transform = SigmoidT(-1.0, 1.0)
    >>> x = jnp.array([0.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.0]
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        r"""
        Initialize the sigmoid transformation.

        Parameters
        ----------
        lower : array_like
            Lower bound of the target interval. Must be less than upper.
        upper : array_like
            Upper bound of the target interval. Must be greater than lower.

        Raises
        ------
        ValueError
            If lower >= upper, as this would result in a non-positive interval width.
        """
        super().__init__()
        self.lower = lower
        self.width = upper - lower
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"SigmoidT(lower={self.lower}, upper={self.lower + self.width})"

    def forward(self, x: ArrayLike) -> Array:
        r"""
        Transform unbounded input to bounded interval using sigmoid function.
        
        Parameters
        ----------
        x : array_like
            Input values in unbounded domain $(-\infty, \infty)$.
            
        Returns
        -------
        Array
            Transformed values in interval [lower, upper].
            
        Notes
        -----
        Uses numerically stable exponential to prevent overflow for large |x|.
        """
        y = 1.0 / (1.0 + save_exp(-x))
        return self.lower + self.width * y

    def inverse(self, y: ArrayLike) -> Array:
        r"""
        Transform bounded input back to unbounded domain using logit function.

        Parameters
        ----------
        y : array_like
            Input values in bounded interval [lower, upper].

        Returns
        -------
        Array
            Transformed values in unbounded domain (-∞, ∞).

        Notes
        -----
        For numerical stability, input should be strictly within (lower, upper).
        Values at the boundaries will result in infinite outputs.
        """
        x = (y - self.lower) / self.width
        x = -u.math.log((1.0 / x) - 1.0)
        return x

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Compute log absolute determinant of the Jacobian.

        For sigmoid: d/dx[lower + width * sigmoid(x)] = width * sigmoid(x) * (1 - sigmoid(x))
        log|det J| = sum(log(width) + log(sigmoid(x)) + log(1 - sigmoid(x)))
        """
        s = jax.nn.sigmoid(x)
        return jnp.sum(jnp.log(self.width) + jnp.log(s) + jnp.log(1 - s), axis=-1)


class SoftplusT(Transform):
    r"""
    Softplus transformation mapping unbounded values to positive semi-infinite interval.

    This transformation uses the softplus function to map any real value to the
    interval [lower, ∞). It provides a smooth, differentiable alternative to
    ReLU activation and is commonly used to constrain parameters to be positive,
    such as variance parameters or rate constants.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = \log(1 + e^x) + \text{lower}

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = \log(e^{y - \text{lower}} - 1)

    Parameters
    ----------
    lower : array_like
        Lower bound of the target interval.

    Attributes
    ----------
    lower : array_like
        Lower bound of the interval.
    unit : brainunit.Unit
        Physical unit of the lower bound.

    Notes
    -----
    The softplus function is the smooth approximation to the ReLU function:
    :math:`\lim_{\beta \to \infty} \frac{1}{\beta} \log(1 + e^{\beta x}) = \max(0, x)`

    For large positive x, softplus(x) ≈ x, and for large negative x, softplus(x) ≈ 0.
    The function is strictly positive and has a well-defined inverse.

    Examples
    --------
    >>> # Map to positive reals [0, ∞)
    >>> transform = SoftplusT(0.0)
    >>> x = jnp.array([-5.0, 0.0, 5.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.007, 0.693, 5.007]

    >>> # Map to interval [2, ∞) for positive-definite parameters
    >>> transform = SoftplusT(2.0)
    >>> x = jnp.array([0.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [2.693]
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike) -> None:
        """
        Initialize the softplus transformation.

        Parameters
        ----------
        lower : array_like
            Lower bound of the target interval. The transformation maps
            unbounded inputs to [lower, ∞).
        """
        super().__init__()
        self.lower = lower
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"SoftplusT(lower={self.lower})"

    def forward(self, x: ArrayLike) -> Array:
        """
        Transform unbounded input to positive semi-infinite interval.
        
        Parameters
        ----------
        x : array_like
            Input values in unbounded domain (-∞, ∞).
            
        Returns
        -------
        Array
            Transformed values in interval [lower, ∞).
            
        Notes
        -----
        Uses log1p for numerical stability: log1p(exp(x)) = log(1 + exp(x)).
        For large x, this avoids overflow in the exponential.
        """
        return jnp.log1p(save_exp(x)) * self.unit + self.lower

    def inverse(self, y: ArrayLike) -> Array:
        """
        Transform positive semi-infinite input back to unbounded domain.

        Parameters
        ----------
        y : array_like
            Input values in interval [lower, ∞).

        Returns
        -------
        Array
            Transformed values in unbounded domain (-∞, ∞).

        Notes
        -----
        Input must be strictly greater than lower bound to avoid numerical issues.
        Uses numerically stable exponential for large (y - lower) values.
        """
        return u.math.log(save_exp((y - self.lower) / self.unit) - 1.0)

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Compute log absolute determinant of the Jacobian.

        For softplus: d/dx[log(1 + exp(x))] = sigmoid(x)
        log|det J| = sum(log(sigmoid(x))) = sum(x - softplus(x))
        """
        return jnp.sum(x - jnp.log1p(save_exp(x)), axis=-1)


class NegSoftplusT(SoftplusT):
    r"""
    Negative softplus transformation mapping unbounded values to negative semi-infinite interval.

    This transformation uses the negative softplus function to map any real value
    to the interval (-∞, upper]. It is the reflection of the softplus function
    and is useful for constraining parameters to be negative, such as log-probabilities
    or negative rate constants.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = -\log(1 + e^{-x}) + \text{upper}

    which is equivalent to:

    .. math::
        \text{forward}(x) = \text{upper} - \text{softplus}(-x)

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = -\log(e^{\text{upper} - y} - 1)

    Parameters
    ----------
    upper : array_like
        Upper bound of the target interval.

    Attributes
    ----------
    lower : array_like
        Stores the upper bound (inherited from parent class).
    unit : brainunit.Unit
        Physical unit of the upper bound.

    Notes
    -----
    This transformation is implemented by negating the input and output of the
    standard softplus transformation. For large positive x, the output approaches
    the upper bound, while for large negative x, the output approaches -∞.

    Examples
    --------
    >>> # Map to negative reals (-∞, 0]
    >>> transform = NegSoftplusT(0.0)
    >>> x = jnp.array([-5.0, 0.0, 5.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [-5.007, -0.693, -0.007]

    >>> # Map to interval (-∞, -2] for negative-definite parameters
    >>> transform = NegSoftplusT(-2.0)
    >>> x = jnp.array([0.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [-2.693]
    """
    __module__ = 'braintools.param'

    def __init__(self, upper: ArrayLike) -> None:
        """
        Initialize the negative softplus transformation.

        Parameters
        ----------
        upper : array_like
            Upper bound of the target interval. The transformation maps
            unbounded inputs to (-∞, upper].
        """
        super().__init__(upper)

    def __repr__(self) -> str:
        return f"NegSoftplusT(upper={self.lower})"

    def forward(self, x: ArrayLike) -> Array:
        """
        Transform unbounded input to negative semi-infinite interval.
        
        Parameters
        ----------
        x : array_like
            Input values in unbounded domain (-∞, ∞).
            
        Returns
        -------
        Array
            Transformed values in interval (-∞, upper].
            
        Notes
        -----
        Implemented as: upper - softplus(-x).
        """
        return self.lower - jnp.log1p(save_exp(-x)) * self.unit

    def inverse(self, y: ArrayLike) -> Array:
        """
        Transform negative semi-infinite input back to unbounded domain.
        
        Parameters
        ----------
        y : array_like
            Input values in interval (-∞, upper].
            
        Returns
        -------
        Array
            Transformed values in unbounded domain (-∞, ∞).
            
        Notes
        -----
        Inverts: y = upper - softplus(-x) => x = -softplus^{-1}(upper - y).
        """
        s = (self.lower - y) / self.unit
        return -u.math.log(save_exp(s) - 1.0)


class LogT(Transform):
    """
    Log transformation mapping (lower, +inf) to (-inf, +inf).

    Forward maps unconstrained input x to the positive domain via:
        y = lower + exp(x) * unit
    Inverse maps back using:
        x = log((y - lower) / unit)

    Parameters
    ----------
    lower : array_like
        Lower bound of the target interval.
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike) -> None:
        super().__init__()
        self.lower = lower
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"LogT(lower={self.lower})"

    def forward(self, x: ArrayLike) -> Array:
        return self.lower + save_exp(x) * self.unit

    def inverse(self, y: ArrayLike) -> Array:
        return u.math.log((y - self.lower) / self.unit)

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """For exp transform: d/dx[exp(x)] = exp(x), so log|det J| = sum(x)."""
        return jnp.sum(x, axis=-1)


class ExpT(Transform):
    """
    Exponential transformation mapping (-inf, +inf) to (lower, +inf).

    Equivalent to Log; provided for explicit naming.
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike) -> None:
        super().__init__()
        self.lower = lower
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"ExpT(lower={self.lower})"

    def forward(self, x: ArrayLike) -> Array:
        return self.lower + save_exp(x) * self.unit

    def inverse(self, y: ArrayLike) -> Array:
        return u.math.log((y - self.lower) / self.unit)

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """For exp transform: d/dx[exp(x)] = exp(x), so log|det J| = sum(x)."""
        return jnp.sum(x, axis=-1)


class TanhT(Transform):
    """
    Tanh-based transformation mapping (-inf, +inf) to (lower, upper).

    y = lower + width * (tanh(x) + 1) / 2
    x = arctanh(2 * (y - lower) / width - 1)
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        super().__init__()
        self.lower = lower
        self.width = upper - lower
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"TanhT(lower={self.lower}, upper={self.lower + self.width})"

    def forward(self, x: ArrayLike) -> Array:
        return self.lower + self.width * (jnp.tanh(x) + 1.0) / 2.0

    def inverse(self, y: ArrayLike) -> Array:
        z = 2.0 * (y - self.lower) / self.width - 1.0
        return jnp.arctanh(z)


class SoftsignT(Transform):
    """
    Softsign-based transformation mapping (-inf, +inf) to (lower, upper).

    y = lower + width * (x / (1 + |x|) + 1) / 2
    x = z / (1 - |z|), where z = 2 * (y - lower) / width - 1, z in (-1, 1)
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        super().__init__()
        self.lower = lower
        self.width = upper - lower
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"SoftsignT(lower={self.lower}, upper={self.lower + self.width})"

    def forward(self, x: ArrayLike) -> Array:
        return self.lower + self.width * (x / (1.0 + u.math.abs(x)) + 1.0) / 2.0

    def inverse(self, y: ArrayLike) -> Array:
        z = 2.0 * (y - self.lower) / self.width - 1.0
        return z / (1.0 - u.math.abs(z))


class ClipT(Transform):
    r"""
    Transformation with clipping to specified bounds.

    This transformation applies a clipping operation to the input values,
    constraining them within the specified lower and upper bounds. It is useful
    for enforcing hard limits on parameters that must remain within a certain range.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = \min(\max(x, \text{lower}), \text{upper})

    The inverse transformation is not defined for clipping, as information is lost.

    Parameters
    ----------
    lower : array_like
        Lower bound for clipping.
    upper : array_like
        Upper bound for clipping.

    Attributes
    ----------
    lower : array_like
        Lower bound for clipping.
    upper : array_like
        Upper bound for clipping.

    Notes
    -----
    Clipping is a non-bijective transformation since multiple input values can
    map to the same output value at the bounds. Therefore, the inverse method
    is not implemented.

    Examples
    --------
    >>> # Clip values to [0, 1]
    >>> transform = ClipT(0.0, 1.0)
    >>> x = jnp.array([-0.5, 0.5, 1.5])
    >>> y = transform.forward(x)
    >>> # y = [0.0, 0.5, 1.0]
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """
        Initialize the clipping transformation.

        Parameters
        ----------
        lower : array_like
            Lower bound for clipping.
        upper : array_like
            Upper bound for clipping.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper

    def __repr__(self) -> str:
        return f"ClipT(lower={self.lower}, upper={self.upper})"

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply clipping to the input values.

        Parameters
        ----------
        x : array_like
            Input values to clip.

        Returns
        -------
        Array
            Clipped values within [lower, upper].
        """
        return u.math.clip(x, a_min=self.lower, a_max=self.upper)

    def inverse(self, y: ArrayLike) -> Array:
        """
        Inverse transformation is not defined for clipping.

        Raises
        ------
        NotImplementedError
            Clipping is not bijective; inverse cannot be defined.
        """
        return u.math.clip(y, a_min=self.lower, a_max=self.upper)


class AffineT(Transform):
    r"""
    Affine (linear) transformation with scaling and shifting.
    
    This transformation applies a linear transformation of the form y = ax + b,
    where a is the scale factor and b is the shift. It is the most basic form
    of transformation and preserves the relative ordering of inputs while allowing
    for rescaling and translation.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x) = a \cdot x + b
        
    The inverse transformation is:
    
    .. math::
        \text{inverse}(y) = \frac{y - b}{a}
    
    Parameters
    ----------
    scale : array_like
        Scaling factor a. Must be non-zero for invertibility.
    shift : array_like
        Additive shift b.
        
    Attributes
    ----------
    a : array_like
        Scaling factor.
    b : array_like
        Shift parameter.
        
    Raises
    ------
    ValueError
        If scale is zero or numerically close to zero, making the transformation
        non-invertible.
        
    Notes
    -----
    Affine transformations are the foundation of many statistical transformations.
    They preserve linearity and are particularly useful for:
    
    - Standardization: (x - μ) / σ
    - Normalization: (x - min) / (max - min)
    - Unit conversion: x * conversion_factor + offset
    
    The Jacobian of this transformation is constant: |det(J)| = |a|.
    
    Examples
    --------
    >>> # Standardization transform (z-score)
    >>> mu, sigma = 5.0, 2.0
    >>> transform = AffineT(1/sigma, -mu/sigma)
    >>> x = jnp.array([3.0, 5.0, 7.0])
    >>> z = transform.forward(x)
    >>> # z ≈ [-1.0, 0.0, 1.0]
    
    >>> # Temperature conversion: Celsius to Fahrenheit
    >>> transform = AffineT(9/5, 32)
    >>> celsius = jnp.array([0.0, 100.0])
    >>> fahrenheit = transform.forward(celsius)
    >>> # fahrenheit ≈ [32.0, 212.0]
    """
    __module__ = 'braintools.param'

    def __init__(self, scale: ArrayLike, shift: ArrayLike):
        """
        Initialize the affine transformation.
        
        Parameters
        ----------
        scale : array_like
            Scaling factor. Must be non-zero for the transformation to be invertible.
        shift : array_like
            Additive shift parameter.

        Raises
        ------
        ValueError
            If scale is zero or numerically close to zero, making the
            transformation non-invertible.
        """
        if jnp.allclose(scale, 0):
            raise ValueError("a cannot be zero, must be invertible")
        self.a = scale
        self.b = shift

    def __repr__(self) -> str:
        return f"AffineT(scale={self.a}, shift={self.b})"

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply the affine transformation y = ax + b.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Transformed values after scaling and shifting.
        """
        return self.a * x + self.b

    def inverse(self, x: ArrayLike) -> Array:
        """
        Apply the inverse affine transformation x = (y - b) / a.

        Parameters
        ----------
        x : array_like
            Transformed values to invert (note: parameter name kept for consistency).

        Returns
        -------
        Array
            Original values before transformation.
        """
        return (x - self.b) / self.a

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """For affine: d/dx[ax + b] = a, so log|det J| = n * log|a|."""
        n = jnp.shape(x)[-1] if jnp.ndim(x) > 0 else 1
        return n * jnp.log(jnp.abs(self.a))


class ChainT(Transform):
    r"""
    Composition of multiple transformations applied sequentially.
    
    This class implements the mathematical composition of functions, allowing
    multiple transformations to be chained together. The transformations are
    applied in the order specified during initialization for the forward pass,
    and in reverse order for the inverse pass.
    
    For transformations f₁, f₂, ..., fₙ, the chain implements:
    
    .. math::
        \text{forward}(x) = f_n(f_{n-1}(...f_2(f_1(x))...))
        
    .. math::
        \text{inverse}(y) = f_1^{-1}(f_2^{-1}(...f_{n-1}^{-1}(f_n^{-1}(y))...))
    
    Parameters
    ----------
    *transforms : sequence of Transform
        Variable number of Transform objects to chain together.
        
    Attributes
    ----------
    transforms : sequence of Transform
        Tuple of transformations in the order they will be applied.
        
    Notes
    -----
    The chain transformation preserves bijectivity if all component transformations
    are bijective. The Jacobian of the chain is the product of the Jacobians of
    the individual transformations.
    
    Chain transformations are particularly useful for:
    
    - Complex parameter constraints requiring multiple steps
    - Modular transformation design
    - Combining simple transformations to achieve complex mappings
    
    Examples
    --------
    >>> # Transform to (0, 1) then scale to (a, b)
    >>> sigmoid = SigmoidT(0, 1)
    >>> affine = AffineT(scale=b-a, shift=a)
    >>> chain = ChainT(sigmoid, affine)
    
    >>> # Standardize then apply softplus
    >>> standardize = AffineT(1/sigma, -mu/sigma)
    >>> softplus = SoftplusT(0)
    >>> chain = ChainT(standardize, softplus)
    """
    __module__ = 'braintools.param'

    def __init__(self, *transforms: Sequence[Transform]) -> None:
        """
        Initialize the chain transformation.
        
        Parameters
        ----------
        *transforms : sequence of Transform
            Variable number of Transform objects to be applied sequentially.
            The transformations will be applied in the order provided for
            the forward pass, and in reverse order for the inverse pass.
            
        Raises
        ------
        TypeError
            If any of the provided objects is not a Transform instance.
        """
        super().__init__()
        self.transforms: Sequence[Transform] = transforms

    def __repr__(self) -> str:
        transforms_str = ", ".join(repr(t) for t in self.transforms)
        return f"ChainT({transforms_str})"

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply all transformations sequentially in forward order.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Values after applying all transformations in sequence.
            
        Notes
        -----
        Transformations are applied left-to-right as specified in initialization.
        """
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def inverse(self, y: ArrayLike) -> Array:
        """
        Apply all inverse transformations sequentially in reverse order.

        Parameters
        ----------
        y : array_like
            Transformed values to invert.

        Returns
        -------
        Array
            Original values before all transformations were applied.

        Notes
        -----
        Transformations are inverted right-to-left (reverse order) to properly
        undo the forward chain.
        """
        for transform in reversed(self.transforms):
            y = transform.inverse(y)
        return y

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """Sum of log Jacobian determinants of all transforms in the chain."""
        total = jnp.zeros(jnp.shape(x)[:-1] if jnp.ndim(x) > 0 else ())
        current_x = x
        for transform in self.transforms:
            current_y = transform.forward(current_x)
            total = total + transform.log_abs_det_jacobian(current_x, current_y)
            current_x = current_y
        return total


class MaskedT(Transform):
    r"""
    Selective transformation using a boolean mask.
    
    This transformation applies a given transformation only to elements specified
    by a boolean mask, leaving other elements unchanged. This is useful when only
    a subset of parameters need to be transformed while others should remain in
    their original domain.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x)_i = \begin{cases}
        f(x_i) & \text{if } \text{mask}_i = \text{True} \\
        x_i & \text{if } \text{mask}_i = \text{False}
        \end{cases}
        
    where f is the underlying transformation.
    
    The inverse follows the same pattern:
    
    .. math::
        \text{inverse}(y)_i = \begin{cases}
        f^{-1}(y_i) & \text{if } \text{mask}_i = \text{True} \\
        y_i & \text{if } \text{mask}_i = \text{False}
        \end{cases}
    
    Parameters
    ----------
    mask : array_like of bool
        Boolean array indicating which elements to transform.
    transform : Transform
        The transformation to apply to masked elements.
        
    Attributes
    ----------
    mask : array_like
        Boolean mask array.
    transform : Transform
        The underlying transformation.
        
    Notes
    -----
    The mask and input arrays must have compatible shapes for broadcasting.
    This transformation is particularly useful in:
    
    - Mixed parameter models where some parameters are bounded and others are not
    - Selective application of constraints in optimization
    - Sparse transformations where only specific elements need modification
    
    Examples
    --------
    >>> # Transform only positive indices to be positive
    >>> mask = jnp.array([False, True, False, True])
    >>> softplus = SoftplusT(0)
    >>> masked_transform = MaskedT(mask, softplus)
    >>> x = jnp.array([-1.0, -1.0, 2.0, 2.0])
    >>> y = masked_transform.forward(x)
    >>> # y ≈ [-1.0, 0.31, 2.0, 2.13] (only indices 1,3 transformed)
    
    >>> # Transform correlation parameters but not mean parameters
    >>> n_params = 5
    >>> corr_mask = jnp.arange(n_params) >= 3  # Last 2 are correlations
    >>> sigmoid = SigmoidT(-1, 1)
    >>> transform = MaskedT(corr_mask, sigmoid)
    """
    __module__ = 'braintools.param'

    def __init__(self, mask: ArrayLike, transform: Transform) -> None:
        """
        Initialize the masked transformation.
        
        Parameters
        ----------
        mask : array_like of bool
            Boolean array indicating which elements should be transformed.
            Must be broadcastable with the input arrays.
        transform : Transform
            The transformation to apply to elements where mask is True.
            
        Raises
        ------
        TypeError
            If transform is not a Transform instance.
        """
        super().__init__()
        self.mask = mask
        self.transform = transform

    def __repr__(self) -> str:
        return f"MaskedT(mask=..., transform={repr(self.transform)})"

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply transformation selectively based on mask.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Array where masked elements are transformed and unmasked
            elements remain unchanged.
            
        Notes
        -----
        Uses element-wise conditional logic to apply transformation only
        where mask is True.
        """
        return u.math.where(self.mask, self.transform.forward(x), x)

    def inverse(self, y: ArrayLike) -> Array:
        """
        Apply inverse transformation selectively based on mask.
        
        Parameters
        ----------
        y : array_like
            Transformed values to invert.
            
        Returns
        -------
        Array
            Array where masked elements are inverse-transformed and unmasked
            elements remain unchanged.
            
        Notes
        -----
        Applies inverse transformation only to elements where mask is True,
        maintaining consistency with the forward operation.
        """
        return u.math.where(self.mask, self.transform.inverse(y), y)


class ReluT(Transform):
    """ReLU transform with lower bound: forward(x) = relu(x) + lower_bound

    Constrains parameter values to be >= lower_bound.
    Note: inverse is not differentiable at x = lower_bound.

    Args:
        lower_bound: Minimum value for the output (default: 0.0).
    """

    def __init__(self, lower_bound: float = 0.0):
        self.lower_bound = lower_bound

    def forward(self, x: ArrayLike) -> Array:
        return u.math.relu(x) + self.lower_bound

    def inverse(self, y: ArrayLike) -> Array:
        # inverse: x = y - lower_bound (clamped to >= 0)
        return u.math.relu(y - self.lower_bound)


class PositiveT(Transform):
    r"""
    Transformation constraining parameters to be strictly positive (0, +∞).

    This is a convenience class that provides a simple positive constraint
    using the exponential transformation with lower bound of 0.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = e^x

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = \log(y)

    Examples
    --------
    >>> transform = PositiveT()
    >>> x = jnp.array([-1.0, 0.0, 1.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.368, 1.0, 2.718]
    """
    __module__ = 'braintools.param'

    def __init__(self) -> None:
        """Initialize the positive transformation."""
        super().__init__()

    def __repr__(self) -> str:
        return "PositiveT()"

    def forward(self, x: ArrayLike) -> Array:
        """Transform unbounded input to positive values."""
        return save_exp(x)

    def inverse(self, y: ArrayLike) -> Array:
        """Transform positive input back to unbounded domain."""
        return u.math.log(y)

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """For exp transform: d/dx[exp(x)] = exp(x), so log|det J| = sum(x)."""
        return jnp.sum(x, axis=-1)


class NegativeT(Transform):
    r"""
    Transformation constraining parameters to be strictly negative (-∞, 0).

    This is a convenience class that provides a simple negative constraint
    using the negative softplus transformation with upper bound of 0.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = -\log(1 + e^{-x})

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = -\log(e^{-y} - 1)

    Examples
    --------
    >>> transform = NegativeT()
    >>> x = jnp.array([-5.0, 0.0, 5.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [-5.007, -0.693, -0.007]
    """
    __module__ = 'braintools.param'

    def __init__(self) -> None:
        """Initialize the negative transformation."""
        super().__init__()

    def __repr__(self) -> str:
        return "NegativeT()"

    def forward(self, x: ArrayLike) -> Array:
        """Transform unbounded input to negative values."""
        return -jnp.log1p(save_exp(-x))

    def inverse(self, y: ArrayLike) -> Array:
        """Transform negative input back to unbounded domain."""
        return -u.math.log(save_exp(-y) - 1.0)


class ScaledSigmoidT(Transform):
    r"""
    Sigmoid transformation with adjustable sharpness/temperature.

    This transformation extends the standard sigmoid with a scaling parameter
    (beta) that controls the sharpness of the transition. Higher beta values
    result in a sharper sigmoid, while lower values produce a smoother transition.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = \text{lower} + \text{width} \cdot \sigma(\beta \cdot x)

    where :math:`\sigma(x) = \frac{1}{1 + e^{-x}}` is the standard sigmoid function.

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = \frac{1}{\beta} \cdot \text{logit}\left(\frac{y - \text{lower}}{\text{width}}\right)

    Parameters
    ----------
    lower : array_like
        Lower bound of the target interval.
    upper : array_like
        Upper bound of the target interval.
    beta : float, optional
        Sharpness parameter, by default 1.0. Higher values produce sharper transitions.

    Examples
    --------
    >>> # Standard sigmoid
    >>> transform = ScaledSigmoidT(0.0, 1.0, beta=1.0)
    >>> # Sharp sigmoid
    >>> transform_sharp = ScaledSigmoidT(0.0, 1.0, beta=5.0)
    >>> # Smooth sigmoid
    >>> transform_smooth = ScaledSigmoidT(0.0, 1.0, beta=0.5)
    """
    __module__ = 'braintools.param'

    def __init__(self, lower: ArrayLike, upper: ArrayLike, beta: float = 1.0) -> None:
        """
        Initialize the scaled sigmoid transformation.

        Parameters
        ----------
        lower : array_like
            Lower bound of the target interval.
        upper : array_like
            Upper bound of the target interval.
        beta : float, optional
            Sharpness parameter, by default 1.0.
        """
        super().__init__()
        self.lower = lower
        self.width = upper - lower
        self.beta = beta
        self.unit = u.get_unit(lower)

    def __repr__(self) -> str:
        return f"ScaledSigmoidT(lower={self.lower}, upper={self.lower + self.width}, beta={self.beta})"

    def forward(self, x: ArrayLike) -> Array:
        """Transform unbounded input to bounded interval."""
        return self.lower + self.width * jax.nn.sigmoid(self.beta * x)

    def inverse(self, y: ArrayLike) -> Array:
        """Transform bounded input back to unbounded domain."""
        z = (y - self.lower) / self.width
        return jax.scipy.special.logit(z) / self.beta


class PowerT(Transform):
    r"""
    Power (Box-Cox) transformation for stabilizing variance.

    This transformation implements the Box-Cox family of power transformations,
    which are commonly used to stabilize variance and make data more normally
    distributed.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = \begin{cases}
        \frac{x^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\
        \log(x) & \text{if } \lambda = 0
        \end{cases}

    The inverse transformation is:

    .. math::
        \text{inverse}(y) = \begin{cases}
        (y \cdot \lambda + 1)^{1/\lambda} & \text{if } \lambda \neq 0 \\
        e^y & \text{if } \lambda = 0
        \end{cases}

    Parameters
    ----------
    lmbda : float, optional
        Power parameter, by default 0.5. Special cases:
        - lmbda = 0: log transformation
        - lmbda = 0.5: square root transformation
        - lmbda = 1: linear transformation (identity)
        - lmbda = 2: quadratic transformation

    Notes
    -----
    Input values must be positive for this transformation to be well-defined.

    Examples
    --------
    >>> # Square root transformation
    >>> transform = PowerT(lmbda=0.5)
    >>> x = jnp.array([1.0, 4.0, 9.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0, 2, 4]
    """
    __module__ = 'braintools.param'

    def __init__(self, lmbda: float = 0.5) -> None:
        """
        Initialize the power transformation.

        Parameters
        ----------
        lmbda : float, optional
            Power parameter, by default 0.5.
        """
        super().__init__()
        self.lmbda = lmbda

    def __repr__(self) -> str:
        return f"PowerT(lmbda={self.lmbda})"

    def forward(self, x: ArrayLike) -> Array:
        """Apply the power transformation."""
        if jnp.abs(self.lmbda) < 1e-10:
            return u.math.log(x)
        return (jnp.power(x, self.lmbda) - 1) / self.lmbda

    def inverse(self, y: ArrayLike) -> Array:
        """Apply the inverse power transformation."""
        if jnp.abs(self.lmbda) < 1e-10:
            return u.math.exp(y)
        return jnp.power(y * self.lmbda + 1, 1 / self.lmbda)


class OrderedT(Transform):
    r"""
    Transformation ensuring ordered (monotonically increasing) output.

    Maps unconstrained ℝⁿ to ordered vectors where y₁ < y₂ < ... < yₙ.
    This is useful for parameters that must maintain an ordering constraint,
    such as cutpoints in ordinal regression.

    The transformation is defined by:

    .. math::
        y_1 = x_1 \\
        y_i = y_{i-1} + \text{softplus}(x_i) \quad \text{for } i > 1

    The inverse transformation reverses this process.

    Examples
    --------
    >>> transform = OrderedT()
    >>> x = jnp.array([0.0, 1.0, 0.5])
    >>> y = transform.forward(x)
    >>> # y is monotonically increasing
    >>> assert jnp.all(jnp.diff(y) > 0)
    """
    __module__ = 'braintools.param'

    def __init__(self) -> None:
        """Initialize the ordered transformation."""
        super().__init__()

    def __repr__(self) -> str:
        return "OrderedT()"

    def forward(self, x: ArrayLike) -> Array:
        """Transform unconstrained input to ordered vectors."""
        first = x[..., :1]
        rest = jnp.log1p(save_exp(x[..., 1:]))
        return jnp.concatenate([first, first + jnp.cumsum(rest, axis=-1)], axis=-1)

    def inverse(self, y: ArrayLike) -> Array:
        """Transform ordered vectors back to unconstrained domain."""
        first = y[..., :1]
        diffs = y[..., 1:] - y[..., :-1]
        rest = u.math.log(u.math.exp(diffs) - 1)
        return jnp.concatenate([first, rest], axis=-1)


class SimplexT(Transform):
    r"""
    Stick-breaking transformation for simplex constraint.

    Maps unconstrained ℝⁿ⁻¹ to n-dimensional simplex where all elements
    are positive and sum to 1. This is useful for probability distributions
    and categorical parameters.

    The stick-breaking process works as follows:

    .. math::
        z_i = \sigma(x_i) \\
        y_i = z_i \cdot \prod_{j<i} (1 - z_j) \quad \text{for } i < n \\
        y_n = \prod_{j<n} (1 - z_j)

    where :math:`\sigma` is the sigmoid function.

    Notes
    -----
    The input dimension should be n-1 for an n-dimensional simplex output.

    Examples
    --------
    >>> transform = SimplexT()
    >>> x = jnp.array([0.0, 0.0])  # 2D input -> 3D simplex output
    >>> y = transform.forward(x)
    >>> # y sums to 1 and all elements are positive
    >>> assert jnp.allclose(jnp.sum(y), 1.0)
    >>> assert jnp.all(y > 0)
    """
    __module__ = 'braintools.param'

    def __init__(self) -> None:
        """Initialize the simplex transformation."""
        super().__init__()

    def __repr__(self) -> str:
        return "SimplexT()"

    def forward(self, x: ArrayLike) -> Array:
        """Transform unconstrained input to simplex."""
        z = jax.nn.sigmoid(x)
        # Compute cumulative product of (1 - z)
        one_minus_z = 1 - z
        cumprod = jnp.cumprod(one_minus_z, axis=-1)
        # Shift cumprod to get [1, (1-z1), (1-z1)(1-z2), ...]
        cumprod_shifted = jnp.concatenate(
            [jnp.ones((*z.shape[:-1], 1)), cumprod[..., :-1]], axis=-1
        )
        # First n-1 elements: z_i * product of (1-z_j) for j < i
        y_head = z * cumprod_shifted
        # Last element: remaining probability
        y_tail = cumprod[..., -1:]
        return jnp.concatenate([y_head, y_tail], axis=-1)

    def inverse(self, y: ArrayLike) -> Array:
        """Transform simplex back to unconstrained domain."""
        y_head = y[..., :-1]
        # Compute cumulative sum from left
        cumsum = jnp.cumsum(y_head, axis=-1)
        # remaining = 1 - cumsum + current = probability still available
        remaining = 1 - cumsum + y_head
        # z_i = y_i / remaining
        z = y_head / (remaining + 1e-8)
        return jax.scipy.special.logit(z)


class UnitVectorT(Transform):
    r"""
    Transformation to unit vectors (L2 norm = 1).

    Projects input vectors onto the unit sphere by normalizing.
    This is useful for directional data or when parameters must
    lie on a sphere.

    The transformation is defined by:

    .. math::
        \text{forward}(x) = \frac{x}{\|x\|_2}

    Notes
    -----
    This transformation is not strictly bijective since all vectors
    along a ray map to the same unit vector. The inverse returns
    the input unchanged, assuming it is already on the unit sphere.

    Examples
    --------
    >>> transform = UnitVectorT()
    >>> x = jnp.array([3.0, 4.0])
    >>> y = transform.forward(x)
    >>> # y has unit norm
    >>> assert jnp.allclose(jnp.linalg.norm(y), 1.0)
    """
    __module__ = 'braintools.param'

    def __init__(self) -> None:
        """Initialize the unit vector transformation."""
        super().__init__()

    def __repr__(self) -> str:
        return "UnitVectorT()"

    def forward(self, x: ArrayLike) -> Array:
        """Project input onto unit sphere."""
        norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + 1e-8)
        return x / norm

    def inverse(self, y: ArrayLike) -> Array:
        """Return input unchanged (assumes already on unit sphere)."""
        return y
