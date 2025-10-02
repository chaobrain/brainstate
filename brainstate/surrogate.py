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

"""
Surrogate gradient functions for training spiking neural networks.

This module provides a comprehensive collection of surrogate gradient functions
that enable backpropagation through the non-differentiable spike generation
mechanism in spiking neural networks (SNNs). Surrogate gradients are essential
for training SNNs with gradient-based optimization methods.

Overview
--------

Spiking neurons use the Heaviside step function to generate spikes, which has
zero gradient almost everywhere and is undefined at zero. This makes direct
backpropagation impossible. Surrogate gradient methods solve this by:

1. Using the actual step function in the forward pass (maintaining binary spikes)
2. Replacing it with a smooth approximation in the backward pass (enabling gradients)

This straight-through estimator approach allows SNNs to be trained with standard
deep learning frameworks while preserving their event-driven, binary nature.

Available Surrogate Functions
------------------------------

The module provides various surrogate gradient functions, each with different
mathematical properties and use cases:

**Sigmoid-based:**

  - :class:`Sigmoid` - Smooth sigmoid-shaped gradient
  - :class:`SoftSign` - Soft sign approximation
  - :class:`Arctan` - Arctangent-based gradient
  - :class:`ERF` - Error function gradient

**Piecewise functions:**

  - :class:`PiecewiseQuadratic` - Quadratic approximation
  - :class:`PiecewiseExp` - Exponential segments
  - :class:`PiecewiseLeakyRelu` - Leaky ReLU segments

**ReLU-based:**

  - :class:`ReluGrad` - Triangular gradient with finite support
  - :class:`LeakyRelu` - Simple piecewise linear
  - :class:`LogTailedRelu` - ReLU with logarithmic tail

**Distribution-inspired:**

  - :class:`GaussianGrad` - Gaussian/normal distribution
  - :class:`MultiGaussianGrad` - Multiple Gaussian components
  - :class:`InvSquareGrad` - Lorentzian/Cauchy-like
  - :class:`SlayerGrad` - Exponential/Laplace-like

**Advanced functions:**

  - :class:`NonzeroSignLog` - Non-zero sign with log
  - :class:`SquarewaveFourierSeries` - Fourier series approximation
  - :class:`S2NN` - Asymmetric gradient for single-step networks
  - :class:`QPseudoSpike` - q-PseudoSpike with tunable tails

Usage Patterns
--------------

There are two ways to use surrogate gradients:

**Class-based (recommended):**

.. code-block:: python

    >>> import brainstate.surrogate as surrogate
    >>> import jax.numpy as jnp
    >>>
    >>> # Create surrogate function instance
    >>> sg_fn = surrogate.Sigmoid(alpha=4.0)
    >>>
    >>> # Apply to membrane potentials
    >>> v_mem = jnp.array([-0.5, 0.0, 0.5])
    >>> spikes = sg_fn(v_mem)  # Forward: step function
    >>> print(spikes)
    [0. 1. 1.]

**Function-based (simplified API):**

.. code-block:: python

    >>> import brainstate.surrogate as surrogate
    >>>
    >>> # Direct function call
    >>> spikes = surrogate.sigmoid(v_mem, alpha=4.0)

Choosing a Surrogate Function
------------------------------

Different surrogate functions have different trade-offs:

- **Sigmoid/SoftSign**: Smooth gradients, good for general use
- **Piecewise functions**: Fast computation, adjustable complexity
- **Gaussian**: Biologically plausible, smooth with all derivatives
- **ReLU-based**: Sparse gradients, computational efficiency
- **Exponential/SLAYER**: Good gradient flow, proven in large networks

Selection criteria:
1. **Gradient flow**: Functions with heavier tails (InvSquareGrad, SlayerGrad)
   maintain gradients far from threshold
2. **Sharpness**: Higher α/steeper gradients for more precise spike timing
3. **Computational cost**: Piecewise and ReLU variants are fastest
4. **Biological plausibility**: Gaussian-based are most biologically motivated

Custom Surrogate Functions
--------------------------

Create custom surrogate gradients by inheriting from :class:`Surrogate`:

.. code-block:: python

    >>> import brainstate
    >>> import jax.numpy as jnp
    >>>
    >>> class CustomSurrogate(brainstate.surrogate.Surrogate):
    ...     def __init__(self, param=1.0):
    ...         super().__init__()
    ...         self.param = param
    ...
    ...     def surrogate_fun(self, x):
    ...         # Define smooth approximation
    ...         return jax.nn.sigmoid(x * self.param)
    ...
    ...     def surrogate_grad(self, x):
    ...         # Define gradient for backprop
    ...         s = jax.nn.sigmoid(x * self.param)
    ...         return self.param * s * (1 - s)

Examples
--------

**Training a simple SNN layer:**

.. code-block:: python

    >>> import brainstate
    >>> import brainstate.surrogate as surrogate
    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> # Define a simple LIF neuron with surrogate gradient
    >>> class LIFNeuron(brainstate.Module):
    ...     def __init__(self, n_neurons, tau=20.0):
    ...         super().__init__()
    ...         self.tau = tau
    ...         self.sg_fn = surrogate.Sigmoid(alpha=4.0)
    ...         self.v = brainstate.State(jnp.zeros(n_neurons))
    ...
    ...     def update(self, x):
    ...         # Membrane dynamics
    ...         self.v.value = self.v.value * (1 - 1/self.tau) + x
    ...         # Spike generation with surrogate gradient
    ...         spikes = self.sg_fn(self.v.value - 1.0)  # threshold = 1.0
    ...         # Reset
    ...         self.v.value = self.v.value * (1 - spikes)
    ...         return spikes

**Comparing different surrogate gradients:**

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>>
    >>> x = jnp.linspace(-3, 3, 1000)
    >>>
    >>> # Compare different surrogate gradients
    >>> surrogates = {
    ...     'Sigmoid': surrogate.Sigmoid(alpha=4.0),
    ...     'Gaussian': surrogate.GaussianGrad(sigma=0.5, alpha=1.0),
    ...     'ReLU': surrogate.ReluGrad(alpha=1.0, width=1.0),
    ...     'SLAYER': surrogate.SlayerGrad(alpha=1.0)
    ... }
    >>>
    >>> for name, sg_fn in surrogates.items():
    ...     grads = jax.vmap(jax.grad(sg_fn))(x)
    ...     plt.plot(x, grads, label=name)
    >>>
    >>> plt.xlabel('Membrane Potential')
    >>> plt.ylabel('Surrogate Gradient')
    >>> plt.legend()
    >>> plt.title('Comparison of Surrogate Gradients')
    >>> plt.show()

References
----------

.. [1] Neftci, E. O., Mostafa, H. & Zenke, F. Surrogate gradient learning in
       spiking neural networks. IEEE Signal Process. Mag. 36, 61–63 (2019).

.. [2] Shrestha, S. B. & Orchard, G. SLAYER: spike layer error reassignment
       in time. NeurIPS 2018.

.. [3] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain
       classification with adaptive spiking recurrent neural networks.
       Nat Mach Intell 3, 905–913 (2021).

See Also
--------
brainstate.nn : Neural network components and layers
brainstate.neurons : Spiking neuron models
brainstate.synapse : Synaptic models and connections

"""

import warnings

import jax
import jax.numpy as jnp
import jax.scipy as sci
from jax.interpreters import batching, ad, mlir

from brainstate._compatible_import import Primitive
from brainstate.util import PrettyObject

__all__ = [
    'Surrogate',
    'Sigmoid',
    'sigmoid',
    'PiecewiseQuadratic',
    'piecewise_quadratic',
    'PiecewiseExp',
    'piecewise_exp',
    'SoftSign',
    'soft_sign',
    'Arctan',
    'arctan',
    'NonzeroSignLog',
    'nonzero_sign_log',
    'ERF',
    'erf',
    'PiecewiseLeakyRelu',
    'piecewise_leaky_relu',
    'SquarewaveFourierSeries',
    'squarewave_fourier_series',
    'S2NN',
    's2nn',
    'QPseudoSpike',
    'q_pseudo_spike',
    'LeakyRelu',
    'leaky_relu',
    'LogTailedRelu',
    'log_tailed_relu',
    'ReluGrad',
    'relu_grad',
    'GaussianGrad',
    'gaussian_grad',
    'InvSquareGrad',
    'inv_square_grad',
    'MultiGaussianGrad',
    'multi_gaussian_grad',
    'SlayerGrad',
    'slayer_grad',
]


def _heaviside_abstract(x, dx):
    return [x]


def _heaviside_imp(x, dx):
    z = jnp.asarray(x >= 0, dtype=x.dtype)
    return [z]


def _heaviside_batching(args, axes):
    x, dx = args
    if axes[0] != axes[1]:
        dx = batching.moveaxis(dx, axes[1], axes[0])
    return heaviside_p.bind(x, dx), tuple([axes[0]])


def _heaviside_jvp(primals, tangents):
    x, dx = primals
    tx, tdx = tangents
    primal_outs = heaviside_p.bind(x, dx)
    tangent_outs = [dx * tx, ]
    return primal_outs, tangent_outs


heaviside_p = Primitive('heaviside_p')
heaviside_p.multiple_results = True
heaviside_p.def_abstract_eval(_heaviside_abstract)
heaviside_p.def_impl(_heaviside_imp)
batching.primitive_batchers[heaviside_p] = _heaviside_batching
ad.primitive_jvps[heaviside_p] = _heaviside_jvp
mlir.register_lowering(heaviside_p, mlir.lower_fun(_heaviside_imp, multiple_results=True))


class Surrogate(PrettyObject):
    """The base surrogate gradient function.

    This abstract base class defines the interface for surrogate gradient functions
    used in training spiking neural networks. Surrogate gradients replace the
    non-differentiable spike function with smooth approximations during backpropagation.

    To customize a surrogate gradient function, inherit this class and implement
    the `surrogate_fun` and `surrogate_grad` methods.

    Methods
    -------
    surrogate_fun(x)
        Defines the smooth surrogate function used for visualization and analysis.
    surrogate_grad(x)
        Defines the gradient of the surrogate function used during backpropagation.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a custom surrogate gradient function
        >>> class MySurrogate(brainstate.surrogate.Surrogate):
        ...     def __init__(self, alpha=1.):
        ...         super().__init__()
        ...         self.alpha = alpha
        ...
        ...     def surrogate_fun(self, x):
        ...         # Define the smooth approximation function
        ...         return jnp.tanh(x * self.alpha) * 0.5 + 0.5
        ...
        ...     def surrogate_grad(self, x):
        ...         # Define its gradient for backpropagation
        ...         return self.alpha * 0.5 * (1 - jnp.tanh(x * self.alpha) ** 2)
        >>>
        >>> # Use the custom surrogate
        >>> my_surrogate = MySurrogate(alpha=2.0)
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = my_surrogate(x)  # Forward: step function
        >>> print(spikes)  # [0., 1., 1.]

    Notes
    -----
    The forward pass always returns a Heaviside step function (0 or 1),
    while the backward pass uses the custom surrogate gradient defined in
    `surrogate_grad` method. This straight-through estimator approach enables
    gradient-based training of spiking neural networks.

    """
    __module__ = 'brainstate.surrogate'

    def __call__(self, x):
        dx = self.surrogate_grad(x)
        return heaviside_p.bind(x, dx)[0]

    def surrogate_fun(self, x) -> jax.Array:
        """The surrogate function."""
        raise NotImplementedError

    def surrogate_grad(self, x) -> jax.Array:
        """The gradient function of the surrogate function."""
        raise NotImplementedError


class Sigmoid(Surrogate):
    """Spike function with the sigmoid-shaped surrogate gradient.

    This class implements a spiking neuron activation with a sigmoid-shaped
    surrogate gradient for backpropagation. It can be used in spiking neural
    networks to approximate the non-differentiable step function during training.

    Parameters
    ----------
    alpha : float, optional
        A parameter controlling the steepness of the sigmoid curve in the
        surrogate gradient. Higher values make the transition sharper.
        Default is 4.0.

    See Also
    --------
    sigmoid : Deprecated function version of this class. Use Sigmoid class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a Sigmoid surrogate gradient function
        >>> sigmoid = brainstate.surrogate.Sigmoid(alpha=4.0)
        >>>
        >>> # Apply to input data
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = sigmoid(x)
        >>> print(spikes)  # Step function output: [0., 1., 1.]
        >>>
        >>> # Use in a spiking neural network layer
        >>> import brainstate.nn as nn
        >>>
        >>> class SpikingLayer(nn.Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.linear = nn.Linear(in_features, out_features)
        ...         self.spike_fn = brainstate.surrogate.Sigmoid(alpha=4.0)
        ...
        ...     def forward(self, x):
        ...         membrane = self.linear(x)
        ...         return self.spike_fn(membrane)

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate.nn as nn
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-2, 2, 1000)
       >>> for alpha in [1., 2., 4.]:
       >>>   sigmoid = brainstate.surrogate.Sigmoid(alpha=alpha)
       >>>   grads = brainstate.augment.vector_grad(sigmoid)(xs)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Notes
    -----
    The forward pass uses a Heaviside step function (1 for x >= 0, 0 for x < 0),
    while the backward pass uses a sigmoid-shaped surrogate gradient for
    smooth optimization. The surrogate gradient is defined as:

    .. math::
        g'(x) = \\alpha \\cdot (1 - \\sigma(\\alpha x)) \\cdot \\sigma(\\alpha x)

    where :math:`\\sigma` is the sigmoid function.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha: float = 4.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        """Compute the surrogate function.

        Parameters
        ----------
        x : jax.Array
            The input array.

        Returns
        -------
        jax.Array
            The output of the surrogate function.
        """
        return sci.special.expit(self.alpha * x)

    def surrogate_grad(self, x):
        """Compute the gradient of the surrogate function.

        Parameters
        ----------
        x : jax.Array
            The input array.

        Returns
        -------
        jax.Array
            The gradient of the surrogate function.
        """
        sgax = sci.special.expit(x * self.alpha)
        dx = (1. - sgax) * sgax * self.alpha
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def sigmoid(
    x: jax.Array,
    alpha: float = 4.,
):
    """
    Spike function with the sigmoid-shaped surrogate gradient.

    See the documentation of :class:`Sigmoid` for details.
    """
    return Sigmoid(alpha=alpha)(x)


class PiecewiseQuadratic(Surrogate):
    """Judge spiking state with a piecewise quadratic function.

    This class implements a surrogate gradient method using a piecewise quadratic
    function for training spiking neural networks. It provides smooth gradients
    within a defined range around zero.

    Parameters
    ----------
    alpha : float, optional
        A parameter controlling the width and steepness of the surrogate gradient.
        Higher values result in a narrower gradient window. Default is 1.0.

    See Also
    --------
    piecewise_quadratic : Function version of this class.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a piecewise quadratic surrogate gradient function
        >>> pq_fn = brainstate.surrogate.PiecewiseQuadratic(alpha=1.0)
        >>>
        >>> # Apply to membrane potentials
        >>> x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        >>> spikes = pq_fn(x)
        >>> print(spikes)  # Binary spike output: [0., 0., 1., 1., 1.]
        >>>
        >>> # Use in a spiking neural network
        >>> import brainstate.nn as nn
        >>>
        >>> class SpikingNeuron(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.spike_fn = brainstate.surrogate.PiecewiseQuadratic(alpha=2.0)
        ...         self.membrane = 0.0
        ...
        ...     def forward(self, input_current):
        ...         self.membrane += input_current
        ...         spike = self.spike_fn(self.membrane)
        ...         self.membrane = self.membrane * (1 - spike)  # Reset on spike
        ...         return spike

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   pq_fn = brainstate.surrogate.PiecewiseQuadratic(alpha=alpha)
       >>>   grads = brainstate.augment.vector_grad(pq_fn)(xs)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Notes
    -----
    The forward pass uses a Heaviside step function (1 for x >= 0, 0 for x < 0),
    while the backward pass uses a piecewise quadratic surrogate gradient.

    The surrogate gradient is non-zero only within the range :math:`[-1/\\alpha, 1/\\alpha]`,
    providing localized gradient flow during backpropagation. This helps prevent
    gradient explosion and vanishing gradients in deep spiking networks.

    The surrogate gradient is defined as:

    .. math::
        g'(x) = \\begin{cases}
        0, & |x| > \\frac{1}{\\alpha} \\\\
        -\\alpha^2|x| + \\alpha, & |x| \\leq \\frac{1}{\\alpha}
        \\end{cases}

    References
    ----------
    .. [1] Esser S K, Merolla P A, Arthur J V, et al. Convolutional networks for fast, energy-efficient neuromorphic computing[J]. Proceedings of the national academy of sciences, 2016, 113(41): 11441-11446.
    .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
    .. [3] Bellec G, Salaj D, Subramoney A, et al. Long short-term memory and learning-to-learn in networks of spiking neurons[C]//Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018: 795-805.
    .. [4] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.
    .. [5] Panda P, Aketi S A, Roy K. Toward scalable, efficient, and accurate deep spiking neural networks with backward residual connections, stochastic softmax, and hybridization[J]. Frontiers in Neuroscience, 2020, 14.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha: float = 1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        """Compute the piecewise quadratic surrogate function.

        Parameters
        ----------
        x : jax.Array
            Input tensor.

        Returns
        -------
        jax.Array
            Output of the surrogate function.
        """
        z = jnp.where(
            x < -1 / self.alpha,
            0.,
            jnp.where(
                x > 1 / self.alpha,
                1.,
                (-self.alpha * jnp.abs(x) / 2 + 1) * self.alpha * x + 0.5
            )
        )
        return z

    def surrogate_grad(self, x):
        """Compute the gradient of the piecewise quadratic function.

        Parameters
        ----------
        x : jax.Array
            Input tensor.

        Returns
        -------
        jax.Array
            Gradient of the surrogate function.
        """
        dx = jnp.where(jnp.abs(x) > 1 / self.alpha, 0., (-(self.alpha * x) ** 2 + self.alpha))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def piecewise_quadratic(
    x: jax.Array,
    alpha: float = 1.,
):
    """
    Spike function with the piecewise quadratic surrogate gradient.

    See the documentation of :class:`PiecewiseQuadratic` for details.
    """
    return PiecewiseQuadratic(alpha=alpha)(x)


class PiecewiseExp(Surrogate):
    """Judge spiking state with a piecewise exponential function.

    This class implements a surrogate gradient method for spiking neural networks
    using a piecewise exponential function. It provides a differentiable approximation
    of the step function used in the forward pass of spiking neurons.

    Parameters
    ----------
    alpha : float, optional
        A parameter controlling the steepness of the surrogate gradient.
        Higher values result in a steeper gradient. Default is 1.0.

    See Also
    --------
    piecewise_exp : Deprecated function version. Use PiecewiseExp class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a piecewise exponential surrogate
        >>> pe_fn = brainstate.surrogate.PiecewiseExp(alpha=1.0)
        >>>
        >>> # Apply to membrane potentials
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = pe_fn(x)
        >>> print(spikes)  # [0., 1., 1.]
        >>>
        >>> # Use in a leaky integrate-and-fire neuron
        >>> import brainstate.nn as nn
        >>>
        >>> class LIFNeuron(nn.Module):
        ...     def __init__(self, tau=20.0):
        ...         super().__init__()
        ...         self.tau = tau
        ...         self.spike_fn = brainstate.surrogate.PiecewiseExp(alpha=2.0)
        ...         self.v = 0.0
        ...
        ...     def forward(self, input_current, dt=1.0):
        ...         self.v = self.v + dt/self.tau * (-self.v + input_current)
        ...         spike = self.spike_fn(self.v - 1.0)  # Threshold at 1.0
        ...         self.v = self.v * (1 - spike)  # Reset
        ...         return spike

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   pe_fn = brainstate.surrogate.PiecewiseExp(alpha=alpha)
       >>>   grads = brainstate.augment.vector_grad(pe_fn)(xs)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Notes
    -----
    The forward pass uses a Heaviside step function (1 for x >= 0, 0 for x < 0),
    while the backward pass uses a piecewise exponential surrogate gradient.

    The piecewise exponential function provides smooth gradients that decay
    exponentially with distance from the threshold, which can help with
    gradient flow in deep networks.

    The surrogate gradient is defined as:

    .. math::
        g'(x) = \\frac{\\alpha}{2} e^{-\\alpha |x|}

    References
    ----------
    .. [1] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha: float = 1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        """Compute the surrogate gradient.

        Parameters
        ----------
        x : jax.Array
            The input array.

        Returns
        -------
        jax.Array
            The surrogate gradient.
        """
        dx = (self.alpha / 2) * jnp.exp(-self.alpha * jnp.abs(x))
        return dx

    def surrogate_fun(self, x):
        """Compute the surrogate function.

        Parameters
        ----------
        x : jax.Array
            The input array.

        Returns
        -------
        jax.Array
            The output of the surrogate function.
        """
        return jnp.where(
            x < 0,
            jnp.exp(self.alpha * x) / 2,
            1 - jnp.exp(-self.alpha * x) / 2
        )

    def __repr__(self):
        """Return a string representation of the PiecewiseExp instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        """Compute a hash value for the PiecewiseExp instance.

        Returns
        -------
        int
            A hash value for the instance.
        """
        return hash((self.__class__, self.alpha))


def piecewise_exp(
    x: jax.Array,
    alpha: float = 1.,
):
    """
    Spike function with the piecewise exponential surrogate gradient.

    See the documentation of :class:`PiecewiseExp` for details.
    """
    return PiecewiseExp(alpha=alpha)(x)


class SoftSign(Surrogate):
    """Judge spiking state with a soft sign function.

    This class implements a surrogate gradient using the soft sign function,
    which provides a smooth approximation to the step function.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       g(x) = \frac{1}{2} (\frac{\alpha x}{1 + |\alpha x|} + 1)
              = \frac{1}{2} (\frac{x}{\frac{1}{\alpha} + |x|} + 1)

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{2(1 + |\alpha x|)^{2}} = \frac{1}{2\alpha(\frac{1}{\alpha} + |x|)^{2}}


    Parameters
    ----------
    alpha : float, optional
        Parameter controlling the steepness of the surrogate gradient.
        Higher values make the transition sharper. Default is 1.0.

    See Also
    --------
    soft_sign : function version.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a soft sign surrogate
        >>> ss_fn = brainstate.surrogate.SoftSign(alpha=2.0)
        >>>
        >>> # Apply to input
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = ss_fn(x)
        >>> print(spikes)  # Binary spike output
        >>>
        >>> # Use in a spiking layer with adaptive threshold
        >>> class AdaptiveSpikingLayer(brainstate.nn.Module):
        ...     def __init__(self, n_neurons):
        ...         super().__init__()
        ...         self.n = n_neurons
        ...         self.spike_fn = brainstate.surrogate.SoftSign(alpha=2.0)
        ...         self.threshold = jnp.ones(n_neurons)
        ...
        ...     def forward(self, membrane_potential):
        ...         spikes = self.spike_fn(membrane_potential - self.threshold)
        ...         # Update threshold based on spike history
        ...         self.threshold += 0.01 * spikes
        ...         return spikes

    Notes
    -----
    The soft sign function provides gradients that decay more slowly than
    exponential functions, which can be beneficial for learning in deep networks.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        """Compute the gradient of the soft sign function.

        Parameters
        ----------
        x : jax.Array
            Input tensor.

        Returns
        -------
        jax.Array
            Gradient of the soft sign function.
        """
        dx = self.alpha * 0.5 / (1 + jnp.abs(self.alpha * x)) ** 2
        return dx

    def surrogate_fun(self, x):
        """Compute the soft sign surrogate function.

        Parameters
        ----------
        x : jax.Array
            Input tensor.

        Returns
        -------
        jax.Array
            Output of the soft sign function.
        """
        return x / (2 / self.alpha + 2 * jnp.abs(x)) + 0.5

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def soft_sign(
    x: jax.Array,
    alpha: float = 1.,

):
    """
    Spike function with the soft sign surrogate gradient.

    See the documentation of :class:`SoftSign` for details.
    """
    return SoftSign(alpha=alpha)(x)


class Arctan(Surrogate):
    """Judge spiking state with an arctan function.

    This class implements a surrogate gradient using the arctangent function.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       g(x) = \frac{1}{\pi} \arctan(\frac{\pi}{2}\alpha x) + \frac{1}{2}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{2(1 + (\frac{\pi}{2}\alpha x)^2)}

    Parameters
    ----------
    alpha : float, optional
        Parameter controlling the steepness of the surrogate gradient.
        Higher values make the transition sharper. Default is 1.0.

    See Also
    --------
    arctan : Deprecated function version. Use Arctan class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create an arctangent surrogate
        >>> arctan_fn = brainstate.surrogate.Arctan(alpha=2.0)
        >>>
        >>> # Apply to membrane potentials
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = arctan_fn(x)
        >>> print(spikes)  # Binary spike output

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   arctan_fn = brainstate.surrogate.Arctan(alpha=alpha)
       >>>   grads = brainstate.augment.vector_grad(arctan_fn)(xs)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Notes
    -----
    The arctangent function provides smooth gradients with polynomial decay,
    offering a balance between the fast decay of exponential functions and
    the slow decay of linear functions.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = self.alpha * 0.5 / (1 + (jnp.pi / 2 * self.alpha * x) ** 2)
        return dx

    def surrogate_fun(self, x):
        return jnp.arctan2(jnp.pi / 2 * self.alpha * x) / jnp.pi + 0.5

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def arctan(
    x: jax.Array,
    alpha: float = 1.,
):
    """
    Spike function with the arctangent surrogate gradient.

    See the documentation of :class:`Arctan` for details.
    """
    return Arctan(alpha=alpha)(x)


class NonzeroSignLog(Surrogate):
    """Judge spiking state with a nonzero sign log function.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       g(x) = \mathrm{NonzeroSign}(x) \log (|\alpha x| + 1)

    where

    .. math::

       \begin{split}\mathrm{NonzeroSign}(x) =
        \begin{cases}
        1, & x \geq 0 \\
        -1, & x < 0 \\
        \end{cases}\end{split}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{1 + |\alpha x|} = \frac{1}{\frac{1}{\alpha} + |x|}

    This surrogate function has the advantage of low computation cost during the backward.

    Parameters
    ----------
    alpha : float, optional
        Parameter controlling the steepness of the surrogate gradient.
        Higher values make the transition sharper. Default is 1.0.

    See Also
    --------
    nonzero_sign_log : Deprecated function version. Use NonzeroSignLog class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a nonzero sign log surrogate
        >>> nsl_fn = brainstate.surrogate.NonzeroSignLog(alpha=1.0)
        >>>
        >>> # Apply to input
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = nsl_fn(x)

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   nsl_fn = brainstate.surrogate.NonzeroSignLog(alpha=alpha)
       >>>   grads = brainstate.augment.vector_grad(nsl_fn)(xs)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = 1. / (1 / self.alpha + jnp.abs(x))
        return dx

    def surrogate_fun(self, x):
        return jnp.where(x < 0, -1., 1.) * jnp.log(jnp.abs(self.alpha * x) + 1)

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def nonzero_sign_log(
    x: jax.Array,
    alpha: float = 1.,
):
    """
    Spike function with the nonzero sign log surrogate gradient.

    See the documentation of :class:`NonzeroSignLog` for details.
    """
    return NonzeroSignLog(alpha=alpha)(x)


class ERF(Surrogate):
    """Judge spiking state with an error function (erf).

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       \begin{split}
        g(x) &= \frac{1}{2}(1-\text{erf}(-\alpha x)) \\
        &= \frac{1}{2} \text{erfc}(-\alpha x) \\
        &= \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\alpha x}e^{-t^2}dt
        \end{split}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{\sqrt{\pi}}e^{-\alpha^2x^2}

    Parameters
    ----------
    alpha : float, optional
        Parameter controlling the steepness of the surrogate gradient.
        Higher values make the transition sharper. Default is 1.0.

    See Also
    --------
    erf : Deprecated function version. Use ERF class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create an ERF surrogate
        >>> erf_fn = brainstate.surrogate.ERF(alpha=1.0)
        >>>
        >>> # Apply to input
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = erf_fn(x)
        >>> print(spikes)  # [0., 1., 1.]

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   erf_fn = brainstate.surrogate.ERF(alpha=alpha)
       >>>   grads = brainstate.augment.vector_grad(erf_fn)(xs)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    References
    ----------
    .. [1] Esser S K, Appuswamy R, Merolla P, et al. Backpropagation for energy-efficient neuromorphic computing[J]. Advances in neural information processing systems, 2015, 28: 1117-1125.
    .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
    .. [3] Yin B, Corradi F, Bohté S M. Effective and efficient computation with multiple-timescale spiking recurrent neural networks[C]//International Conference on Neuromorphic Systems 2020. 2020: 1-8.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = (self.alpha / jnp.sqrt(jnp.pi)) * jnp.exp(-jnp.power(self.alpha, 2) * x * x)
        return dx

    def surrogate_fun(self, x):
        return sci.special.erf(-self.alpha * x) * 0.5

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def erf(
    x: jax.Array,
    alpha: float = 1.,
):
    """
    Spike function with the error function surrogate gradient.

    See the documentation of :class:`ERF` for details.
    """
    return ERF(alpha=alpha)(x)


class PiecewiseLeakyRelu(Surrogate):
    """Judge spiking state with a piecewise leaky relu function.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       \begin{split}g(x) =
        \begin{cases}
        cx + cw, & x < -w \\
        \frac{1}{2w}x + \frac{1}{2}, & -w \leq x \leq w \\
        cx - cw + 1, & x > w \\
        \end{cases}\end{split}

    Backward function:

    .. math::

       \begin{split}g'(x) =
        \begin{cases}
        \frac{1}{w}, & |x| \leq w \\
        c, & |x| > w
        \end{cases}\end{split}

    Parameters
    ----------
    c : float, optional
        Leakiness parameter for gradients outside the window. Default is 0.01.
    w : float, optional
        Half-width of the gradient window. Default is 1.0.

    See Also
    --------
    piecewise_leaky_relu : Deprecated function version. Use PiecewiseLeakyRelu class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a piecewise leaky ReLU surrogate
        >>> plr_fn = brainstate.surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0)
        >>>
        >>> # Apply to input
        >>> x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        >>> spikes = plr_fn(x)
        >>> print(spikes)  # [0., 0., 1., 1., 1.]

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for c in [0.01, 0.05, 0.1]:
       >>>   for w in [1., 2.]:
       >>>     plr_fn = brainstate.surrogate.PiecewiseLeakyRelu(c=c, w=w)
       >>>     grads = brainstate.augment.vector_grad(plr_fn)(xs)
       >>>     plt.plot(xs, grads, label=f'c={c}, w={w}')
       >>> plt.legend()
       >>> plt.show()

    Notes
    -----
    This surrogate provides a leaky gradient outside the window [-w, w], which can
    help with gradient flow in deep networks while maintaining a strong gradient
    near the threshold.

    References
    ----------
    .. [1] Yin S, Venkataramanaiah S K, Chen G K, et al. Algorithm and hardware design of discrete-time spiking neural networks based on back propagation with binary activations[C]//2017 IEEE Biomedical Circuits and Systems Conference (BioCAS). IEEE, 2017: 1-5.
    .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
    .. [3] Huh D, Sejnowski T J. Gradient descent for spiking neural networks[C]//Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018: 1440-1450.
    .. [4] Wu Y, Deng L, Li G, et al. Direct training for spiking neural networks: Faster, larger, better[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 1311-1318.
    .. [5] Gu P, Xiao R, Pan G, et al. STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks[C]//IJCAI. 2019: 1366-1372.
    .. [6] Roy D, Chakraborty I, Roy K. Scaling deep spiking neural networks with binary stochastic activations[C]//2019 IEEE International Conference on Cognitive Computing (ICCC). IEEE, 2019: 50-58.
    .. [7] Cheng X, Hao Y, Xu J, et al. LISNN: Improving Spiking Neural Networks with Lateral Interactions for Robust Object Recognition[C]//IJCAI. 1519-1525.
    .. [8] Kaiser J, Mostafa H, Neftci E. Synaptic plasticity dynamics for deep continuous local learning (DECOLLE)[J]. Frontiers in Neuroscience, 2020, 14: 424.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, c=0.01, w=1.):
        super().__init__()
        self.c = c
        self.w = w

    def surrogate_fun(self, x):
        z = jnp.where(
            x < -self.w,
            self.c * x + self.c * self.w,
            jnp.where(
                x > self.w,
                self.c * x - self.c * self.w + 1,
                0.5 * x / self.w + 0.5
            )
        )
        return z

    def surrogate_grad(self, x):
        dx = jnp.where(jnp.abs(x) > self.w, self.c, 1 / self.w)
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c}, w={self.w})'

    def __hash__(self):
        return hash((self.__class__, self.c, self.w))


def piecewise_leaky_relu(
    x: jax.Array,
    c: float = 0.01,
    w: float = 1.,
):
    """
    Spike function with the piecewise leaky ReLU surrogate gradient.

    See the documentation of :class:`PiecewiseLeakyRelu` for details.
    """
    return PiecewiseLeakyRelu(c=c, w=w)(x)


class SquarewaveFourierSeries(Surrogate):
    """Judge spiking state with a squarewave Fourier series.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       g(x) = 0.5 + \frac{1}{\pi}*\sum_{i=1}^n {\sin\left({(2i-1)*2\pi}*x/T\right) \over 2i-1 }

    Backward function:

    .. math::

       g'(x) = \sum_{i=1}^n\frac{4\cos\left((2 * i - 1.) * 2\pi * x / T\right)}{T}

    Parameters
    ----------
    n : int, optional
        Number of Fourier terms. Default is 2.
    t_period : float, optional
        Period of the square wave. Default is 8.0.

    See Also
    --------
    squarewave_fourier_series : Deprecated function version. Use SquarewaveFourierSeries class instead.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a squarewave Fourier series surrogate
        >>> sfs_fn = brainstate.surrogate.SquarewaveFourierSeries(n=4, t_period=8.0)
        >>>
        >>> # Apply to input
        >>> x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> spikes = sfs_fn(x)
        >>> print(spikes)  # [0., 0., 1., 1., 1.]

    .. plot::
       :include-source: True

       >>> import jax
       >>> import brainstate as brainstate
       >>> import matplotlib.pyplot as plt
       >>> xs = jax.numpy.linspace(-3, 3, 1000)
       >>> for n in [2, 4, 8]:
       >>>   sfs_fn = brainstate.surrogate.SquarewaveFourierSeries(n=n)
       >>>   grads = brainstate.augment.vector_grad(sfs_fn)(xs)
       >>>   plt.plot(xs, grads, label=f'n={n}')
       >>> plt.legend()
       >>> plt.show()

    Notes
    -----
    This surrogate uses a Fourier series approximation of a square wave,
    providing a periodic gradient that can be useful for certain types of
    spiking neural networks.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, n=2, t_period=8.):
        super().__init__()
        self.n = n
        self.t_period = t_period

    def surrogate_grad(self, x):

        w = jnp.pi * 2. / self.t_period
        dx = jnp.cos(w * x)
        for i in range(2, self.n):
            dx += jnp.cos((2 * i - 1.) * w * x)
        dx *= 4. / self.t_period
        return dx

    def surrogate_fun(self, x):

        w = jnp.pi * 2. / self.t_period
        ret = jnp.sin(w * x)
        for i in range(2, self.n):
            c = (2 * i - 1.)
            ret += jnp.sin(c * w * x) / c
        z = 0.5 + 2. / jnp.pi * ret
        return z

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self.n}, t_period={self.t_period})'

    def __hash__(self):
        return hash((self.__class__, self.n, self.t_period))


def squarewave_fourier_series(
    x: jax.Array,
    n: int = 2,
    t_period: float = 8.,
):
    """
    Spike function with the squarewave Fourier series surrogate gradient.

    See the documentation of :class:`SquarewaveFourierSeries` for details.
    """
    return SquarewaveFourierSeries(n=n, t_period=t_period)(x)


class S2NN(Surrogate):
    r"""Judge spiking state with the S2NN surrogate spiking function [1]_.

    The S2NN (Single-Step Neural Network) surrogate gradient is designed for
    training energy-efficient single-step neural networks. It provides asymmetric
    gradients for positive and negative inputs, enabling better gradient flow
    during training.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       \begin{split}g_{origin}(x) = \begin{cases}
          \mathrm{sigmoid} (\alpha x), & x < 0 \\
          \beta \ln(|x + 1|) + 0.5, & x \ge 0
      \end{cases}\end{split}

    Backward gradient:

    .. math::

       \begin{split}g'(x) = \begin{cases}
          \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x), & x < 0 \\
          \frac{\beta}{(x + 1)}, & x \ge 0
      \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-3, 3, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different parameters
       >>> for alpha, beta in [(4., 1.), (8., 2.), (2., 0.5)]:
       >>>     s2nn_fn = surrogate.S2NN(alpha=alpha, beta=beta)
       >>>     grads = jax.vmap(jax.grad(s2nn_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\alpha={alpha}, \beta={beta}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('S2NN Surrogate Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>>
       >>> # Plot the original function for origin=True
       >>> for alpha, beta in [(4., 1.), (8., 2.)]:
       >>>     s2nn_fn = surrogate.S2NN(alpha=alpha, beta=beta)
       >>>     s2nn_fn.origin = True
       >>>     ys = jax.vmap(s2nn_fn)(xs)
       >>>     ax2.plot(xs, ys, label=rf'$\alpha={alpha}, \beta={beta}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Output')
       >>> ax2.set_title('S2NN Original Function')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter controlling gradient when x < 0. Default is 4.0.
        Larger values create steeper gradients for negative inputs.
    beta : float, optional
        Parameter controlling gradient when x >= 0. Default is 1.0.
        Larger values create stronger gradients for positive inputs.
    epsilon : float, optional
        Small value to avoid numerical issues in logarithm. Default is 1e-8.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create S2NN surrogate function
        >>> s2nn_fn = surrogate.S2NN(alpha=4.0, beta=1.0)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-1., 0., 1.])
        >>> spikes = s2nn_fn(x)
        >>> print(spikes)
        [0. 1. 1.]
        >>>
        >>> # Compute gradients
        >>> grad_fn = jax.grad(lambda x: s2nn_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(grads)

    See Also
    --------
    s2nn : Functional version of S2NN surrogate gradient.
    Sigmoid : Symmetric sigmoid-based surrogate gradient.
    PiecewiseQuadratic : Quadratic approximation surrogate gradient.

    References
    ----------
    .. [1] Suetake, Kazuma et al. "S2NN: Time Step Reduction of Spiking Surrogate
           Gradients for Training Energy Efficient Single-Step Neural Networks."
           ArXiv abs/2201.10879 (2022): n. pag.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=4., beta=1., epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def surrogate_fun(self, x):
        z = jnp.where(
            x < 0.,
            sci.special.expit(x * self.alpha),
            self.beta * jnp.log(jnp.abs((x + 1.)) + self.epsilon) + 0.5
        )
        return z

    def surrogate_grad(self, x):
        sg = sci.special.expit(self.alpha * x)
        dx = jnp.where(x < 0., self.alpha * sg * (1. - sg), self.beta / (x + 1.))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta}, epsilon={self.epsilon})'

    def __hash__(self):
        return hash((self.__class__, self.alpha, self.beta, self.epsilon))


def s2nn(
    x: jax.Array,
    alpha: float = 4.,
    beta: float = 1.,
    epsilon: float = 1e-8,
):
    """
    Spike function with the S2NN surrogate gradient.

    See the documentation of :class:`S2NN` for details.
    """
    return S2NN(alpha=alpha, beta=beta, epsilon=epsilon)(x)


class QPseudoSpike(Surrogate):
    r"""Judge spiking state with the q-PseudoSpike surrogate function [1]_.

    The q-PseudoSpike surrogate gradient provides a flexible framework for
    controlling the tail behavior of the gradient function. The parameter q
    (represented as alpha in the implementation) controls the tail fatness,
    allowing for various gradient profiles from heavy-tailed to compact support.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       \begin{split}g_{origin}(x) =
        \begin{cases}
        \frac{1}{2}(1-\frac{2x}{\alpha-1})^{1-\alpha}, & x < 0 \\
        1 - \frac{1}{2}(1+\frac{2x}{\alpha-1})^{1-\alpha}, & x \geq 0.
        \end{cases}\end{split}

    Backward gradient:

    .. math::

       g'(x) = (1+\frac{2|x|}{\alpha-1})^{-\alpha}

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-3, 3, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different alpha values
       >>> for alpha in [0.5, 1.0, 2.0, 4.0]:
       >>>     qps_fn = surrogate.QPseudoSpike(alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(qps_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('q-PseudoSpike Surrogate Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>> ax1.set_ylim([0, 1.2])
       >>>
       >>> # Plot the original function for origin=True
       >>> for alpha in [1.5, 2.0, 3.0]:
       >>>     qps_fn = surrogate.QPseudoSpike(alpha=alpha)
       >>>     qps_fn.origin = True
       >>>     ys = jax.vmap(qps_fn)(xs)
       >>>     ax2.plot(xs, ys, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Output')
       >>> ax2.set_title('q-PseudoSpike Original Function')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter to control tail fatness of gradient. Default is 2.0.

        - alpha < 1: Heavy-tailed gradient (slower decay)
        - alpha = 1: Exponential-like decay
        - alpha > 1: Compact support (faster decay)
        - alpha = 2: Quadratic decay (default)

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create q-PseudoSpike surrogate function
        >>> qps_fn = surrogate.QPseudoSpike(alpha=2.0)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-1., 0., 1.])
        >>> spikes = qps_fn(x)
        >>> print(spikes)
        [0. 1. 1.]
        >>>
        >>> # Compute gradients with different tail behaviors
        >>> for alpha in [0.5, 2.0, 4.0]:
        ...     qps_fn = surrogate.QPseudoSpike(alpha=alpha)
        ...     grad_fn = jax.grad(lambda x: qps_fn(x).sum())
        ...     grads = grad_fn(jax.numpy.array([0.5]))
        ...     print(f"alpha={alpha}: gradient={grads[0]:.4f}")

    See Also
    --------
    q_pseudo_spike : Functional version of q-PseudoSpike surrogate gradient.
    Sigmoid : Sigmoid-based surrogate gradient.
    S2NN : Asymmetric surrogate gradient for single-step networks.

    References
    ----------
    .. [1] Herranz-Celotti, Luca and Jean Rouat. "Surrogate Gradients Design."
           ArXiv abs/2202.00282 (2022): n. pag.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=2.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = jnp.power(1 + 2 / (self.alpha + 1) * jnp.abs(x), -self.alpha)
        return dx

    def surrogate_fun(self, x):
        z = jnp.where(
            x < 0.,
            0.5 * jnp.power(1 - 2 / (self.alpha - 1) * jnp.abs(x), 1 - self.alpha),
            1. - 0.5 * jnp.power(1 + 2 / (self.alpha - 1) * jnp.abs(x), 1 - self.alpha)
        )
        return z

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def q_pseudo_spike(
    x: jax.Array,
    alpha: float = 2.,
):
    """
    Spike function with the q-PseudoSpike surrogate gradient.

    See the documentation of :class:`QPseudoSpike` for details.
    """
    return QPseudoSpike(alpha=alpha)(x)


class LeakyRelu(Surrogate):
    r"""Judge spiking state with the Leaky ReLU function.

    The Leaky ReLU surrogate gradient provides a simple piecewise linear
    approximation with different slopes for positive and negative inputs.
    This allows gradients to flow even for negative inputs, preventing the
    "dying ReLU" problem in spiking neural networks.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       \begin{split}g_{origin}(x) =
        \begin{cases}
        \beta \cdot x, & x \geq 0 \\
        \alpha \cdot x, & x < 0 \\
        \end{cases}\end{split}

    Backward gradient:

    .. math::

       \begin{split}g'(x) =
        \begin{cases}
        \beta, & x \geq 0 \\
        \alpha, & x < 0 \\
        \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-3, 3, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different parameters
       >>> for alpha, beta in [(0.0, 1.0), (0.1, 1.0), (0.3, 1.0), (0.1, 0.5)]:
       >>>     lr_fn = surrogate.LeakyRelu(alpha=alpha, beta=beta)
       >>>     grads = jax.vmap(jax.grad(lr_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\alpha={alpha}, \beta={beta}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('Leaky ReLU Surrogate Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>> ax1.set_ylim([-0.1, 1.2])
       >>>
       >>> # Plot the original function for origin=True
       >>> for alpha, beta in [(0.1, 1.0), (0.3, 1.0), (0.1, 0.5)]:
       >>>     lr_fn = surrogate.LeakyRelu(alpha=alpha, beta=beta)
       >>>     lr_fn.origin = True
       >>>     ys = jax.vmap(lr_fn)(xs)
       >>>     ax2.plot(xs, ys, label=rf'$\alpha={alpha}, \beta={beta}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Output')
       >>> ax2.set_title('Leaky ReLU Original Function')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter to control gradient when x < 0. Default is 0.1.
        Setting alpha=0 gives standard ReLU behavior.
    beta : float, optional
        Parameter to control gradient when x >= 0. Default is 1.0.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create Leaky ReLU surrogate function
        >>> lr_fn = surrogate.LeakyRelu(alpha=0.1, beta=1.0)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-1., 0., 1.])
        >>> spikes = lr_fn(x)
        >>> print(spikes)
        [0. 1. 1.]
        >>>
        >>> # Compute gradients
        >>> grad_fn = jax.grad(lambda x: lr_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(grads)
        [0.1 1.  1. ]

    See Also
    --------
    leaky_relu : Functional version of Leaky ReLU surrogate gradient.
    ReluGrad : Standard ReLU-based surrogate gradient.
    PiecewiseLeakyRelu : Piecewise approximation with leaky ReLU.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=0.1, beta=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def surrogate_fun(self, x):
        return jnp.where(x < 0., self.alpha * x, self.beta * x)

    def surrogate_grad(self, x):
        dx = jnp.where(x < 0., self.alpha, self.beta)
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'

    def __hash__(self):
        return hash((self.__class__, self.alpha, self.beta))


def leaky_relu(
    x: jax.Array,
    alpha: float = 0.1,
    beta: float = 1.,
):
    """
    Spike function with the Leaky ReLU surrogate gradient.

    See the documentation of :class:`LeakyRelu` for details.
    """
    return LeakyRelu(alpha=alpha, beta=beta)(x)


class LogTailedRelu(Surrogate):
    r"""Judge spiking state with the Log-tailed ReLU function [1]_.

    The Log-tailed ReLU surrogate gradient combines linear behavior for small
    positive inputs with logarithmic scaling for large inputs. This provides
    bounded gradients for large activations while maintaining responsiveness
    for smaller values, useful for handling wide dynamic ranges in spiking
    neural networks.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    The original function:

    .. math::

       \begin{split}g_{origin}(x) =
        \begin{cases}
        \alpha x, & x \leq 0 \\
        x, & 0 < x \leq 1 \\
        \log(x), & x > 1 \\
        \end{cases}\end{split}

    Backward gradient:

    .. math::

       \begin{split}g'(x) =
        \begin{cases}
        \alpha, & x \leq 0 \\
        1, & 0 < x \leq 1 \\
        \frac{1}{x}, & x > 1 \\
        \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-2, 4, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different alpha values
       >>> for alpha in [0.0, 0.1, 0.3]:
       >>>     ltr_fn = surrogate.LogTailedRelu(alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(ltr_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('Log-tailed ReLU Surrogate Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>> ax1.set_ylim([-0.1, 1.2])
       >>>
       >>> # Plot the original function for origin=True
       >>> for alpha in [0.0, 0.1, 0.3]:
       >>>     ltr_fn = surrogate.LogTailedRelu(alpha=alpha)
       >>>     ltr_fn.origin = True
       >>>     ys = jax.vmap(ltr_fn)(xs)
       >>>     ax2.plot(xs, ys, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Output')
       >>> ax2.set_title('Log-tailed ReLU Original Function')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter to control the gradient for negative inputs. Default is 0.0.

        - alpha = 0: No gradient for negative inputs (standard behavior)
        - alpha > 0: Leaky gradient for negative inputs

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create Log-tailed ReLU surrogate function
        >>> ltr_fn = surrogate.LogTailedRelu(alpha=0.1)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-1., 0.5, 2.])
        >>> spikes = ltr_fn(x)
        >>> print(spikes)
        [0. 1. 1.]
        >>>
        >>> # Compute gradients showing different regimes
        >>> grad_fn = jax.grad(lambda x: ltr_fn(x).sum())
        >>> x_test = jax.numpy.array([-1., 0.5, 2.])
        >>> grads = grad_fn(x_test)
        >>> print(grads)  # Shows alpha, 1.0, 1/2 respectively

    See Also
    --------
    log_tailed_relu : Functional version of Log-tailed ReLU surrogate gradient.
    LeakyRelu : Simple leaky ReLU surrogate gradient.
    ReluGrad : Standard ReLU-based surrogate gradient.

    References
    ----------
    .. [1] Cai, Zhaowei et al. "Deep Learning with Low Precision by Half-Wave
           Gaussian Quantization." 2017 IEEE Conference on Computer Vision and
           Pattern Recognition (CVPR) (2017): 5406-5414.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=0.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        z = jnp.where(
            x > 1,
            jnp.log(x),
            jnp.where(
                x > 0,
                x,
                self.alpha * x
            )
        )
        return z

    def surrogate_grad(self, x):
        dx = jnp.where(x > 1,
                       1 / x,
                       jnp.where(x > 0,
                                 1.,
                                 self.alpha))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def log_tailed_relu(
    x: jax.Array,
    alpha: float = 0.,
):
    """
    Spike function with the Log-tailed ReLU surrogate gradient.

    See the documentation of :class:`LogTailedRelu` for details.
    """
    return LogTailedRelu(alpha=alpha)(x)


class ReluGrad(Surrogate):
    r"""Judge spiking state with the ReLU gradient function [1]_.

    The ReLU gradient surrogate provides a triangular-shaped gradient function
    with finite support. It creates a linear decrease from the center to the
    edges, providing a simple and computationally efficient gradient that is
    non-zero only within a specified width around zero.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward gradient:

    .. math::

       g'(x) = \text{ReLU}(\alpha * (\text{width} - |x|))
       = \max(0, \alpha * (\text{width} - |x|))

    This creates a triangular gradient centered at x=0 with:

    - Peak value: α × width at x=0
    - Linear decrease to 0 at x=±width
    - Zero gradient for |x| > width

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-3, 3, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different parameter combinations
       >>> for alpha in [0.3, 0.5, 1.0]:
       >>>     for width in [1.0, 2.0]:
       >>>         rg_fn = surrogate.ReluGrad(alpha=alpha, width=width)
       >>>         grads = jax.vmap(jax.grad(rg_fn))(xs)
       >>>         ax1.plot(xs, grads, label=rf'$\alpha={alpha}, w={width}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('ReLU Surrogate Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>>
       >>> # Show effect of width parameter
       >>> alpha_fixed = 0.5
       >>> for width in [0.5, 1.0, 1.5, 2.0]:
       >>>     rg_fn = surrogate.ReluGrad(alpha=alpha_fixed, width=width)
       >>>     grads = jax.vmap(jax.grad(rg_fn))(xs)
       >>>     ax2.plot(xs, grads, label=rf'$width={width}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Gradient')
       >>> ax2.set_title(f'Width Effect (α={alpha_fixed})')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter to control the gradient magnitude. Default is 0.3.
        The peak gradient value is alpha × width.
    width : float, optional
        Parameter to control the width of the gradient support. Default is 1.0.
        Gradient is non-zero only for |x| < width.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create ReLU gradient surrogate function
        >>> rg_fn = surrogate.ReluGrad(alpha=0.3, width=1.0)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-2., -0.5, 0., 0.5, 2.])
        >>> spikes = rg_fn(x)
        >>> print(spikes)
        [0. 0. 1. 1. 1.]
        >>>
        >>> # Compute gradients
        >>> grad_fn = jax.grad(lambda x: rg_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(grads)  # Shows 0, 0.15, 0.3, 0.15, 0

    See Also
    --------
    relu_grad : Functional version of ReLU gradient surrogate.
    LeakyRelu : Leaky ReLU surrogate gradient.
    PiecewiseLinear : General piecewise linear surrogate gradient.

    References
    ----------
    .. [1] Neftci, E. O., Mostafa, H. & Zenke, F. Surrogate gradient learning
           in spiking neural networks. IEEE Signal Process. Mag. 36, 61–63 (2019).

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=0.3, width=1.):
        super().__init__()
        self.alpha = alpha
        self.width = width

    def surrogate_grad(self, x):
        dx = jnp.maximum(self.alpha * self.width - jnp.abs(x) * self.alpha, 0)
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, width={self.width})'

    def __hash__(self):
        return hash((self.__class__, self.alpha, self.width))


def relu_grad(
    x: jax.Array,
    alpha: float = 0.3,
    width: float = 1.,
):
    """
    Spike function with the ReLU gradient surrogate.

    See the documentation of :class:`ReluGrad` for details.
    """
    return ReluGrad(alpha=alpha, width=width)(x)


class GaussianGrad(Surrogate):
    r"""Judge spiking state with the Gaussian gradient function [1]_.

    The Gaussian gradient surrogate provides a smooth, bell-shaped gradient
    function based on the Gaussian (normal) distribution. This creates a
    differentiable approximation to the Heaviside step function with
    continuous derivatives of all orders, making it particularly suitable
    for gradient-based optimization in spiking neural networks.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward gradient:

    .. math::

       g'(x) = \alpha \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{x^2}{2\sigma^2}\right)

    where the gradient follows a Gaussian distribution centered at x=0 with:

    - Standard deviation σ controlling the width
    - Scaling factor α controlling the peak height
    - Peak value at x=0: α/(σ√(2π))

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-4, 4, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different sigma values
       >>> alpha = 0.5
       >>> for sigma in [0.3, 0.5, 1.0, 2.0]:
       >>>     gg_fn = surrogate.GaussianGrad(sigma=sigma, alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(gg_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\sigma={sigma}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title(f'Gaussian Gradients (α={alpha})')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>>
       >>> # Plot gradients for different alpha values
       >>> sigma = 0.5
       >>> for alpha in [0.25, 0.5, 1.0, 2.0]:
       >>>     gg_fn = surrogate.GaussianGrad(sigma=sigma, alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(gg_fn))(xs)
       >>>     ax2.plot(xs, grads, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Gradient')
       >>> ax2.set_title(f'Scaling Effect (σ={sigma})')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    sigma : float, optional
        Parameter to control the variance (width) of Gaussian distribution. Default is 0.5.
        Smaller values create sharper gradients, larger values create smoother gradients.
    alpha : float, optional
        Parameter to control the scale (height) of the gradient. Default is 0.5.
        Determines the maximum gradient value at x=0.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create Gaussian gradient surrogate function
        >>> gg_fn = surrogate.GaussianGrad(sigma=0.5, alpha=0.5)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-1., 0., 1.])
        >>> spikes = gg_fn(x)
        >>> print(spikes)
        [0. 1. 1.]
        >>>
        >>> # Compute gradients
        >>> grad_fn = jax.grad(lambda x: gg_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(f"Gradients: {grads}")

    See Also
    --------
    gaussian_grad : Functional version of Gaussian gradient surrogate.
    MultiGaussianGrad : Multi-component Gaussian gradient.
    Sigmoid : Sigmoid-based surrogate gradient.

    References
    ----------
    .. [1] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain
           classification with adaptive spiking recurrent neural networks.
           Nat Mach Intell 3, 905–913 (2021).

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, sigma=0.5, alpha=0.5):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = jnp.exp(-(x ** 2) / 2 * jnp.power(self.sigma, 2)) / (jnp.sqrt(2 * jnp.pi) * self.sigma)
        return self.alpha * dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma})'

    def __hash__(self):
        return hash((self.__class__, self.alpha, self.sigma))


def gaussian_grad(
    x: jax.Array,
    sigma: float = 0.5,
    alpha: float = 0.5,
):
    """
    Spike function with the Gaussian gradient surrogate.

    See the documentation of :class:`GaussianGrad` for details.
    """
    return GaussianGrad(sigma=sigma, alpha=alpha)(x)


class MultiGaussianGrad(Surrogate):
    r"""Judge spiking state with the multi-Gaussian gradient function [1]_.

    The Multi-Gaussian gradient surrogate combines three Gaussian components
    to create a more complex gradient profile. It uses a positive central
    Gaussian and two negative side Gaussians, allowing for enhanced gradient
    flow and potentially better training dynamics in spiking neural networks.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward gradient:

    .. math::

       g'(x) = \text{scale} \cdot \left[
       (1+h) \cdot \mathcal{N}(x; 0, \sigma^2) -
       h \cdot \mathcal{N}(x; \sigma, (s\sigma)^2) -
       h \cdot \mathcal{N}(x; -\sigma, (s\sigma)^2)
       \right]

    where :math:`\mathcal{N}(x; \mu, \sigma^2)` is the Gaussian PDF with mean μ and variance σ².

    The gradient consists of:

    - A central positive Gaussian at x=0 with weight (1+h)
    - Two negative side Gaussians at x=±σ with weight -h
    - Side Gaussians have wider spread controlled by parameter s

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-3, 3, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot default multi-Gaussian gradient
       >>> mgg_fn = surrogate.MultiGaussianGrad()
       >>> grads = jax.vmap(jax.grad(mgg_fn))(xs)
       >>> ax1.plot(xs, grads, label='Multi-Gaussian', linewidth=2)
       >>>
       >>> # Compare with single Gaussian
       >>> gg_fn = surrogate.GaussianGrad(sigma=0.5, alpha=0.5)
       >>> grads_single = jax.vmap(jax.grad(gg_fn))(xs)
       >>> ax1.plot(xs, grads_single, '--', label='Single Gaussian', alpha=0.7)
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('Multi-Gaussian vs Single Gaussian')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>> ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
       >>>
       >>> # Show effect of h parameter (side peak weight)
       >>> for h in [0.0, 0.15, 0.3, 0.5]:
       >>>     mgg_fn = surrogate.MultiGaussianGrad(h=h, s=6.0, sigma=0.5, scale=0.5)
       >>>     grads = jax.vmap(jax.grad(mgg_fn))(xs)
       >>>     ax2.plot(xs, grads, label=rf'$h={h}$')
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('Gradient')
       >>> ax2.set_title('Effect of h Parameter')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3)
       >>> ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    h : float, optional
        Weight parameter for side Gaussians. Default is 0.15.
        Controls the depth of negative side lobes.
    s : float, optional
        Width scaling factor for side Gaussians. Default is 6.0.
        Larger values make side Gaussians wider.
    sigma : float, optional
        Standard deviation of central Gaussian and position of side peaks. Default is 0.5.
    scale : float, optional
        Overall gradient scaling factor. Default is 0.5.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create multi-Gaussian gradient surrogate
        >>> mgg_fn = surrogate.MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-1., -0.5, 0., 0.5, 1.])
        >>> spikes = mgg_fn(x)
        >>> print(spikes)
        [0. 0. 1. 1. 1.]
        >>>
        >>> # Compute gradients showing multi-peak structure
        >>> grad_fn = jax.grad(lambda x: mgg_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(f"Gradients: {grads}")

    See Also
    --------
    multi_gaussian_grad : Functional version of multi-Gaussian gradient.
    GaussianGrad : Single Gaussian gradient surrogate.
    Sigmoid : Sigmoid-based surrogate gradient.

    References
    ----------
    .. [1] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain
           classification with adaptive spiking recurrent neural networks.
           Nat Mach Intell 3, 905–913 (2021).

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, h=0.15, s=6.0, sigma=0.5, scale=0.5):
        super().__init__()
        self.h = h
        self.s = s
        self.sigma = sigma
        self.scale = scale

    def surrogate_grad(self, x):
        g1 = jnp.exp(-x ** 2 / (2 * jnp.power(self.sigma, 2))) / (jnp.sqrt(2 * jnp.pi) * self.sigma)
        g2 = jnp.exp(
            -(x - self.sigma) ** 2 / (2 * jnp.power(self.s * self.sigma, 2))
        ) / (jnp.sqrt(2 * jnp.pi) * self.s * self.sigma)
        g3 = jnp.exp(
            -(x + self.sigma) ** 2 / (2 * jnp.power(self.s * self.sigma, 2))
        ) / (jnp.sqrt(2 * jnp.pi) * self.s * self.sigma)
        dx = g1 * (1. + self.h) - g2 * self.h - g3 * self.h
        return self.scale * dx

    def __repr__(self):
        return f'{self.__class__.__name__}(h={self.h}, s={self.s}, sigma={self.sigma}, scale={self.scale})'

    def __hash__(self):
        return hash((self.__class__, self.h, self.s, self.sigma, self.scale))


def multi_gaussian_grad(
    x: jax.Array,
    h: float = 0.15,
    s: float = 6.0,
    sigma: float = 0.5,
    scale: float = 0.5,
):
    """
    Spike function with the multi-Gaussian gradient surrogate.

    See the documentation of :class:`MultiGaussianGrad` for details.
    """
    return MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale)(x)


class InvSquareGrad(Surrogate):
    r"""Judge spiking state with the inverse-square surrogate gradient function.

    The inverse-square gradient surrogate provides a smooth approximation
    with a Lorentzian-like profile. It has heavier tails than Gaussian
    gradients, allowing for gradient flow even far from the threshold,
    while maintaining a sharp peak at the origin.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward gradient:

    .. math::

       g'(x) = \frac{1}{(\alpha \cdot |x| + 1)^2}

    This creates a gradient with:

    - Peak value of 1 at x=0
    - Power-law decay proportional to 1/x² for large |x|
    - Width controlled by 1/α

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-1, 1, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different alpha values
       >>> for alpha in [10., 50., 100., 200.]:
       >>>     isg_fn = surrogate.InvSquareGrad(alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(isg_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('Inverse-Square Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>>
       >>> # Compare with other surrogate gradients on log scale
       >>> xs_wide = jnp.linspace(-3, 3, 1000)
       >>> isg_fn = surrogate.InvSquareGrad(alpha=100.)
       >>> grads_inv = jax.vmap(jax.grad(isg_fn))(xs_wide)
       >>>
       >>> # Compare with Gaussian
       >>> gg_fn = surrogate.GaussianGrad(sigma=0.1, alpha=1.0)
       >>> grads_gauss = jax.vmap(jax.grad(gg_fn))(xs_wide)
       >>>
       >>> ax2.semilogy(xs_wide, jnp.abs(grads_inv), label='Inverse-Square', linewidth=2)
       >>> ax2.semilogy(xs_wide, jnp.abs(grads_gauss), '--', label='Gaussian', alpha=0.7)
       >>>
       >>> ax2.set_xlabel('Input (x)')
       >>> ax2.set_ylabel('|Gradient| (log scale)')
       >>> ax2.set_title('Tail Behavior Comparison')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3, which="both")
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter to control gradient sharpness. Default is 100.0.

        - Larger α creates sharper, more localized gradients
        - Smaller α creates wider, more distributed gradients
        - Effective width ≈ 2/α

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create inverse-square gradient surrogate
        >>> isg_fn = surrogate.InvSquareGrad(alpha=100.0)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-0.1, 0., 0.1])
        >>> spikes = isg_fn(x)
        >>> print(spikes)
        [0. 1. 1.]
        >>>
        >>> # Compute gradients
        >>> grad_fn = jax.grad(lambda x: isg_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(f"Gradients: {grads}")
        >>> # Shows heavy-tailed behavior

    See Also
    --------
    inv_square_grad : Functional version of inverse-square gradient.
    GaussianGrad : Gaussian-based surrogate gradient.
    SlayerGrad : Exponential decay surrogate gradient.

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=100.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = 1. / (self.alpha * jnp.abs(x) + 1.0) ** 2
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def inv_square_grad(
    x: jax.Array,
    alpha: float = 100.
):
    """
    Spike function with the inverse-square surrogate gradient.

    See the documentation of :class:`InvSquareGrad` for details.
    """
    return InvSquareGrad(alpha=alpha)(x)


class SlayerGrad(Surrogate):
    r"""Judge spiking state with the slayer surrogate gradient function [1]_.

    The SLAYER (Spike LAYer Error Reassignment) gradient provides an
    exponential decay surrogate that enables error backpropagation in
    spiking neural networks. It uses a Laplace-like distribution for
    the gradient, offering a good balance between gradient magnitude
    near the threshold and computational efficiency.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward gradient:

    .. math::

       g'(x) = \exp(-\alpha \cdot |x|)

    This creates an exponentially decaying gradient with:

    - Peak value of 1 at x=0
    - Exponential decay rate controlled by α
    - Symmetric profile around the threshold

    .. plot::
       :include-source: True

       >>> import jax
       >>> import jax.numpy as jnp
       >>> import brainstate
       >>> import brainstate.surrogate as surrogate
       >>> import matplotlib.pyplot as plt
       >>>
       >>> xs = jnp.linspace(-4, 4, 1000)
       >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
       >>>
       >>> # Plot gradients for different alpha values
       >>> for alpha in [0.5, 1.0, 2.0, 4.0]:
       >>>     sg_fn = surrogate.SlayerGrad(alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(sg_fn))(xs)
       >>>     ax1.plot(xs, grads, label=rf'$\alpha={alpha}$')
       >>>
       >>> ax1.set_xlabel('Input (x)')
       >>> ax1.set_ylabel('Gradient')
       >>> ax1.set_title('SLAYER Surrogate Gradients')
       >>> ax1.legend()
       >>> ax1.grid(True, alpha=0.3)
       >>>
       >>> # Compare decay rates on semi-log plot
       >>> xs_pos = jnp.linspace(0, 4, 500)
       >>> for alpha in [0.5, 1.0, 2.0, 4.0]:
       >>>     sg_fn = surrogate.SlayerGrad(alpha=alpha)
       >>>     grads = jax.vmap(jax.grad(sg_fn))(xs_pos)
       >>>     ax2.semilogy(xs_pos, grads, label=rf'$\alpha={alpha}$')
       >>>
       >>> # Add theoretical exponential decay lines
       >>> for alpha in [1.0, 2.0]:
       >>>     theoretical = jnp.exp(-alpha * xs_pos)
       >>>     ax2.semilogy(xs_pos, theoretical, '--', alpha=0.5)
       >>>
       >>> ax2.set_xlabel('Distance from threshold')
       >>> ax2.set_ylabel('Gradient (log scale)')
       >>> ax2.set_title('Exponential Decay Behavior')
       >>> ax2.legend()
       >>> ax2.grid(True, alpha=0.3, which="both")
       >>> plt.tight_layout()
       >>> plt.show()

    Parameters
    ----------
    alpha : float, optional
        Parameter to control the decay rate of the gradient. Default is 1.0.

        - Larger α creates faster decay (sharper gradients)
        - Smaller α creates slower decay (wider gradients)
        - Decay length scale = 1/α

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainstate.surrogate as surrogate
        >>>
        >>> # Create SLAYER gradient surrogate
        >>> sg_fn = surrogate.SlayerGrad(alpha=1.0)
        >>>
        >>> # Apply to input
        >>> x = jax.numpy.array([-2., -1., 0., 1., 2.])
        >>> spikes = sg_fn(x)
        >>> print(spikes)
        [0. 0. 1. 1. 1.]
        >>>
        >>> # Compute gradients showing exponential decay
        >>> grad_fn = jax.grad(lambda x: sg_fn(x).sum())
        >>> grads = grad_fn(x)
        >>> print(f"Gradients: {grads}")
        >>> # Shows exp(-|x|) behavior

    See Also
    --------
    slayer_grad : Functional version of SLAYER gradient.
    GaussianGrad : Gaussian-based surrogate gradient.
    InvSquareGrad : Power-law decay surrogate gradient.

    References
    ----------
    .. [1] Shrestha, S. B. & Orchard, G. Slayer: spike layer error reassignment
           in time. In Advances in Neural Information Processing Systems
           Vol. 31, 1412–1421 (NeurIPS, 2018).

    """
    __module__ = 'brainstate.surrogate'

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = jnp.exp(-self.alpha * jnp.abs(x))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'

    def __hash__(self):
        return hash((self.__class__, self.alpha))


def slayer_grad(
    x: jax.Array,
    alpha: float = 1.
):
    """
    Spike function with the SLAYER surrogate gradient.

    See the documentation of :class:`SlayerGrad` for details.
    """
    return SlayerGrad(alpha=alpha)(x)
