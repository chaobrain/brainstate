# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
Random number generation module for BrainState.

This module provides a comprehensive set of random number generation functions and utilities
for neural network simulations and scientific computing. It wraps JAX's random number
generation capabilities with a stateful interface that simplifies usage while maintaining
reproducibility and performance.

The module includes:

- Standard random distributions (uniform, normal, exponential, etc.)
- Random state management with automatic key splitting
- Seed management utilities for reproducible simulations
- NumPy-compatible API for easy migration

Key Features
------------

- **Stateful random generation**: Automatic management of JAX's PRNG keys
- **NumPy compatibility**: Drop-in replacement for most NumPy random functions
- **Reproducibility**: Robust seed management and state tracking
- **Performance**: JIT-compiled random functions for efficient generation
- **Thread-safe**: Proper handling of random state in parallel computations

Random State Management
-----------------------

The module uses a global `DEFAULT` RandomState instance that automatically manages
JAX's PRNG keys. This eliminates the need to manually track and split keys:

.. code-block:: python

    >>> import brainstate as bs
    >>> import brainstate.random as bsr
    >>>
    >>> # Set a global seed for reproducibility
    >>> bsr.seed(42)
    >>>
    >>> # Generate random numbers without manual key management
    >>> x = bsr.normal(0, 1, size=(3, 3))
    >>> y = bsr.uniform(0, 1, size=(100,))

Custom Random States
--------------------

For more control, you can create custom RandomState instances:

.. code-block:: python

    >>> import brainstate.random as bsr
    >>>
    >>> # Create a custom random state
    >>> rng = bsr.RandomState(seed=123)
    >>>
    >>> # Use it for generation
    >>> data = rng.normal(0, 1, size=(10, 10))
    >>>
    >>> # Get the current key
    >>> current_key = rng.value

Available Distributions
-----------------------

The module provides a wide range of probability distributions:

**Uniform Distributions:**

- `rand`, `random`, `random_sample`, `ranf`, `sample` - Uniform [0, 1)
- `randint`, `random_integers` - Uniform integers
- `choice` - Random selection from array
- `permutation`, `shuffle` - Random ordering

**Normal Distributions:**

- `randn`, `normal` - Normal (Gaussian) distribution
- `standard_normal` - Standard normal distribution
- `multivariate_normal` - Multivariate normal distribution
- `truncated_normal` - Truncated normal distribution

**Other Continuous Distributions:**

- `beta` - Beta distribution
- `exponential`, `standard_exponential` - Exponential distribution
- `gamma`, `standard_gamma` - Gamma distribution
- `gumbel` - Gumbel distribution
- `laplace` - Laplace distribution
- `logistic` - Logistic distribution
- `pareto` - Pareto distribution
- `rayleigh` - Rayleigh distribution
- `standard_cauchy` - Cauchy distribution
- `standard_t` - Student's t-distribution
- `uniform` - Uniform distribution over [low, high)
- `weibull` - Weibull distribution

**Discrete Distributions:**

- `bernoulli` - Bernoulli distribution
- `binomial` - Binomial distribution
- `poisson` - Poisson distribution

Seed Management
---------------

The module provides utilities for managing random seeds:

.. code-block:: python

    >>> import brainstate.random as bsr
    >>>
    >>> # Set a global seed
    >>> bsr.seed(42)
    >>>
    >>> # Get current seed/key
    >>> key = bsr.get_key()
    >>>
    >>> # Split the key for parallel operations
    >>> keys = bsr.split_key(n=4)
    >>>
    >>> # Use context manager for temporary seed
    >>> with bsr.local_seed(123):
    ...     x = bsr.normal(0, 1, (5,))  # Uses seed 123
    >>> y = bsr.normal(0, 1, (5,))  # Uses original seed

Examples
--------

**Basic random number generation:**

.. code-block:: python

    >>> import brainstate.random as bsr
    >>> import jax.numpy as jnp
    >>>
    >>> # Set seed for reproducibility
    >>> bsr.seed(0)
    >>>
    >>> # Generate uniform random numbers
    >>> uniform_data = bsr.random((3, 3))
    >>> print(uniform_data.shape)
    (3, 3)
    >>>
    >>> # Generate normal random numbers
    >>> normal_data = bsr.normal(loc=0, scale=1, size=(100,))
    >>> print(f"Mean: {normal_data.mean():.3f}, Std: {normal_data.std():.3f}")
    Mean: -0.045, Std: 0.972

**Sampling and shuffling:**

.. code-block:: python

    >>> import brainstate.random as bsr
    >>> import jax.numpy as jnp
    >>>
    >>> bsr.seed(42)
    >>>
    >>> # Random choice from array
    >>> arr = jnp.array([1, 2, 3, 4, 5])
    >>> samples = bsr.choice(arr, size=3, replace=False)
    >>> print(samples)
    [4 1 5]
    >>>
    >>> # Random permutation
    >>> perm = bsr.permutation(10)
    >>> print(perm)
    [3 5 1 7 9 0 2 8 4 6]
    >>>
    >>> # In-place shuffle
    >>> data = jnp.arange(5)
    >>> bsr.shuffle(data)
    >>> print(data)
    [2 0 4 1 3]

**Advanced distributions:**

.. code-block:: python

    >>> import brainstate.random as bsr
    >>> import matplotlib.pyplot as plt
    >>>
    >>> bsr.seed(123)
    >>>
    >>> # Generate samples from different distributions
    >>> normal_samples = bsr.normal(0, 1, 1000)
    >>> exponential_samples = bsr.exponential(1.0, 1000)
    >>> beta_samples = bsr.beta(2, 5, 1000)
    >>>
    >>> # Plot histograms
    >>> fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    >>> axes[0].hist(normal_samples, bins=30, density=True)
    >>> axes[0].set_title('Normal Distribution')
    >>> axes[1].hist(exponential_samples, bins=30, density=True)
    >>> axes[1].set_title('Exponential Distribution')
    >>> axes[2].hist(beta_samples, bins=30, density=True)
    >>> axes[2].set_title('Beta Distribution')
    >>> plt.show()

**Using with neural network simulations:**

.. code-block:: python

    >>> import brainstate as bs
    >>> import brainstate.random as bsr
    >>> import brainstate.nn as nn
    >>>
    >>> class NoisyNeuron(bs.Module):
    ...     def __init__(self, n_neurons, noise_scale=0.1):
    ...         super().__init__()
    ...         self.n_neurons = n_neurons
    ...         self.noise_scale = noise_scale
    ...         self.membrane = bs.State(jnp.zeros(n_neurons))
    ...
    ...     def update(self, input_current):
    ...         # Add noise to input current
    ...         noise = bsr.normal(0, self.noise_scale, self.n_neurons)
    ...         self.membrane.value += input_current + noise
    ...         return self.membrane.value
    >>>
    >>> # Create and run noisy neuron model
    >>> bsr.seed(42)
    >>> neuron = NoisyNeuron(100)
    >>> output = neuron.update(jnp.ones(100) * 0.5)

Notes
-----

- This module is designed to work seamlessly with JAX's functional programming model
- Random functions are JIT-compilable for optimal performance
- The global DEFAULT state is thread-local to avoid race conditions
- For deterministic results, always set a seed before random operations

See Also
--------

jax.random : JAX's random number generation module
numpy.random : NumPy's random number generation module
RandomState : The stateful random number generator class

References
----------
.. [1] JAX Random Number Generation:
   https://jax.readthedocs.io/en/latest/jax.random.html
.. [2] NumPy Random Sampling:
   https://numpy.org/doc/stable/reference/random/index.html

"""

from ._fun import *
from ._fun import __all__ as __all_random__
from ._seed import *
from ._seed import __all__ as __all_seed__
from ._state import *
from ._state import __all__ as __all_state__

__all__ = __all_random__ + __all_state__ + __all_seed__
del __all_random__, __all_state__, __all_seed__
