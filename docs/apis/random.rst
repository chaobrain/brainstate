``brainstate.random`` module
============================

.. currentmodule:: brainstate.random
.. automodule:: brainstate.random


Random State Management
-----------------------

Core components for managing random number generator state and ensuring reproducible computations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    RandomState

Seed and Key Management
~~~~~~~~~~~~~~~~~~~~~~~

Functions for controlling the global random state and creating independent random number generators.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   seed
   seed_context
   default_rng
   clone_rng
   set_key
   get_key
   restore_key

Key Splitting and Parallel Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for creating independent random keys for parallel computation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   split_key
   split_keys
   self_assign_multi_keys


Random Sampling Functions
-------------------------

Comprehensive collection of probability distributions and sampling functions, providing
NumPy-compatible interfaces with JAX backend acceleration.

Basic Random Sampling
~~~~~~~~~~~~~~~~~~~~~

Fundamental random number generation functions for common use cases.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   rand
   randn
   random
   random_sample
   ranf
   sample
   randint
   random_integers

Array-like Generation (PyTorch compatibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions that generate random arrays with shapes matching existing arrays.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   rand_like
   randint_like
   randn_like

Array Manipulation
~~~~~~~~~~~~~~~~~~

Functions for random permutations and selections.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   choice
   permutation
   shuffle

Continuous Distributions
~~~~~~~~~~~~~~~~~~~~~~~~

Probability distributions for continuous random variables.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   beta
   exponential
   gamma
   gumbel
   laplace
   logistic
   normal
   pareto
   standard_cauchy
   standard_exponential
   standard_gamma
   standard_normal
   standard_t
   uniform
   truncated_normal
   lognormal
   power
   rayleigh
   triangular
   vonmises
   wald
   weibull
   weibull_min
   maxwell
   t
   loggamma

Discrete Distributions
~~~~~~~~~~~~~~~~~~~~~~

Probability distributions for discrete random variables.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   bernoulli
   binomial
   categorical
   geometric
   hypergeometric
   logseries
   multinomial
   negative_binomial
   poisson
   zipf

Special Distributions
~~~~~~~~~~~~~~~~~~~~~

Specialized distributions for statistical and scientific applications.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   chisquare
   dirichlet
   f
   multivariate_normal
   noncentral_chisquare
   noncentral_f
   orthogonal

