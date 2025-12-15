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


# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np

from brainstate._utils import set_module_as
from brainstate.typing import DTypeLike, Size, SeedOrKey
from ._state import RandomState, DEFAULT

__all__ = [
    # numpy compatibility
    'rand',
    'randint',
    'random_integers',
    'randn',
    'random',
    'random_sample',
    'ranf',
    'sample',
    'choice',
    'permutation',
    'shuffle',
    'beta',
    'exponential',
    'gamma',
    'gumbel',
    'laplace',
    'logistic',
    'normal',
    'pareto',
    'poisson',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'uniform',
    'truncated_normal',
    'bernoulli',
    'lognormal',
    'binomial',
    'chisquare',
    'dirichlet',
    'geometric',
    'f',
    'hypergeometric',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'power',
    'rayleigh',
    'triangular',
    'vonmises',
    'wald',
    'weibull',
    'weibull_min',
    'zipf',
    'maxwell',
    't',
    'orthogonal',
    'loggamma',
    'categorical',

    # pytorch compatibility
    'rand_like',
    'randint_like',
    'randn_like',
]


@set_module_as('brainstate.random')
def rand(
    *dn,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Random values in a given shape.

    Create an array of the given shape and populate it with
    random samples from a uniform distribution
    over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.
    dtype : dtype, optional
        Desired dtype of the result. Byteorder must be native.
        The default value is float.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray, shape ``(d0, d1, ..., dn)``
        Random values.

    See Also
    --------
    random

    Examples
    --------
    Generate random values in a 3x2 array:

    .. code-block:: python

        >>> import brainstate
        >>> arr = brainstate.random.rand(3, 2)
        >>> print(arr.shape)  # (3, 2)
        >>> print((arr >= 0).all() and (arr < 1).all())  # True
    """
    return DEFAULT.rand(*dn, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def randint(
    low,
    high=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int or array-like of ints, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
        If array-like, must contain integer values
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.
    dtype : dtype, optional
        Desired dtype of the result. Byteorder must be native.
        The default value is int.

    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    random_integers : similar to `randint`, only for the closed
        interval [`low`, `high`], and 1 is the lowest value if `high` is
        omitted.
    Generator.integers: which should be used for new code.

    Examples
    --------
    Generate 10 random integers from 0 to 1 (exclusive):

    .. code-block:: python

        >>> import brainstate
        >>> arr = brainstate.random.randint(2, size=10)
        >>> print(arr.shape)  # (10,)
        >>> print((arr >= 0).all() and (arr < 2).all())  # True

    Generate a 2x4 array of integers from 0 to 4 (exclusive):

    .. code-block:: python

        >>> arr = brainstate.random.randint(5, size=(2, 4))
        >>> print(arr.shape)  # (2, 4)
        >>> print((arr >= 0).all() and (arr < 5).all())  # True

    Generate integers with different upper bounds using broadcasting:

    .. code-block:: python

        >>> arr = brainstate.random.randint(1, [3, 5, 10])
        >>> print(arr.shape)  # (3,)

    Generate integers with different lower bounds:

    .. code-block:: python

        >>> arr = brainstate.random.randint([1, 5, 7], 10)
        >>> print(arr.shape)  # (3,)
        >>> print((arr >= [1, 5, 7]).all())  # True
    """

    return DEFAULT.randint(low, high, size=size, dtype=dtype, key=key)


@set_module_as('brainstate.random')
def random_integers(
    low,
    high=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Random integers of type `np.int_` between `low` and `high`, inclusive.

    Return random integers of type `np.int_` from the "discrete uniform"
    distribution in the closed interval [`low`, `high`].  If `high` is
    None (the default), then results are from [1, `low`]. The `np.int_`
    type translates to the C long integer type and its precision
    is platform dependent.

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int, optional
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    randint : Similar to `random_integers`, only for the half-open
        interval [`low`, `high`), and 0 is the lowest value if `high` is
        omitted.

    Notes
    -----
    To sample from N evenly spaced floating-point numbers between a and b,
    use::

      a + (b - a) * (brainstate.random.random_integers(N) - 1) / (N - 1.)

    Examples
    --------
    Generate a single random integer from 1 to 5 (inclusive):

    .. code-block:: python

        >>> import brainstate
        >>> val = brainstate.random.random_integers(5)
        >>> print(type(val))  # <class 'numpy.int64'>
        >>> print(1 <= val <= 5)  # True

    Generate a 3x2 array of random integers from 1 to 5 (inclusive):

    .. code-block:: python

        >>> arr = brainstate.random.random_integers(5, size=(3, 2))
        >>> print(arr.shape)  # (3, 2)
        >>> print((arr >= 1).all() and (arr <= 5).all())  # True

    Choose five random numbers from the set of five evenly-spaced
    numbers between 0 and 2.5, inclusive (*i.e.*, from the set
    :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

    .. code-block:: python

        >>> vals = 2.5 * (brainstate.random.random_integers(5, size=(5,)) - 1) / 4.
        >>> print(vals.shape)  # (5,)

    Roll two six sided dice 1000 times and sum the results:

    .. code-block:: python

        >>> d1 = brainstate.random.random_integers(1, 6, 1000)
        >>> d2 = brainstate.random.random_integers(1, 6, 1000)
        >>> dsums = d1 + d2
        >>> print(dsums.shape)  # (1000,)
        >>> print((dsums >= 2).all() and (dsums <= 12).all())  # True
    """

    return DEFAULT.random_integers(low, high, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def randn(
    *dn,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Return a sample (or samples) from the "standard normal" distribution.

    If positive int_like arguments are provided, `randn` generates an array
    of shape ``(d0, d1, ..., dn)``, filled
    with random floats sampled from a univariate "normal" (Gaussian)
    distribution of mean 0 and variance 1. A single float randomly sampled
    from the distribution is returned if no argument is provided.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
        The dimensions of the returned array, must be non-negative.
        If no argument is given a single Python float is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    Z : ndarray or float
        A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
        the standard normal distribution, or a single such float if
        no parameters were supplied.

    See Also
    --------
    standard_normal : Similar, but takes a tuple as its argument.
    normal : Also accepts mu and sigma arguments.

    Notes
    -----
    For random samples from :math:`N(\mu, \sigma^2)`, use:

    ``sigma * brainstate.random.randn(...) + mu``

    Examples
    --------
    Generate a single random number from standard normal distribution:

    .. code-block:: python

        >>> import brainstate
        >>> val = brainstate.random.randn()
        >>> print(type(val))  # <class 'numpy.float64'>

    Generate a 2x4 array of standard normal samples:

    .. code-block:: python

        >>> arr = brainstate.random.randn(2, 4)
        >>> print(arr.shape)  # (2, 4)

    Two-by-four array of samples from N(3, 6.25):

    .. code-block:: python

        >>> arr = 3 + 2.5 * brainstate.random.randn(2, 4)
        >>> print(arr.shape)  # (2, 4)
    """

    return DEFAULT.randn(*dn, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def random(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Return random floats in the half-open interval [0.0, 1.0). Alias for
    `random_sample` to ease forward-porting to the new random API.
    """
    return DEFAULT.random(size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def random_sample(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Return random floats in the half-open interval [0.0, 1.0).

    Results are from the "continuous uniform" distribution over the
    stated interval.  To sample :math:`Unif[a, b), b > a` multiply
    the output of `random_sample` by `(b-a)` and add `a`::

      (b - a) * random_sample() + a

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : float or ndarray of floats
        Array of random floats of shape `size` (unless ``size=None``, in which
        case a single float is returned).

    See Also
    --------
    Generator.random: which should be used for new code.

    Examples
    --------
    Generate a single random float:

    .. code-block:: python

        >>> import brainstate
        >>> val = brainstate.random.random_sample()
        >>> print(type(val))  # <class 'float'>
        >>> print(0.0 <= val < 1.0)  # True

    Generate an array of 5 random floats:

    .. code-block:: python

        >>> arr = brainstate.random.random_sample((5,))
        >>> print(arr.shape)  # (5,)
        >>> print((arr >= 0.0).all() and (arr < 1.0).all())  # True

    Three-by-two array of random numbers from [-5, 0):

    .. code-block:: python

        >>> arr = 5 * brainstate.random.random_sample((3, 2)) - 5
        >>> print(arr.shape)  # (3, 2)
        print((arr >= -5.0).all() and (arr < 0.0).all())  # True
    """
    return DEFAULT.random_sample(size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def ranf(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    This is an alias of `random_sample`. See `random_sample`  for the complete
    documentation.
    """
    return DEFAULT.ranf(size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def sample(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    """
    This is an alias of `random_sample`. See `random_sample`  for the complete
    documentation.
    """
    return DEFAULT.sample(size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def choice(
    a,
    size: Optional[Size] = None,
    replace=True,
    p=None,
    key: Optional[SeedOrKey] = None
):
    r"""
    Generates a random sample from a given 1-D array

    Parameters
    ----------
    a : 1-D array-like or int
        If an ndarray, a random sample is generated from its elements.
        If an int, the random sample is generated as if it were ``np.arange(a)``
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement. Default is True,
        meaning that a value of ``a`` can be selected multiple times.
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given, the sample assumes a uniform distribution over all
        entries in ``a``.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    samples : single item or ndarray
        The generated random samples

    Raises
    ------
    ValueError
        If a is an int and less than zero, if a or p are not 1-dimensional,
        if a is an array-like of size 0, if p is not a vector of
        probabilities, if a and p have different lengths, or if
        replace=False and the sample size is greater than the population
        size

    See Also
    --------
    randint, shuffle, permutation
    Generator.choice: which should be used in new code

    Notes
    -----
    Setting user-specified probabilities through ``p`` uses a more general but less
    efficient sampler than the default. The general sampler produces a different sample
    than the optimized sampler even if each element of ``p`` is 1 / len(a).

    Sampling random rows from a 2-D array is not possible with this function,
    but is possible with `Generator.choice` through its ``axis`` keyword.

    Examples
    --------
    Generate a uniform random sample from np.arange(5) of size 3:

    .. code-block:: python

        >>> import brainstate
        >>> result = brainstate.random.choice(5, 3)
        >>> print(result.shape)  # (3,)
        >>> print((result >= 0).all() and (result < 5).all())  # True

    Generate a non-uniform random sample from np.arange(5) of size 3:

    .. code-block:: python

        >>> result = brainstate.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        >>> print(result.shape)  # (3,)
        >>> print(set(result).issubset({0, 2, 3}))  # True (only non-zero prob elements)

    Generate a uniform random sample from np.arange(5) of size 3 without replacement:

    .. code-block:: python

        >>> result = brainstate.random.choice(5, 3, replace=False)
        >>> print(result.shape)  # (3,)
        >>> print(len(set(result)) == 3)  # True (all unique)

    Generate a non-uniform random sample from np.arange(5) of size 3 without replacement:

    .. code-block:: python

        >>> result = brainstate.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        >>> print(result.shape)  # (3,)
        >>> print(len(set(result)) == 3)  # True (all unique)

    Any of the above can be repeated with an arbitrary array-like instead of just integers:

    .. code-block:: python

        >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
        >>> result = brainstate.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
        >>> print(result.shape)  # (5,)
        >>> print(result.dtype.kind)  # 'U' (unicode string)
    """
    a = a
    return DEFAULT.choice(a, size=size, replace=replace, p=p, key=key)


@set_module_as('brainstate.random')
def permutation(
    x,
    axis: int = 0,
    independent: bool = False,
    key: Optional[SeedOrKey] = None
):
    r"""
    Randomly permute a sequence, or return a permuted range.

    If `x` is a multi-dimensional array, it is only shuffled along its
    first index.

    Parameters
    ----------
    x : int or array_like
        If `x` is an integer, randomly permute ``np.arange(x)``.
        If `x` is an array, make a copy and shuffle the elements
        randomly.
    axis : int, optional
        The axis which `x` is shuffled along. Default is 0.
    independent : bool, optional
        Whether to use independent random permutations for each
        batch. If ``False`` (default), the same random permutation is
        used for all batches. If ``True``, each batch is shuffled
        independently. Ignored if `x` is an integer.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray
        Permuted sequence or array range.

    Examples
    --------
    Permute integers from 0 to 9:

    .. code-block:: python

        >>> import brainstate
        >>> result = brainstate.random.permutation(10)
        >>> print(result.shape)  # (10,)
        >>> print(sorted(result))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Permute a given array:

    .. code-block:: python

        >>> arr = [1, 4, 9, 12, 15]
        >>> result = brainstate.random.permutation(arr)
        >>> print(result.shape)  # (5,)
        >>> print(sorted(result))  # [1, 4, 9, 12, 15]

    Permute rows of a 2D array:

    .. code-block:: python

        >>> import numpy as np
        >>> arr = np.arange(9).reshape((3, 3))
        >>> result = brainstate.random.permutation(arr)
        >>> print(result.shape)  # (3, 3)
        >>> print(result.flatten().sort() == np.arange(9).sort())  # True
    """
    return DEFAULT.permutation(x, axis=axis, independent=independent, key=key)


@set_module_as('brainstate.random')
def shuffle(
    x,
    axis: int = 0,
    key: Optional[SeedOrKey] = None
):
    r"""
    Modify a sequence in-place by shuffling its contents.

    This function only shuffles the array along the first axis of a
    multi-dimensional array. The order of sub-arrays is changed but
    their contents remains the same.

    Parameters
    ----------
    x : ndarray or MutableSequence
        The array, list or mutable sequence to be shuffled.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    None

    Examples
    --------
    Shuffle a 1D array in-place:

    .. code-block:: python

        >>> import brainstate
        >>> import numpy as np
        >>> arr = np.arange(10)
        >>> original_elements = set(arr)
        >>> brainstate.random.shuffle(arr)
        >>> print(set(arr) == original_elements)  # True (same elements)

    Multi-dimensional arrays are only shuffled along the first axis:

    .. code-block:: python

        >>> arr = np.arange(9).reshape((3, 3))
        >>> original_shape = arr.shape
        >>> brainstate.random.shuffle(arr)
        >>> print(arr.shape == original_shape)  # True (shape preserved)
        >>> print(sorted(arr.flatten()) == list(range(9)))  # True (same elements)
    """
    return DEFAULT.shuffle(x, axis=axis, key=key)


@set_module_as('brainstate.random')
def beta(
    a,
    b,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Beta distribution.

    The Beta distribution is a special case of the Dirichlet distribution,
    and is related to the Gamma distribution.  It has the probability
    distribution function

    .. math:: f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
                                                     (1 - x)^{\beta - 1},

    where the normalization, B, is the beta function,

    .. math:: B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                                 (1 - t)^{\beta - 1} dt.

    It is often seen in Bayesian inference and order statistics.

    Parameters
    ----------
    a : float or array_like of floats
        Alpha, positive (>0).
    b : float or array_like of floats
        Beta, positive (>0).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` and ``b`` are both scalars.
        Otherwise, ``np.broadcast(a, b).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized beta distribution.
    """
    return DEFAULT.beta(a, b, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def exponential(
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from an exponential distribution.

    Its probability density function is

    .. math:: f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}),

    for ``x > 0`` and 0 elsewhere. :math:`\beta` is the scale parameter,
    which is the inverse of the rate parameter :math:`\lambda = 1/\beta`.
    The rate parameter is an alternative, widely used parameterization
    of the exponential distribution [3]_.

    The exponential distribution is a continuous analogue of the
    geometric distribution.  It describes many common situations, such as
    the size of raindrops measured over many rainstorms [1]_, or the time
    between page requests to Wikipedia [2]_.

    Parameters
    ----------
    scale : float or array_like of floats
        The scale parameter, :math:`\beta = 1/\lambda`. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.  Otherwise,
        ``np.array(scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized exponential distribution.

    References
    ----------
    .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
           Random Signal Principles", 4th ed, 2001, p. 57.
    .. [2] Wikipedia, "Poisson process",
           https://en.wikipedia.org/wiki/Poisson_process
    .. [3] Wikipedia, "Exponential distribution",
           https://en.wikipedia.org/wiki/Exponential_distribution
    """
    return DEFAULT.exponential(scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def gamma(
    shape,
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    `shape` (sometimes designated "k") and `scale` (sometimes designated
    "theta"), where both parameters are > 0.

    Parameters
    ----------
    shape : float or array_like of floats
        The shape of the gamma distribution. Must be non-negative.
    scale : float or array_like of floats, optional
        The scale of the gamma distribution. Must be non-negative.
        Default is equal to 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(shape, scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized gamma distribution.


    Notes
    -----
    The probability density for the Gamma distribution is

    .. math:: p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},

    where :math:`k` is the shape and :math:`\theta` the scale,
    and :math:`\Gamma` is the Gamma function.

    The Gamma distribution is often used to model the times to failure of
    electronic components, and arises naturally in processes for which the
    waiting times between Poisson distributed events are relevant.

    References
    ----------
    .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/GammaDistribution.html
    .. [2] Wikipedia, "Gamma distribution",
           https://en.wikipedia.org/wiki/Gamma_distribution

    """
    return DEFAULT.gamma(shape, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def gumbel(
    loc=None,
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Gumbel distribution.

    Draw samples from a Gumbel distribution with specified location and
    scale.  For more information on the Gumbel distribution, see
    Notes and References below.

    Parameters
    ----------
    loc : float or array_like of floats, optional
        The location of the mode of the distribution. Default is 0.
    scale : float or array_like of floats, optional
        The scale parameter of the distribution. Default is 1. Must be non-
        negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Gumbel distribution.

    Notes
    -----
    The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
    Value Type I) distribution is one of a class of Generalized Extreme
    Value (GEV) distributions used in modeling extreme value problems.
    The Gumbel is a special case of the Extreme Value Type I distribution
    for maximums from distributions with "exponential-like" tails.

    The probability density for the Gumbel distribution is

    .. math:: p(x) = \frac{e^{-(x - \mu)/ \beta}}{\beta} e^{ -e^{-(x - \mu)/
              \beta}},

    where :math:`\mu` is the mode, a location parameter, and
    :math:`\beta` is the scale parameter.

    The Gumbel (named for German mathematician Emil Julius Gumbel) was used
    very early in the hydrology literature, for modeling the occurrence of
    flood events. It is also used for modeling maximum wind speed and
    rainfall rates.  It is a "fat-tailed" distribution - the probability of
    an event in the tail of the distribution is larger than if one used a
    Gaussian, hence the surprisingly frequent occurrence of 100-year
    floods. Floods were initially modeled as a Gaussian process, which
    underestimated the frequency of extreme events.

    It is one of a class of extreme value distributions, the Generalized
    Extreme Value (GEV) distributions, which also includes the Weibull and
    Frechet.

    The function has a mean of :math:`\mu + 0.57721\beta` and a variance
    of :math:`\frac{\pi^2}{6}\beta^2`.

    References
    ----------
    .. [1] Gumbel, E. J., "Statistics of Extremes,"
           New York: Columbia University Press, 1958.
    .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
           Values from Insurance, Finance, Hydrology and Other Fields,"
           Basel: Birkhauser Verlag, 2001.
    """
    return DEFAULT.gumbel(loc, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def laplace(
    loc=None,
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).

    The Laplace distribution is similar to the Gaussian/normal distribution,
    but is sharper at the peak and has fatter tails. It represents the
    difference between two independent, identically distributed exponential
    random variables.

    Parameters
    ----------
    loc : float or array_like of floats, optional
        The position, :math:`\mu`, of the distribution peak. Default is 0.
    scale : float or array_like of floats, optional
        :math:`\lambda`, the exponential decay. Default is 1. Must be non-
        negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Laplace distribution.

    Notes
    -----
    It has the probability density function

    .. math:: f(x; \mu, \lambda) = \frac{1}{2\lambda}
                                   \exp\left(-\frac{|x - \mu|}{\lambda}\right).

    The first law of Laplace, from 1774, states that the frequency
    of an error can be expressed as an exponential function of the
    absolute magnitude of the error, which leads to the Laplace
    distribution. For many problems in economics and health
    sciences, this distribution seems to model the data better
    than the standard Gaussian distribution.

    References
    ----------
    .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
           Mathematical Functions with Formulas, Graphs, and Mathematical
           Tables, 9th printing," New York: Dover, 1972.
    .. [2] Kotz, Samuel, et. al. "The Laplace Distribution and
           Generalizations, " Birkhauser, 2001.
    .. [3] Weisstein, Eric W. "Laplace Distribution."
           From MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/LaplaceDistribution.html
    .. [4] Wikipedia, "Laplace distribution",
           https://en.wikipedia.org/wiki/Laplace_distribution

    Examples
    --------
    Draw samples from the distribution

    >>> loc, scale = 0., 1.
    >>> s = brainstate.random.laplace(loc, scale, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt  # noqa  # noqa
    >>> count, bins, ignored = plt.hist(s, 30, density=True)
    >>> x = np.arange(-8., 8., .01)
    >>> pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
    >>> plt.plot(x, pdf)

    Plot Gaussian for comparison:

    >>> g = (1/(scale * np.sqrt(2 * np.pi)) *
    ...      np.exp(-(x - loc)**2 / (2 * scale**2)))
    >>> plt.plot(x,g)
    """
    return DEFAULT.laplace(loc, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def logistic(
    loc=None,
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a logistic distribution.

    Samples are drawn from a logistic distribution with specified
    parameters, loc (location or mean, also median), and scale (>0).

    Parameters
    ----------
    loc : float or array_like of floats, optional
        Parameter of the distribution. Default is 0.
    scale : float or array_like of floats, optional
        Parameter of the distribution. Must be non-negative.
        Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized logistic distribution.

    Notes
    -----
    The probability density for the Logistic distribution is

    .. math:: P(x) = P(x) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2},

    where :math:`\mu` = location and :math:`s` = scale.

    The Logistic distribution is used in Extreme Value problems where it
    can act as a mixture of Gumbel distributions, in Epidemiology, and by
    the World Chess Federation (FIDE) where it is used in the Elo ranking
    system, assuming the performance of each player is a logistically
    distributed random variable.

    References
    ----------
    .. [1] Reiss, R.-D. and Thomas M. (2001), "Statistical Analysis of
           Extreme Values, from Insurance, Finance, Hydrology and Other
           Fields," Birkhauser Verlag, Basel, pp 132-133.
    .. [2] Weisstein, Eric W. "Logistic Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/LogisticDistribution.html
    .. [3] Wikipedia, "Logistic-distribution",
           https://en.wikipedia.org/wiki/Logistic_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> loc, scale = 10, 1
    >>> s = brainstate.random.logistic(loc, scale, 10000)
    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, ignored = plt.hist(s, bins=50)

    #   plot against distribution

    >>> def logist(x, loc, scale):
    ...     return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
    >>> lgst_val = logist(bins, loc, scale)
    >>> plt.plot(bins, lgst_val * count.max() / lgst_val.max())
    >>> plt.show()
    """
    return DEFAULT.logistic(loc, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def normal(
    loc=None,
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw random samples from a normal (Gaussian) distribution.

    The probability density function of the normal distribution, first
    derived by De Moivre and 200 years later by both Gauss and Laplace
    independently [2]_, is often called the bell curve because of
    its characteristic shape (see the example below).

    The normal distributions occurs often in nature.  For example, it
    describes the commonly occurring distribution of samples influenced
    by a large number of tiny, random disturbances, each with its own
    unique distribution [2]_.

    Parameters
    ----------
    loc : float or array_like of floats
        Mean ("centre") of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    The probability density for the Gaussian distribution is

    .. math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                     e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },

    where :math:`\mu` is the mean and :math:`\sigma` the standard
    deviation. The square of the standard deviation, :math:`\sigma^2`,
    is called the variance.

    The function has its peak at the mean, and its "spread" increases with
    the standard deviation (the function reaches 0.607 times its maximum at
    :math:`x + \sigma` and :math:`x - \sigma` [2]_).  This implies that
    normal is more likely to return samples lying close to the mean, rather
    than those far away.

    References
    ----------
    .. [1] Wikipedia, "Normal distribution",
           https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001,
           pp. 51, 51, 125.

    Examples
    --------
    Draw samples from the distribution:

    .. code-block:: python

        >>> import brainstate
        >>> import numpy as np
        >>> mu, sigma = 0, 0.1  # mean and standard deviation
        >>> s = brainstate.random.normal(mu, sigma, 1000)
        >>> print(s.shape)  # (1000,)
        >>> print(abs(mu - np.mean(s)) < 0.1)  # True (approximately correct mean)
        >>> print(abs(sigma - np.std(s, ddof=1)) < 0.1)  # True (approximately correct std)

    Two-by-four array of samples from the normal distribution with
    mean 3 and standard deviation 2.5:

    .. code-block:: python

        >>> samples = brainstate.random.normal(3, 2.5, size=(2, 4))
        >>> print(samples.shape)  # (2, 4)
    """
    return DEFAULT.normal(loc, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def pareto(
    a,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Pareto II or Lomax distribution with
    specified shape.

    The Lomax or Pareto II distribution is a shifted Pareto
    distribution. The classical Pareto distribution can be
    obtained from the Lomax distribution by adding 1 and
    multiplying by the scale parameter ``m`` (see Notes).  The
    smallest value of the Lomax distribution is zero while for the
    classical Pareto distribution it is ``mu``, where the standard
    Pareto distribution has location ``mu = 1``.  Lomax can also
    be considered as a simplified version of the Generalized
    Pareto distribution (available in SciPy), with the scale set
    to one and the location set to zero.

    The Pareto distribution must be greater than zero, and is
    unbounded above.  It is also known as the "80-20 rule".  In
    this distribution, 80 percent of the weights are in the lowest
    20 percent of the range, while the other 20 percent fill the
    remaining 80 percent of the range.

    Parameters
    ----------
    a : float or array_like of floats
        Shape of the distribution. Must be positive.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar.  Otherwise,
        ``np.array(a).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Pareto distribution.

    See Also
    --------
    scipy.stats.lomax : probability density function, distribution or
        cumulative density function, etc.
    scipy.stats.genpareto : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Pareto distribution is

    .. math:: p(x) = \frac{am^a}{x^{a+1}}

    where :math:`a` is the shape and :math:`m` the scale.

    The Pareto distribution, named after the Italian economist
    Vilfredo Pareto, is a power law probability distribution
    useful in many real world problems.  Outside the field of
    economics it is generally referred to as the Bradford
    distribution. Pareto developed the distribution to describe
    the distribution of wealth in an economy.  It has also found
    use in insurance, web page access statistics, oil field sizes,
    and many other problems, including the download frequency for
    projects in Sourceforge [1]_.  It is one of the so-called
    "fat-tailed" distributions.

    References
    ----------
    .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
           Sourceforge projects.
    .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
    .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
           Values, Birkhauser Verlag, Basel, pp 23-30.
    .. [4] Wikipedia, "Pareto distribution",
           https://en.wikipedia.org/wiki/Pareto_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> a, m = 3., 2.  # shape and mode
    >>> s = (brainstate.random.pareto(a, 1000) + 1) * m

    Display the histogram of the samples, along with the probability
    density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, _ = plt.hist(s, 100, density=True)
    >>> fit = a*m**a / bins**(a+1)
    >>> plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
    >>> plt.show()
    """
    return DEFAULT.pareto(a, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def poisson(
    lam=1.0,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Poisson distribution.

    The Poisson distribution is the limit of the binomial distribution
    for large N.

    Parameters
    ----------
    lam : float or array_like of floats
        Expected number of events occurring in a fixed-time interval,
        must be >= 0. A sequence must be broadcastable over the requested
        size.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``lam`` is a scalar. Otherwise,
        ``np.array(lam).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Poisson distribution.

    Notes
    -----
    The Poisson distribution

    .. math:: f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}

    For events with an expected separation :math:`\lambda` the Poisson
    distribution :math:`f(k; \lambda)` describes the probability of
    :math:`k` events occurring within the observed
    interval :math:`\lambda`.

    Because the output is limited to the range of the C int64 type, a
    ValueError is raised when `lam` is within 10 sigma of the maximum
    representable value.

    References
    ----------
    .. [1] Weisstein, Eric W. "Poisson Distribution."
           From MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/PoissonDistribution.html
    .. [2] Wikipedia, "Poisson distribution",
           https://en.wikipedia.org/wiki/Poisson_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import numpy as np
    >>> s = brainstate.random.poisson(5, 10000)

    Display histogram of the sample:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, ignored = plt.hist(s, 14, density=True)
    >>> plt.show()

    Draw each 100 values for lambda 100 and 500:

    >>> s = brainstate.random.poisson(lam=(100., 500.), size=(100, 2))
    """
    return DEFAULT.poisson(lam, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def standard_cauchy(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a standard Cauchy distribution with mode = 0.

    Also known as the Lorentz distribution.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    samples : ndarray or scalar
        The drawn samples.

    Notes
    -----
    The probability density function for the full Cauchy distribution is

    .. math:: P(x; x_0, \gamma) = \frac{1}{\pi \gamma \bigl[ 1+
              (\frac{x-x_0}{\gamma})^2 \bigr] }

    and the Standard Cauchy distribution just sets :math:`x_0=0` and
    :math:`\gamma=1`

    The Cauchy distribution arises in the solution to the driven harmonic
    oscillator problem, and also describes spectral line broadening. It
    also describes the distribution of values at which a line tilted at
    a random angle will cut the x axis.

    When studying hypothesis tests that assume normality, seeing how the
    tests perform on data from a Cauchy distribution is a good indicator of
    their sensitivity to a heavy-tailed distribution, since the Cauchy looks
    very much like a Gaussian distribution, but with heavier tails.

    References
    ----------
    .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
          Distribution",
          https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
    .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
          Wolfram Web Resource.
          http://mathworld.wolfram.com/CauchyDistribution.html
    .. [3] Wikipedia, "Cauchy distribution"
          https://en.wikipedia.org/wiki/Cauchy_distribution

    Examples
    --------
    Draw samples and plot the distribution:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> s = brainstate.random.standard_cauchy(1000000)
    >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
    >>> plt.hist(s, bins=100)
    >>> plt.show()
    """
    return DEFAULT.standard_cauchy(size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def standard_exponential(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from the standard exponential distribution.

    `standard_exponential` is identical to the exponential distribution
    with a scale parameter of 1.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : float or ndarray
        Drawn samples.

    Examples
    --------
    Output a 3x8000 array:

    >>> n = brainstate.random.standard_exponential((3, 8000))
    """
    return DEFAULT.standard_exponential(size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def standard_gamma(
    shape,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a standard Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    shape (sometimes designated "k") and scale=1.

    Parameters
    ----------
    shape : float or array_like of floats
        Parameter, must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` is a scalar.  Otherwise,
        ``np.array(shape).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized standard gamma distribution.

    See Also
    --------
    scipy.stats.gamma : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Gamma distribution is

    .. math:: p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},

    where :math:`k` is the shape and :math:`\theta` the scale,
    and :math:`\Gamma` is the Gamma function.

    The Gamma distribution is often used to model the times to failure of
    electronic components, and arises naturally in processes for which the
    waiting times between Poisson distributed events are relevant.

    References
    ----------
    .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/GammaDistribution.html
    .. [2] Wikipedia, "Gamma distribution",
           https://en.wikipedia.org/wiki/Gamma_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> shape, scale = 2., 1. # mean and width
    >>> s = brainstate.random.standard_gamma(shape, 1000000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> import scipy.special as sps  # doctest: +SKIP
    >>> count, bins, ignored = plt.hist(s, 50, density=True)
    >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/  # doctest: +SKIP
    ...                       (sps.gamma(shape) * scale**shape))
    >>> plt.plot(bins, y, linewidth=2, color='r')  # doctest: +SKIP
    >>> plt.show()
    """
    return DEFAULT.standard_gamma(shape, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def standard_normal(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : float or ndarray
        A floating-point array of shape ``size`` of drawn samples, or a
        single sample if ``size`` was not specified.

    See Also
    --------
    normal :
        Equivalent function with additional ``loc`` and ``scale`` arguments
        for setting the mean and standard deviation.

    Notes
    -----
    For random samples from the normal distribution with mean ``mu`` and
    standard deviation ``sigma``, use one of::

        mu + sigma * brainstate.random.standard_normal(size=...)
        brainstate.random.normal(mu, sigma, size=...)

    Examples
    --------
    Generate a single standard normal sample:

    .. code-block:: python

        >>> import brainstate
        >>> val = brainstate.random.standard_normal()
        >>> print(type(val))  # <class 'numpy.float64'>

    Generate an array of 8000 standard normal samples:

    .. code-block:: python

        >>> s = brainstate.random.standard_normal(8000)
        >>> print(s.shape)  # (8000,)

    Generate a 3x4x2 array of standard normal samples:

    .. code-block:: python

        >>> s = brainstate.random.standard_normal(size=(3, 4, 2))
        >>> print(s.shape)  # (3, 4, 2)

    Two-by-four array of samples from the normal distribution with
    mean 3 and standard deviation 2.5:

    .. code-block:: python

        >>> samples = 3 + 2.5 * brainstate.random.standard_normal(size=(2, 4))
        print(samples.shape)  # (2, 4)
    """
    return DEFAULT.standard_normal(size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def standard_t(
    df,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a standard Student's t distribution with `df` degrees
    of freedom.

    A special case of the hyperbolic distribution.  As `df` gets
    large, the result resembles that of the standard normal
    distribution (`standard_normal`).

    Parameters
    ----------
    df : float or array_like of floats
        Degrees of freedom, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``df`` is a scalar.  Otherwise,
        ``np.array(df).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized standard Student's t distribution.

    Notes
    -----
    The probability density function for the t distribution is

    .. math:: P(x, df) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{\pi df}
              \Gamma(\frac{df}{2})}\Bigl( 1+\frac{x^2}{df} \Bigr)^{-(df+1)/2}

    The t test is based on an assumption that the data come from a
    Normal distribution. The t test provides a way to test whether
    the sample mean (that is the mean calculated from the data) is
    a good estimate of the true mean.

    The derivation of the t-distribution was first published in
    1908 by William Gosset while working for the Guinness Brewery
    in Dublin. Due to proprietary issues, he had to publish under
    a pseudonym, and so he used the name Student.

    References
    ----------
    .. [1] Dalgaard, Peter, "Introductory Statistics With R",
           Springer, 2002.
    .. [2] Wikipedia, "Student's t-distribution"
           https://en.wikipedia.org/wiki/Student's_t-distribution

    Examples
    --------
    From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
    women in kilojoules (kJ) is:

    >>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \
    ...                    7515, 8230, 8770])

    Does their energy intake deviate systematically from the recommended
    value of 7725 kJ? Our null hypothesis will be the absence of deviation,
    and the alternate hypothesis will be the presence of an effect that could be
    either positive or negative, hence making our test 2-tailed.

    Because we are estimating the mean and we have N=11 values in our sample,
    we have N-1=10 degrees of freedom. We set our significance level to 95% and
    compute the t statistic using the empirical mean and empirical standard
    deviation of our intake. We use a ddof of 1 to base the computation of our
    empirical standard deviation on an unbiased estimate of the variance (note:
    the final estimate is not unbiased due to the concave nature of the square
    root).

    >>> np.mean(intake)
    6753.636363636364
    >>> intake.std(ddof=1)
    1142.1232221373727
    >>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
    >>> t
    -2.8207540608310198

    We draw 1000000 samples from Student's t distribution with the adequate
    degrees of freedom.

    >>> import matplotlib.pyplot as plt  # noqa
    >>> s = brainstate.random.standard_t(10, size=1000000)
    >>> h = plt.hist(s, bins=100, density=True)

    Does our t statistic land in one of the two critical regions found at
    both tails of the distribution?

    >>> np.sum(np.abs(t) < np.abs(s)) / float(len(s))
    0.018318  #random < 0.05, statistic is in critical region

    The probability value for this 2-tailed test is about 1.83%, which is
    lower than the 5% pre-determined significance threshold.

    Therefore, the probability of observing values as extreme as our intake
    conditionally on the null hypothesis being true is too low, and we reject
    the null hypothesis of no deviation.
    """
    return DEFAULT.standard_t(df, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def uniform(
    low=0.0,
    high=1.0,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float or array_like of floats, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float or array_like of floats
        Upper boundary of the output interval.  All values generated will be
        less than or equal to high.  The high limit may be included in the
        returned array of floats due to floating-point rounding in the
        equation ``low + (high-low) * random_sample()``.  The default value
        is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``low`` and ``high`` are both scalars.
        Otherwise, ``np.broadcast(low, high).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    randint : Discrete uniform distribution, yielding integers.
    random_integers : Discrete uniform distribution over the closed
                      interval ``[low, high]``.
    random_sample : Floats uniformly distributed over ``[0, 1)``.
    random : Alias for `random_sample`.
    rand : Convenience function that accepts dimensions as input, e.g.,
           ``rand(2,2)`` would generate a 2-by-2 array of floats,
           uniformly distributed over ``[0, 1)``.

    Notes
    -----
    The probability density function of the uniform distribution is

    .. math:: p(x) = \frac{1}{b - a}

    anywhere within the interval ``[a, b)``, and zero elsewhere.

    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined
    and may eventually raise an error, i.e. do not rely on this
    function to behave when passed arguments satisfying that
    inequality condition. The ``high`` limit may be included in the
    returned array of floats due to floating-point rounding in the
    equation ``low + (high-low) * random_sample()``. For example:

    >>> x = np.float32(5*0.99999999)
    >>> x
    5.0


    Examples
    --------
    Draw samples from the distribution:

    >>> s = brainstate.random.uniform(-1,0,1000)

    All values are within the given interval:

    >>> np.all(s >= -1)
    True
    >>> np.all(s < 0)
    True

    Display the histogram of the samples, along with the
    probability density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, ignored = plt.hist(s, 15, density=True)
    >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    >>> plt.show()
    """
    return DEFAULT.uniform(low, high, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def truncated_normal(
    lower,
    upper,
    size: Optional[Size] = None,
    loc=0.0,
    scale=1.0,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None,
    check_valid: bool = True
):
    r"""Sample truncated standard normal random values with given shape and dtype.

    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf


    Notes
    -----
    This distribution is the normal distribution centered on ``loc`` (default
    0), with standard deviation ``scale`` (default 1), and clipped at ``a``,
    ``b`` standard deviations to the left, right (respectively) from ``loc``.
    If ``myclip_a`` and ``myclip_b`` are clip values in the sample space (as
    opposed to the number of standard deviations) then they can be converted
    to the required form according to::

        a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale


    Parameters
    ----------
    lower : float, ndarray
      A float or array of floats representing the lower bound for
      truncation. Must be broadcast-compatible with ``upper``.
    upper : float, ndarray
      A float or array of floats representing the  upper bound for
      truncation. Must be broadcast-compatible with ``lower``.
    loc : float, ndarray
      Mean ("centre") of the distribution before truncating. Note that
      the mean of the truncated distribution will not be exactly equal
      to ``loc``.
    size : optional, list of int, tuple of int
      A tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
      default (None) produces a result shape by broadcasting ``lower`` and
      ``upper``.
    loc: optional, float, ndarray
      A float or array of floats representing the mean of the
      distribution. Default is 0.
    scale : float, ndarray
      Standard deviation (spread or "width") of the distribution. Must be
      non-negative. Default is 1.
    dtype: optional
      The float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.
    check_valid: optional, bool
      Whether to check the validity of the input parameters. Default is True.

    Returns
    -------
    out : Array
      A random array with the specified dtype and shape given by ``shape`` if
      ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
      Returns values in the open interval ``(lower, upper)``.
    """
    return DEFAULT.truncated_normal(
        lower,
        upper,
        loc=loc,
        scale=scale,
        size=size,
        key=key,
        dtype=dtype,
        check_valid=check_valid,
    )


RandomState.truncated_normal.__doc__ = truncated_normal.__doc__


@set_module_as('brainstate.random')
def bernoulli(
    p=0.5,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    check_valid: bool = True,
):
    r"""Sample Bernoulli random values with given shape and mean.

    Parameters
    ----------
    p: float, array_like, optional
      A float or array of floats for the mean of the random
      variables. Must be broadcast-compatible with ``shape`` and the values
      should be within [0, 1]. Default 0.5.
    size: optional, tuple of int, int
      A tuple of nonnegative integers representing the result
      shape. Must be broadcast-compatible with ``p.shape``. The default (None)
      produces a result shape equal to ``p.shape``.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out: array_like
      A random array with boolean dtype and shape given by ``shape`` if ``shape``
      is not None, or else ``p.shape``.
    """
    return DEFAULT.bernoulli(p, size=size, key=key, check_valid=check_valid)


@set_module_as('brainstate.random')
def lognormal(
    mean=None,
    sigma=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a log-normal distribution.

    Draw samples from a log-normal distribution with specified mean,
    standard deviation, and array shape.  Note that the mean and standard
    deviation are not the values for the distribution itself, but of the
    underlying normal distribution it is derived from.

    Parameters
    ----------
    mean : float or array_like of floats, optional
        Mean value of the underlying normal distribution. Default is 0.
    sigma : float or array_like of floats, optional
        Standard deviation of the underlying normal distribution. Must be
        non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mean`` and ``sigma`` are both scalars.
        Otherwise, ``np.broadcast(mean, sigma).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized log-normal distribution.

    See Also
    --------
    scipy.stats.lognorm : probability density function, distribution,
        cumulative density function, etc.

    Notes
    -----
    A variable `x` has a log-normal distribution if `log(x)` is normally
    distributed.  The probability density function for the log-normal
    distribution is:

    .. math:: p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
                     e^{(-\frac{(ln(x)-\mu)^2}{2\sigma^2})}

    where :math:`\mu` is the mean and :math:`\sigma` is the standard
    deviation of the normally distributed logarithm of the variable.
    A log-normal distribution results if a random variable is the *product*
    of a large number of independent, identically-distributed variables in
    the same way that a normal distribution results if the variable is the
    *sum* of a large number of independent, identically-distributed
    variables.

    References
    ----------
    .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
           Distributions across the Sciences: Keys and Clues,"
           BioScience, Vol. 51, No. 5, May, 2001.
           https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
    .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
           Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

    Examples
    --------
    Draw samples from the distribution:

    >>> mu, sigma = 3., 1. # mean and standard deviation
    >>> s = brainstate.random.lognormal(mu, sigma, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

    >>> x = np.linspace(min(bins), max(bins), 10000)
    >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
    ...        / (x * sigma * np.sqrt(2 * np.pi)))

    >>> plt.plot(x, pdf, linewidth=2, color='r')
    >>> plt.axis('tight')
    >>> plt.show()

    Demonstrate that taking the products of random samples from a uniform
    distribution can be fit well by a log-normal probability density
    function.

    >>> # Generate a thousand samples: each is the product of 100 random
    >>> # values, drawn from a normal distribution.
    >>> b = []
    >>> for i in range(1000):
    ...    a = 10. + brainstate.random.standard_normal(100)
    ...    b.append(np.product(a))

    >>> b = np.array(b) / np.min(b) # scale values to be positive
    >>> count, bins, ignored = plt.hist(b, 100, density=True, align='mid')
    >>> sigma = np.std(np.log(b))
    >>> mu = np.mean(np.log(b))

    >>> x = np.linspace(min(bins), max(bins), 10000)
    >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
    ...        / (x * sigma * np.sqrt(2 * np.pi)))

    >>> plt.plot(x, pdf, color='r', linewidth=2)
    >>> plt.show()
    """
    return DEFAULT.lognormal(mean, sigma, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def binomial(
    n,
    p,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None,
    check_valid: bool = True
):
    r"""
    Draw samples from a binomial distribution.

    Samples are drawn from a binomial distribution with specified
    parameters, n trials and p probability of success where
    n an integer >= 0 and p is in the interval [0,1]. (n may be
    input as a float, but it is truncated to an integer in use)

    Parameters
    ----------
    n : int or array_like of ints
        Parameter of the distribution, >= 0. Floats are also accepted,
        but they will be truncated to integers.
    p : float or array_like of floats
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``n`` and ``p`` are both scalars.
        Otherwise, ``np.broadcast(n, p).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized binomial distribution, where
        each sample is equal to the number of successes over the n trials.

    See Also
    --------
    scipy.stats.binom : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the binomial distribution is

    .. math:: P(N) = \binom{n}{N}p^N(1-p)^{n-N},

    where :math:`n` is the number of trials, :math:`p` is the probability
    of success, and :math:`N` is the number of successes.

    When estimating the standard error of a proportion in a population by
    using a random sample, the normal distribution works well unless the
    product p*n <=5, where p = population proportion estimate, and n =
    number of samples, in which case the binomial distribution is used
    instead. For example, a sample of 15 people shows 4 who are left
    handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
    so the binomial distribution should be used in this case.

    References
    ----------
    .. [1] Dalgaard, Peter, "Introductory Statistics with R",
           Springer-Verlag, 2002.
    .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
           Fifth Edition, 2002.
    .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
           and Quigley, 1972.
    .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/BinomialDistribution.html
    .. [5] Wikipedia, "Binomial distribution",
           https://en.wikipedia.org/wiki/Binomial_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> n, p = 10, .5  # number of trials, probability of each trial
    >>> s = brainstate.random.binomial(n, p, 1000)
    # result of flipping a coin 10 times, tested 1000 times.

    A real world example. A company drills 9 wild-cat oil exploration
    wells, each with an estimated probability of success of 0.1. All nine
    wells fail. What is the probability of that happening?

    Let's do 20,000 trials of the model, and count the number that
    generate zero positive results.

    >>> sum(brainstate.random.binomial(9, 0.1, 20000) == 0)/20000.
    # answer = 0.38885, or 38%.
    """
    return DEFAULT.binomial(n, p, size=size, key=key, dtype=dtype, check_valid=check_valid)


@set_module_as('brainstate.random')
def chisquare(
    df,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a chi-square distribution.

    When `df` independent random variables, each with standard normal
    distributions (mean 0, variance 1), are squared and summed, the
    resulting distribution is chi-square (see Notes).  This distribution
    is often used in hypothesis testing.

    Parameters
    ----------
    df : float or array_like of floats
         Number of degrees of freedom, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``df`` is a scalar.  Otherwise,
        ``np.array(df).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized chi-square distribution.

    Raises
    ------
    ValueError
        When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
        is given.

    Notes
    -----
    The variable obtained by summing the squares of `df` independent,
    standard normally distributed random variables:

    .. math:: Q = \sum_{i=0}^{\mathtt{df}} X^2_i

    is chi-square distributed, denoted

    .. math:: Q \sim \chi^2_k.

    The probability density function of the chi-squared distribution is

    .. math:: p(x) = \frac{(1/2)^{k/2}}{\Gamma(k/2)}
                     x^{k/2 - 1} e^{-x/2},

    where :math:`\Gamma` is the gamma function,

    .. math:: \Gamma(x) = \int_0^{-\infty} t^{x - 1} e^{-t} dt.

    References
    ----------
    .. [1] NIST "Engineering Statistics Handbook"
           https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

    Examples
    --------
    Generate chi-square samples with 2 degrees of freedom:

    .. code-block:: python

        >>> import brainstate
        >>> samples = brainstate.random.chisquare(2, 4)
        >>> print(samples.shape)  # (4,)
        >>> print((samples >= 0).all())  # True (chi-square is always non-negative)
    """
    return DEFAULT.chisquare(df, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def dirichlet(
    alpha,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from the Dirichlet distribution.

    Draw `size` samples of dimension k from a Dirichlet distribution. A
    Dirichlet-distributed random variable can be seen as a multivariate
    generalization of a Beta distribution. The Dirichlet distribution
    is a conjugate prior of a multinomial distribution in Bayesian
    inference.

    Parameters
    ----------
    alpha : sequence of floats, length k
        Parameter of the distribution (length ``k`` for sample of
        length ``k``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        vector of length ``k`` is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    samples : ndarray,
        The drawn samples, of shape ``(size, k)``.

    Raises
    ------
    ValueError
        If any value in ``alpha`` is less than or equal to zero

    Notes
    -----
    The Dirichlet distribution is a distribution over vectors
    :math:`x` that fulfil the conditions :math:`x_i>0` and
    :math:`\sum_{i=1}^k x_i = 1`.

    The probability density function :math:`p` of a
    Dirichlet-distributed random vector :math:`X` is
    proportional to

    .. math:: p(x) \propto \prod_{i=1}^{k}{x^{\alpha_i-1}_i},

    where :math:`\alpha` is a vector containing the positive
    concentration parameters.

    The method uses the following property for computation: let :math:`Y`
    be a random vector which has components that follow a standard gamma
    distribution, then :math:`X = \frac{1}{\sum_{i=1}^k{Y_i}} Y`
    is Dirichlet-distributed

    References
    ----------
    .. [1] David McKay, "Information Theory, Inference and Learning
           Algorithms," chapter 23,
           http://www.inference.org.uk/mackay/itila/
    .. [2] Wikipedia, "Dirichlet distribution",
           https://en.wikipedia.org/wiki/Dirichlet_distribution

    Examples
    --------
    Taking an example cited in Wikipedia, this distribution can be used if
    one wanted to cut strings (each of initial length 1.0) into K pieces
    with different lengths, where each piece had, on average, a designated
    average length, but allowing some variation in the relative sizes of
    the pieces.

    >>> import brainstate
    >>> s = brainstate.random.dirichlet((10, 5, 3), 20).transpose()

    >>> import matplotlib.pyplot as plt  # noqa
    >>> plt.barh(range(20), s[0])
    >>> plt.barh(range(20), s[1], left=s[0], color='g')
    >>> plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
    >>> plt.title("Lengths of Strings")
    """
    return DEFAULT.dirichlet(alpha, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def geometric(
    p,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from the geometric distribution.

    Bernoulli trials are experiments with one of two outcomes:
    success or failure (an example of such an experiment is flipping
    a coin).  The geometric distribution models the number of trials
    that must be run in order to achieve success.  It is therefore
    supported on the positive integers, ``k = 1, 2, ...``.

    The probability mass function of the geometric distribution is

    .. math:: f(k) = (1 - p)^{k - 1} p

    where `p` is the probability of success of an individual trial.

    Parameters
    ----------
    p : float or array_like of floats
        The probability of success of an individual trial.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``p`` is a scalar.  Otherwise,
        ``np.array(p).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized geometric distribution.

    Examples
    --------
    Draw ten thousand values from the geometric distribution,
    with the probability of an individual success equal to 0.35:

    >>> import brainstate
    >>> z = brainstate.random.geometric(p=0.35, size=10000)

    How many trials succeeded after a single run?

    >>> (z == 1).sum() / 10000.
    0.34889999999999999 #random
    """
    return DEFAULT.geometric(p, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def f(
    dfnum,
    dfden,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from an F distribution.

    Samples are drawn from an F distribution with specified parameters,
    `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
    freedom in denominator), where both parameters must be greater than
    zero.

    The random variate of the F distribution (also known as the
    Fisher distribution) is a continuous probability distribution
    that arises in ANOVA tests, and is the ratio of two chi-square
    variates.

    Parameters
    ----------
    dfnum : float or array_like of floats
        Degrees of freedom in numerator, must be > 0.
    dfden : float or array_like of float
        Degrees of freedom in denominator, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``dfnum`` and ``dfden`` are both scalars.
        Otherwise, ``np.broadcast(dfnum, dfden).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Fisher distribution.

    See Also
    --------
    scipy.stats.f : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The F statistic is used to compare in-group variances to between-group
    variances. Calculating the distribution depends on the sampling, and
    so it is a function of the respective degrees of freedom in the
    problem.  The variable `dfnum` is the number of samples minus one, the
    between-groups degrees of freedom, while `dfden` is the within-groups
    degrees of freedom, the sum of the number of samples in each group
    minus the number of groups.

    References
    ----------
    .. [1] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
           Fifth Edition, 2002.
    .. [2] Wikipedia, "F-distribution",
           https://en.wikipedia.org/wiki/F-distribution

    Examples
    --------
    An example from Glantz[1], pp 47-40:

    Two groups, children of diabetics (25 people) and children from people
    without diabetes (25 controls). Fasting blood glucose was measured,
    case group had a mean value of 86.1, controls had a mean value of
    82.2. Standard deviations were 2.09 and 2.49 respectively. Are these
    data consistent with the null hypothesis that the parents diabetic
    status does not affect their children's blood glucose levels?
    Calculating the F statistic from the data gives a value of 36.01.

    Draw samples from the distribution:

    >>> import brainstate
    >>> dfnum = 1. # between group degrees of freedom
    >>> dfden = 48. # within groups degrees of freedom
    >>> s = brainstate.random.f(dfnum, dfden, 1000)

    The lower bound for the top 1% of the samples is :

    >>> np.sort(s)[-10]
    7.61988120985 # random

    So there is about a 1% chance that the F statistic will exceed 7.62,
    the measured value is 36, so the null hypothesis is rejected at the 1%
    level.
    """
    return DEFAULT.f(dfnum, dfden, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def hypergeometric(
    ngood,
    nbad,
    nsample,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Hypergeometric distribution.

    Samples are drawn from a hypergeometric distribution with specified
    parameters, `ngood` (ways to make a good selection), `nbad` (ways to make
    a bad selection), and `nsample` (number of items sampled, which is less
    than or equal to the sum ``ngood + nbad``).

    Parameters
    ----------
    ngood : int or array_like of ints
        Number of ways to make a good selection.  Must be nonnegative.
    nbad : int or array_like of ints
        Number of ways to make a bad selection.  Must be nonnegative.
    nsample : int or array_like of ints
        Number of items sampled.  Must be at least 1 and at most
        ``ngood + nbad``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if `ngood`, `nbad`, and `nsample`
        are all scalars.  Otherwise, ``np.broadcast(ngood, nbad, nsample).size``
        samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized hypergeometric distribution. Each
        sample is the number of good items within a randomly selected subset of
        size `nsample` taken from a set of `ngood` good items and `nbad` bad items.

    See Also
    --------
    scipy.stats.hypergeom : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Hypergeometric distribution is

    .. math:: P(x) = \frac{\binom{g}{x}\binom{b}{n-x}}{\binom{g+b}{n}},

    where :math:`0 \le x \le n` and :math:`n-b \le x \le g`

    for P(x) the probability of ``x`` good results in the drawn sample,
    g = `ngood`, b = `nbad`, and n = `nsample`.

    Consider an urn with black and white marbles in it, `ngood` of them
    are black and `nbad` are white. If you draw `nsample` balls without
    replacement, then the hypergeometric distribution describes the
    distribution of black balls in the drawn sample.

    Note that this distribution is very similar to the binomial
    distribution, except that in this case, samples are drawn without
    replacement, whereas in the Binomial case samples are drawn with
    replacement (or the sample space is infinite). As the sample space
    becomes large, this distribution approaches the binomial.

    References
    ----------
    .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
           and Quigley, 1972.
    .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/HypergeometricDistribution.html
    .. [3] Wikipedia, "Hypergeometric distribution",
           https://en.wikipedia.org/wiki/Hypergeometric_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import brainstate
    >>> ngood, nbad, nsamp = 100, 2, 10
    # number of good, number of bad, and number of samples
    >>> s = brainstate.random.hypergeometric(ngood, nbad, nsamp, 1000)
    >>> from matplotlib.pyplot import hist  # noqa
    >>> hist(s)
    #   note that it is very unlikely to grab both bad items

    Suppose you have an urn with 15 white and 15 black marbles.
    If you pull 15 marbles at random, how likely is it that
    12 or more of them are one color?

    >>> s = brainstate.random.hypergeometric(15, 15, 15, 100000)
    >>> sum(s>=12)/100000. + sum(s<=3)/100000.
    #   answer = 0.003 ... pretty unlikely!
    """
    return DEFAULT.hypergeometric(ngood, nbad, nsample, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def logseries(
    p,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a logarithmic series distribution.

    Samples are drawn from a log series distribution with specified
    shape parameter, 0 <= ``p`` < 1.

    Parameters
    ----------
    p : float or array_like of floats
        Shape parameter for the distribution.  Must be in the range [0, 1).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``p`` is a scalar.  Otherwise,
        ``np.array(p).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized logarithmic series distribution.

    See Also
    --------
    scipy.stats.logser : probability density function, distribution or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Log Series distribution is

    .. math:: P(k) = \frac{-p^k}{k \ln(1-p)},

    where p = probability.

    The log series distribution is frequently used to represent species
    richness and occurrence, first proposed by Fisher, Corbet, and
    Williams in 1943 [2].  It may also be used to model the numbers of
    occupants seen in cars [3].

    References
    ----------
    .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional
           species diversity through the log series distribution of
           occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,
           Volume 5, Number 5, September 1999 , pp. 187-195(9).
    .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The
           relation between the number of species and the number of
           individuals in a random sample of an animal population.
           Journal of Animal Ecology, 12:42-58.
    .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small
           Data Sets, CRC Press, 1994.
    .. [4] Wikipedia, "Logarithmic distribution",
           https://en.wikipedia.org/wiki/Logarithmic_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import brainstate
    >>> a = .6
    >>> s = brainstate.random.logseries(a, 10000)
    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, ignored = plt.hist(s)

    #   plot against distribution

    >>> def logseries(k, p):
    ...     return -p**k/(k*np.log(1-p))
    >>> plt.plot(bins, logseries(bins, a)*count.max()/
    ...          logseries(bins, a).max(), 'r')
    >>> plt.show()
    """
    return DEFAULT.logseries(p, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def multinomial(
    n,
    pvals,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None,
    check_valid: bool = True,
):
    r"""
    Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalization of the
    binomial distribution.  Take an experiment with one of ``p``
    possible outcomes.  An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6.  Each sample drawn from the
    distribution represents `n` such experiments.  Its values,
    ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
    outcome was ``i``.

    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        must sum to 1 (however, the last element is always assumed to
        account for the remaining probability, as long as
        ``sum(pvals[:-1]) <= 1)``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.

        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Examples
    --------
    Throw a dice 20 times:

    .. code-block:: python

        >>> import brainstate
        >>> result = brainstate.random.multinomial(20, [1/6.]*6, size=1)
        >>> print(result.shape)  # (1, 6)
        >>> print(result.sum())  # 20 (total throws)

    Now, throw the dice 20 times, and 20 times again:

    .. code-block:: python

        >>> result = brainstate.random.multinomial(20, [1/6.]*6, size=2)
        >>> print(result.shape)  # (2, 6)
        >>> print(result.sum(axis=1))  # [20, 20] (total throws per experiment)

    A loaded die is more likely to land on number 6:

    .. code-block:: python

        >>> result = brainstate.random.multinomial(100, [1/7.]*5 + [2/7.])
        >>> print(result.shape)  # (6,)
        >>> print(result.sum())  # 100 (total throws)

    The probability inputs should be normalized. A biased coin which has
    twice as much weight on one side as on the other should be sampled like so:

    .. code-block:: python

        >>> result = brainstate.random.multinomial(100, [1.0 / 3, 2.0 / 3])
        >>> print(result.shape)  # (2,)
        print(result.sum())  # 100 (total throws)
    """
    return DEFAULT.multinomial(n, pvals, size=size, key=key, dtype=dtype, check_valid=check_valid)


@set_module_as('brainstate.random')
def multivariate_normal(
    mean,
    cov,
    size: Optional[Size] = None,
    method: str = 'cholesky',
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw random samples from a multivariate normal distribution.

    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions.  Such a distribution is specified by its mean and
    covariance matrix.  These parameters are analogous to the mean
    (average or "center") and variance (standard deviation, or "width,"
    squared) of the one-dimensional normal distribution.

    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    size : int or tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
        each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
        If no shape is specified, a single (`N`-D) sample is returned.
    method : {'cholesky', 'eig'}, optional
        The method used to generate the random samples.  The default is 'cholesky'.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.
    dtype : data-type, optional
        The desired data-type for the output. Default is `float32`.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.

        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Notes
    -----
    The mean is a coordinate in N-dimensional space, which represents the
    location where samples are most likely to be generated.  This is
    analogous to the peak of the bell curve for the one-dimensional or
    univariate normal distribution.

    Covariance indicates the level to which two variables vary together.
    From the multivariate normal distribution, we draw N-dimensional
    samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
    element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
    The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
    "spread").

    Instead of specifying the full covariance matrix, popular
    approximations include:

      - Spherical covariance (`cov` is a multiple of the identity matrix)
      - Diagonal covariance (`cov` has non-negative elements, and only on
        the diagonal)

    This geometrical property can be seen in two dimensions by plotting
    generated data-points:

    >>> mean = [0, 0]
    >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

    Diagonal covariance means that points are oriented along x or y-axis:

    >>> import brainstate
    >>> import matplotlib.pyplot as plt  # noqa
    >>> x, y = brainstate.random.multivariate_normal(mean, cov, 5000).T
    >>> plt.plot(x, y, 'x')
    >>> plt.axis('equal')
    >>> plt.show()

    Note that the covariance matrix must be positive semidefinite (a.k.a.
    nonnegative-definite). Otherwise, the behavior of this method is
    undefined and backwards compatibility is not guaranteed.

    References
    ----------
    .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
           Processes," 3rd ed., New York: McGraw-Hill, 1991.
    .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
           Classification," 2nd ed., New York: Wiley, 2001.

    Examples
    --------
    >>> mean = (1, 2)
    >>> cov = [[1, 0], [0, 1]]
    >>> x = brainstate.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)

    Here we generate 800 samples from the bivariate normal distribution
    with mean [0, 0] and covariance matrix [[6, -3], [-3, 3.5]].  The
    expected variances of the first and second components of the sample
    are 6 and 3.5, respectively, and the expected correlation
    coefficient is -3/sqrt(6*3.5) ≈ -0.65465.

    >>> cov = np.array([[6, -3], [-3, 3.5]])
    >>> pts = brainstate.random.multivariate_normal([0, 0], cov, size=800)

    Check that the mean, covariance, and correlation coefficient of the
    sample are close to the expected values:

    >>> pts.mean(axis=0)
    array([ 0.0326911 , -0.01280782])  # may vary
    >>> np.cov(pts.T)
    array([[ 5.96202397, -2.85602287],
           [-2.85602287,  3.47613949]])  # may vary
    >>> np.corrcoef(pts.T)[0, 1]
    -0.6273591314603949  # may vary

    We can visualize this data with a scatter plot.  The orientation
    of the point cloud illustrates the negative correlation of the
    components of this sample.

    >>> import matplotlib.pyplot as plt  # noqa
    >>> plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
    >>> plt.axis('equal')
    >>> plt.grid()
    >>> plt.show()
    """
    return DEFAULT.multivariate_normal(mean, cov, size=size, method=method, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def negative_binomial(
    n,
    p,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a negative binomial distribution.

    Samples are drawn from a negative binomial distribution with specified
    parameters, `n` successes and `p` probability of success where `n`
    is > 0 and `p` is in the interval [0, 1].

    Parameters
    ----------
    n : float or array_like of floats
        Parameter of the distribution, > 0.
    p : float or array_like of floats
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``n`` and ``p`` are both scalars.
        Otherwise, ``np.broadcast(n, p).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized negative binomial distribution,
        where each sample is equal to N, the number of failures that
        occurred before a total of n successes was reached.

    Notes
    -----
    The probability mass function of the negative binomial distribution is

    .. math:: P(N;n,p) = \frac{\Gamma(N+n)}{N!\Gamma(n)}p^{n}(1-p)^{N},

    where :math:`n` is the number of successes, :math:`p` is the
    probability of success, :math:`N+n` is the number of trials, and
    :math:`\Gamma` is the gamma function. When :math:`n` is an integer,
    :math:`\frac{\Gamma(N+n)}{N!\Gamma(n)} = \binom{N+n-1}{N}`, which is
    the more common form of this term in the pmf. The negative
    binomial distribution gives the probability of N failures given n
    successes, with a success on the last trial.

    If one throws a die repeatedly until the third time a "1" appears,
    then the probability distribution of the number of non-"1"s that
    appear before the third "1" is a negative binomial distribution.

    References
    ----------
    .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/NegativeBinomialDistribution.html
    .. [2] Wikipedia, "Negative binomial distribution",
           https://en.wikipedia.org/wiki/Negative_binomial_distribution

    Examples
    --------
    Draw samples from the distribution:

    A real world example. A company drills wild-cat oil
    exploration wells, each with an estimated probability of
    success of 0.1.  What is the probability of having one success
    for each successive well, that is what is the probability of a
    single success after drilling 5 wells, after 6 wells, etc.?

    >>> import brainstate
    >>> s = brainstate.random.negative_binomial(1, 0.1, 100000)
    >>> for i in range(1, 11): # doctest: +SKIP
    ...    probability = sum(s<i) / 100000.
    ...    print(i, "wells drilled, probability of one success =", probability)
    """
    return DEFAULT.negative_binomial(n, p, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def noncentral_chisquare(
    df,
    nonc,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a noncentral chi-square distribution.

    The noncentral :math:`\chi^2` distribution is a generalization of
    the :math:`\chi^2` distribution.

    Parameters
    ----------
    df : float or array_like of floats
        Degrees of freedom, must be > 0.
    nonc : float or array_like of floats
        Non-centrality, must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``df`` and ``nonc`` are both scalars.
        Otherwise, ``np.broadcast(df, nonc).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized noncentral chi-square distribution.

    Notes
    -----
    The probability density function for the noncentral Chi-square
    distribution is

    .. math:: P(x;df,nonc) = \sum^{\infty}_{i=0}
                           \frac{e^{-nonc/2}(nonc/2)^{i}}{i!}
                           P_{Y_{df+2i}}(x),

    where :math:`Y_{q}` is the Chi-square with q degrees of freedom.

    References
    ----------
    .. [1] Wikipedia, "Noncentral chi-squared distribution"
           https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram

    >>> import brainstate
    >>> import matplotlib.pyplot as plt  # noqa
    >>> values = plt.hist(brainstate.random.noncentral_chisquare(3, 20, 100000),
    ...                   bins=200, density=True)
    >>> plt.show()

    Draw values from a noncentral chisquare with very small noncentrality,
    and compare to a chisquare.

    >>> plt.figure()
    >>> values = plt.hist(brainstate.random.noncentral_chisquare(3, .0000001, 100000),
    ...                   bins=np.arange(0., 25, .1), density=True)
    >>> values2 = plt.hist(brainstate.random.chisquare(3, 100000),
    ...                    bins=np.arange(0., 25, .1), density=True)
    >>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
    >>> plt.show()

    Demonstrate how large values of non-centrality lead to a more symmetric
    distribution.

    >>> plt.figure()
    >>> values = plt.hist(brainstate.random.noncentral_chisquare(3, 20, 100000),
    ...                   bins=200, density=True)
    >>> plt.show()
    """
    return DEFAULT.noncentral_chisquare(df, nonc, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def noncentral_f(
    dfnum,
    dfden,
    nonc,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from the noncentral F distribution.

    Samples are drawn from an F distribution with specified parameters,
    `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
    freedom in denominator), where both parameters > 1.
    `nonc` is the non-centrality parameter.

    Parameters
    ----------
    dfnum : float or array_like of floats
        Numerator degrees of freedom, must be > 0.
    dfden : float or array_like of floats
        Denominator degrees of freedom, must be > 0.
    nonc : float or array_like of floats
        Non-centrality parameter, the sum of the squares of the numerator
        means, must be >= 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``dfnum``, ``dfden``, and ``nonc``
        are all scalars.  Otherwise, ``np.broadcast(dfnum, dfden, nonc).size``
        samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized noncentral Fisher distribution.

    Notes
    -----
    When calculating the power of an experiment (power = probability of
    rejecting the null hypothesis when a specific alternative is true) the
    non-central F statistic becomes important.  When the null hypothesis is
    true, the F statistic follows a central F distribution. When the null
    hypothesis is not true, then it follows a non-central F statistic.

    References
    ----------
    .. [1] Weisstein, Eric W. "Noncentral F-Distribution."
           From MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/NoncentralF-Distribution.html
    .. [2] Wikipedia, "Noncentral F-distribution",
           https://en.wikipedia.org/wiki/Noncentral_F-distribution

    Examples
    --------
    In a study, testing for a specific alternative to the null hypothesis
    requires use of the Noncentral F distribution. We need to calculate the
    area in the tail of the distribution that exceeds the value of the F
    distribution for the null hypothesis.  We'll plot the two probability
    distributions for comparison.

    >>> import brainstate
    >>> dfnum = 3 # between group deg of freedom
    >>> dfden = 20 # within groups degrees of freedom
    >>> nonc = 3.0
    >>> nc_vals = brainstate.random.noncentral_f(dfnum, dfden, nonc, 1000000)
    >>> NF = np.histogram(nc_vals, bins=50, density=True)
    >>> c_vals = brainstate.random.f(dfnum, dfden, 1000000)
    >>> F = np.histogram(c_vals, bins=50, density=True)
    >>> import matplotlib.pyplot as plt  # noqa
    >>> plt.plot(F[1][1:], F[0])
    >>> plt.plot(NF[1][1:], NF[0])
    >>> plt.show()
    """
    return DEFAULT.noncentral_f(dfnum, dfden, nonc, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def power(
    a,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draws samples in [0, 1] from a power distribution with positive
    exponent a - 1.

    Also known as the power function distribution.

    Parameters
    ----------
    a : float or array_like of floats
        Parameter of the distribution. Must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar.  Otherwise,
        ``np.array(a).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized power distribution.

    Raises
    ------
    ValueError
        If a <= 0.

    Notes
    -----
    The probability density function is

    .. math:: P(x; a) = ax^{a-1}, 0 \le x \le 1, a>0.

    The power function distribution is just the inverse of the Pareto
    distribution. It may also be seen as a special case of the Beta
    distribution.

    It is used, for example, in modeling the over-reporting of insurance
    claims.

    References
    ----------
    .. [1] Christian Kleiber, Samuel Kotz, "Statistical size distributions
           in economics and actuarial sciences", Wiley, 2003.
    .. [2] Heckert, N. A. and Filliben, James J. "NIST Handbook 148:
           Dataplot Reference Manual, Volume 2: Let Subcommands and Library
           Functions", National Institute of Standards and Technology
           Handbook Series, June 2003.
           https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/powpdf.pdf

    Examples
    --------
    Draw samples from the distribution:

    >>> import brainstate
    >>> a = 5. # shape
    >>> samples = 1000
    >>> s = brainstate.random.power(a, samples)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> count, bins, ignored = plt.hist(s, bins=30)
    >>> x = np.linspace(0, 1, 100)
    >>> y = a*x**(a-1.)
    >>> normed_y = samples*np.diff(bins)[0]*y
    >>> plt.plot(x, normed_y)
    >>> plt.show()

    Compare the power function distribution to the inverse of the Pareto.

    >>> from scipy import stats # doctest: +SKIP
    >>> rvs = brainstate.random.power(5, 1000000)
    >>> rvsp = brainstate.random.pareto(5, 1000000)
    >>> xx = np.linspace(0,1,100)
    >>> powpdf = stats.powerlaw.pdf(xx,5)  # doctest: +SKIP

    >>> plt.figure()
    >>> plt.hist(rvs, bins=50, density=True)
    >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
    >>> plt.title('brainstate.random.power(5)')

    >>> plt.figure()
    >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
    >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
    >>> plt.title('inverse of 1 + brainstate.random.pareto(5)')

    >>> plt.figure()
    >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
    >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
    >>> plt.title('inverse of stats.pareto(5)')
    """
    return DEFAULT.power(a, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def rayleigh(
    scale=1.0,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Rayleigh distribution.

    The :math:`\chi` and Weibull distributions are generalizations of the
    Rayleigh.

    Parameters
    ----------
    scale : float or array_like of floats, optional
        Scale, also equals the mode. Must be non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.  Otherwise,
        ``np.array(scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Rayleigh distribution.

    Notes
    -----
    The probability density function for the Rayleigh distribution is

    .. math:: P(x;scale) = \frac{x}{scale^2}e^{\frac{-x^2}{2 \cdotp scale^2}}

    The Rayleigh distribution would arise, for example, if the East
    and North components of the wind velocity had identical zero-mean
    Gaussian distributions.  Then the wind speed would have a Rayleigh
    distribution.

    References
    ----------
    .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
           https://web.archive.org/web/20090514091424/http://brighton-webs.co.uk:80/distributions/rayleigh.asp
    .. [2] Wikipedia, "Rayleigh distribution"
           https://en.wikipedia.org/wiki/Rayleigh_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram

    >>> import brainstate
    >>> from matplotlib.pyplot import hist  # noqa
    >>> values = hist(brainstate.random.rayleigh(3, 100000), bins=200, density=True)

    Wave heights tend to follow a Rayleigh distribution. If the mean wave
    height is 1 meter, what fraction of waves are likely to be larger than 3
    meters?

    >>> meanvalue = 1
    >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
    >>> s = brainstate.random.rayleigh(modevalue, 1000000)

    The percentage of waves larger than 3 meters is:

    >>> 100.*sum(s>3)/1000000.
    0.087300000000000003 # random
    """
    return DEFAULT.rayleigh(scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def triangular(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None
):
    r"""
    Draw samples from the triangular distribution over the
    interval ``[left, right]``.

    The triangular distribution is a continuous probability
    distribution with lower limit left, peak at mode, and upper
    limit right. Unlike the other distributions, these parameters
    directly define the shape of the pdf.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``left``, ``mode``, and ``right``
        are all scalars.  Otherwise, ``np.broadcast(left, mode, right).size``
        samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized triangular distribution.

    Notes
    -----
    The probability density function for the triangular distribution is

    .. math:: P(x;l, m, r) = \begin{cases}
              \frac{2(x-l)}{(r-l)(m-l)}& \text{for $l \leq x \leq m$},\\
              \frac{2(r-x)}{(r-l)(r-m)}& \text{for $m \leq x \leq r$},\\
              0& \text{otherwise}.
              \end{cases}

    The triangular distribution is often used in ill-defined
    problems where the underlying distribution is not known, but
    some knowledge of the limits and mode exists. Often it is used
    in simulations.

    References
    ----------
    .. [1] Wikipedia, "Triangular distribution"
           https://en.wikipedia.org/wiki/Triangular_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram:

    >>> import brainstate
    >>> import matplotlib.pyplot as plt  # noqa
    >>> h = plt.hist(brainstate.random.triangular(-3, 0, 8, 100000), bins=200,
    ...              density=True)
    >>> plt.show()
    """
    return DEFAULT.triangular(size=size, key=key)


@set_module_as('brainstate.random')
def vonmises(
    mu,
    kappa,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a von Mises distribution.

    Samples are drawn from a von Mises distribution with specified mode
    (mu) and dispersion (kappa), on the interval [-pi, pi].

    The von Mises distribution (also known as the circular normal
    distribution) is a continuous probability distribution on the unit
    circle.  It may be thought of as the circular analogue of the normal
    distribution.

    Parameters
    ----------
    mu : float or array_like of floats
        Mode ("center") of the distribution.
    kappa : float or array_like of floats
        Dispersion of the distribution, has to be >=0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mu`` and ``kappa`` are both scalars.
        Otherwise, ``np.broadcast(mu, kappa).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized von Mises distribution.

    See Also
    --------
    scipy.stats.vonmises : probability density function, distribution, or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the von Mises distribution is

    .. math:: p(x) = \frac{e^{\kappa cos(x-\mu)}}{2\pi I_0(\kappa)},

    where :math:`\mu` is the mode and :math:`\kappa` the dispersion,
    and :math:`I_0(\kappa)` is the modified Bessel function of order 0.

    The von Mises is named for Richard Edler von Mises, who was born in
    Austria-Hungary, in what is now the Ukraine.  He fled to the United
    States in 1939 and became a professor at Harvard.  He worked in
    probability theory, aerodynamics, fluid mechanics, and philosophy of
    science.

    References
    ----------
    .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
           Mathematical Functions with Formulas, Graphs, and Mathematical
           Tables, 9th printing," New York: Dover, 1972.
    .. [2] von Mises, R., "Mathematical Theory of Probability
           and Statistics", New York: Academic Press, 1964.

    Examples
    --------
    Draw samples from the distribution:

    >>> import brainstate
    >>> mu, kappa = 0.0, 4.0 # mean and dispersion
    >>> s = brainstate.random.vonmises(mu, kappa, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> from scipy.special import i0  # doctest: +SKIP
    >>> plt.hist(s, 50, density=True)
    >>> x = np.linspace(-np.pi, np.pi, num=51)
    >>> y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))  # doctest: +SKIP
    >>> plt.plot(x, y, linewidth=2, color='r')  # doctest: +SKIP
    >>> plt.show()
    """
    return DEFAULT.vonmises(mu, kappa, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def wald(
    mean,
    scale,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Wald, or inverse Gaussian, distribution.

    As the scale approaches infinity, the distribution becomes more like a
    Gaussian. Some references claim that the Wald is an inverse Gaussian
    with mean equal to 1, but this is by no means universal.

    The inverse Gaussian distribution was first studied in relationship to
    Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
    because there is an inverse relationship between the time to cover a
    unit distance and distance covered in unit time.

    Parameters
    ----------
    mean : float or array_like of floats
        Distribution mean, must be > 0.
    scale : float or array_like of floats
        Scale parameter, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mean`` and ``scale`` are both scalars.
        Otherwise, ``np.broadcast(mean, scale).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Wald distribution.

    Notes
    -----
    The probability density function for the Wald distribution is

    .. math:: P(x;mean,scale) = \sqrt{\frac{scale}{2\pi x^3}}e^
                                \frac{-scale(x-mean)^2}{2\cdotp mean^2x}

    As noted above the inverse Gaussian distribution first arise
    from attempts to model Brownian motion. It is also a
    competitor to the Weibull for use in reliability modeling and
    modeling stock returns and interest rate processes.

    References
    ----------
    .. [1] Brighton Webs Ltd., Wald Distribution,
           https://web.archive.org/web/20090423014010/http://www.brighton-webs.co.uk:80/distributions/wald.asp
    .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
           Distribution: Theory : Methodology, and Applications", CRC Press,
           1988.
    .. [3] Wikipedia, "Inverse Gaussian distribution"
           https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution

    Examples
    --------
    Draw values from the distribution and plot the histogram:

    >>> import brainstate
    >>> import matplotlib.pyplot as plt  # noqa
    >>> h = plt.hist(brainstate.random.wald(3, 2, 100000), bins=200, density=True)
    >>> plt.show()
    """
    return DEFAULT.wald(mean, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def weibull(
    a,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Weibull distribution.

    Draw samples from a 1-parameter Weibull distribution with the given
    shape parameter `a`.

    .. math:: X = (-ln(U))^{1/a}

    Here, U is drawn from the uniform distribution over (0,1].

    The more common 2-parameter Weibull, including a scale parameter
    :math:`\lambda` is just :math:`X = \lambda(-ln(U))^{1/a}`.

    Parameters
    ----------
    a : float or array_like of floats
        Shape parameter of the distribution.  Must be nonnegative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar.  Otherwise,
        ``np.array(a).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Weibull distribution.

    Notes
    -----
    The Weibull (or Type III asymptotic extreme value distribution
    for smallest values, SEV Type III, or Rosin-Rammler
    distribution) is one of a class of Generalized Extreme Value
    (GEV) distributions used in modeling extreme value problems.
    This class includes the Gumbel and Frechet distributions.

    The probability density for the Weibull distribution is

    .. math:: p(x) = \frac{a}
                     {\lambda}(\frac{x}{\lambda})^{a-1}e^{-(x/\lambda)^a},

    where :math:`a` is the shape and :math:`\lambda` the scale.

    The function has its peak (the mode) at
    :math:`\lambda(\frac{a-1}{a})^{1/a}`.

    When ``a = 1``, the Weibull distribution reduces to the exponential
    distribution.

    References
    ----------
    .. [1] Waloddi Weibull, Royal Technical University, Stockholm,
           1939 "A Statistical Theory Of The Strength Of Materials",
           Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
           Generalstabens Litografiska Anstalts Forlag, Stockholm.
    .. [2] Waloddi Weibull, "A Statistical Distribution Function of
           Wide Applicability", Journal Of Applied Mechanics ASME Paper
           1951.
    .. [3] Wikipedia, "Weibull distribution",
           https://en.wikipedia.org/wiki/Weibull_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> import brainstate
    >>> a = 5. # shape
    >>> s = brainstate.random.weibull(a, 1000)

    Display the histogram of the samples, along with
    the probability density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> x = np.arange(1,100.)/50.
    >>> def weib(x,n,a):
    ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

    >>> count, bins, ignored = plt.hist(brainstate.random.weibull(5.,1000))
    >>> x = np.arange(1,100.)/50.
    >>> scale = count.max()/weib(x, 1., 5.).max()
    >>> plt.plot(x, weib(x, 1., 5.)*scale)
    >>> plt.show()

    """
    return DEFAULT.weibull(a, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def weibull_min(
    a,
    scale=None,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    """Sample from a Weibull distribution.

    The scipy counterpart is `scipy.stats.weibull_min`.

    Args:
      scale: The scale parameter of the distribution.
      concentration: The concentration parameter of the distribution.
      shape: The shape added to the parameters loc and scale broadcastable shape.
      dtype: The type used for samples.
      key: a PRNG key or a seed.

    Returns:
      A jnp.array of samples.

    """
    return DEFAULT.weibull_min(a, scale, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def zipf(
    a,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    r"""
    Draw samples from a Zipf distribution.

    Samples are drawn from a Zipf distribution with specified parameter
    `a` > 1.

    The Zipf distribution (also known as the zeta distribution) is a
    discrete probability distribution that satisfies Zipf's law: the
    frequency of an item is inversely proportional to its rank in a
    frequency table.

    Parameters
    ----------
    a : float or array_like of floats
        Distribution parameter. Must be greater than 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar. Otherwise,
        ``np.array(a).size`` samples are drawn.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Zipf distribution.

    See Also
    --------
    scipy.stats.zipf : probability density function, distribution, or
        cumulative density function, etc.

    Notes
    -----
    The probability density for the Zipf distribution is

    .. math:: p(k) = \frac{k^{-a}}{\zeta(a)},

    for integers :math:`k \geq 1`, where :math:`\zeta` is the Riemann Zeta
    function.

    It is named for the American linguist George Kingsley Zipf, who noted
    that the frequency of any word in a sample of a language is inversely
    proportional to its rank in the frequency table.

    References
    ----------
    .. [1] Zipf, G. K., "Selected Studies of the Principle of Relative
           Frequency in Language," Cambridge, MA: Harvard Univ. Press,
           1932.

    Examples
    --------
    Draw samples from the distribution:

    >>> import brainstate
    >>> a = 4.0
    >>> n = 20000
    >>> s = brainstate.random.zipf(a, n)

    Display the histogram of the samples, along with
    the expected histogram based on the probability
    density function:

    >>> import matplotlib.pyplot as plt  # noqa
    >>> from scipy.special import zeta  # doctest: +SKIP

    `bincount` provides a fast histogram for small integers.

    >>> count = np.bincount(s)
    >>> k = np.arange(1, s.max() + 1)

    >>> plt.bar(k, count[1:], alpha=0.5, label='sample count')
    >>> plt.plot(k, n*(k**-a)/zeta(a), 'k.-', alpha=0.5,
    ...          label='expected count')   # doctest: +SKIP
    >>> plt.semilogy()
    >>> plt.grid(alpha=0.4)
    >>> plt.legend()
    >>> plt.title(f'Zipf sample, a={a}, size={n}')
    >>> plt.show()
    """
    return DEFAULT.zipf(a, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def maxwell(
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    """Sample from a one sided Maxwell distribution.

    The scipy counterpart is `scipy.stats.maxwell`.

    Args:
      key: a PRNG key.
      size: The shape of the returned samples.
      dtype: The type used for samples.
      key: a PRNG key or a seed.

    Returns:
      A jnp.array of samples, of shape `shape`.

    """
    return DEFAULT.maxwell(size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def t(
    df,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    """Sample Student’s t random values.

    Parameters
    ----------
    df: float, array_like
      A float or array of floats broadcast-compatible with shape representing the parameter of the distribution.
    size: optional, int, tuple of int
      A tuple of non-negative integers specifying the result shape.
      Must be broadcast-compatible with `df`. The default (None) produces a result shape equal to `df.shape`.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out: array_like
      The sampled value.
    """
    return DEFAULT.t(df, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def orthogonal(
    n: int,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    """Sample uniformly from the orthogonal group `O(n)`.

    Parameters
    ----------
    n: int
       An integer indicating the resulting dimension.
    size: optional, int, tuple of int
      The batch dimensions of the result.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out: Array
      The sampled results.
    """
    return DEFAULT.orthogonal(n, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def loggamma(
    a,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None,
    dtype: DTypeLike = None
):
    """Sample log-gamma random values.

    Parameters
    ----------
    a: float, array_like
      A float or array of floats broadcast-compatible with shape representing the parameter of the distribution.
    size: optional, int, tuple of int
      A tuple of nonnegative integers specifying the result shape.
      Must be broadcast-compatible with `a`. The default (None) produces a result shape equal to `a.shape`.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.
    key : PRNGKey, optional
        The key for the random number generator. If not given, the
        default random number generator is used.

    Returns
    -------
    out: array_like
      The sampled results.
    """
    return DEFAULT.loggamma(a, size=size, key=key, dtype=dtype)


@set_module_as('brainstate.random')
def categorical(
    logits,
    axis: int = -1,
    size: Optional[Size] = None,
    key: Optional[SeedOrKey] = None
):
    """Sample random values from categorical distributions.

    Args:
      logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
        so that `softmax(logits, axis)` gives the corresponding probabilities.
      axis: Axis along which logits belong to the same categorical distribution.
      shape: Optional, a tuple of nonnegative integers representing the result shape.
        Must be broadcast-compatible with ``np.delete(logits.shape, axis)``.
        The default (None) produces a result shape equal to ``np.delete(logits.shape, axis)``.
      key: a PRNG key used as the random key.

    Returns:
      A random array with int dtype and shape given by ``shape`` if ``shape``
      is not None, or else ``np.delete(logits.shape, axis)``.
    """
    return DEFAULT.categorical(logits, axis, size=size, key=key)


@set_module_as('brainstate.random')
def rand_like(
    input,
    *,
    dtype=None,
    key: Optional[SeedOrKey] = None
):
    """Similar to ``rand_like`` in torch.

    Returns a tensor with the same size as input that is filled with random
    numbers from a uniform distribution on the interval ``[0, 1)``.

    Args:
      input:  the ``size`` of input will determine size of the output tensor.
      dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
      key: the seed or key for the random.

    Returns:
      The random data.
    """
    return DEFAULT.rand_like(input, dtype=dtype, key=key)


@set_module_as('brainstate.random')
def randn_like(
    input,
    *,
    dtype=None,
    key: Optional[SeedOrKey] = None
):
    """Similar to ``randn_like`` in torch.

    Returns a tensor with the same size as ``input`` that is filled with
    random numbers from a normal distribution with mean 0 and variance 1.

    Args:
      input:  the ``size`` of input will determine size of the output tensor.
      dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
      key: the seed or key for the random.

    Returns:
      The random data.
    """
    return DEFAULT.randn_like(input, dtype=dtype, key=key)


@set_module_as('brainstate.random')
def randint_like(
    input,
    low=0,
    high=None,
    *,
    dtype=None,
    key: Optional[SeedOrKey] = None
):
    """Similar to ``randint_like`` in torch.

    Returns a tensor with the same shape as Tensor ``input`` filled with
    random integers generated uniformly between ``low`` (inclusive) and ``high`` (exclusive).

    Args:
      input:  the ``size`` of input will determine size of the output tensor.
      low: Lowest integer to be drawn from the distribution. Default: 0.
      high: One above the highest integer to be drawn from the distribution.
      dtype: the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
      key: the seed or key for the random.

    Returns:
      The random data.
    """
    return DEFAULT.randint_like(input=input, low=low, high=high, dtype=dtype, key=key)


# ---------------------------------------------------------------------------------------------------------------


for __k in dir(RandomState):
    __t = getattr(RandomState, __k)
    if not __k.startswith('__') and callable(__t) and (not __t.__doc__):
        __r = globals().get(__k, None)
        if __r is not None and callable(__r):
            __t.__doc__ = __r.__doc__
