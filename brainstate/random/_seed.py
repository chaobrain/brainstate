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
Random seed management utilities for BrainState.

This module provides comprehensive random seed management functionality, enabling
reproducible computations across JAX and NumPy backends. It supports both traditional
integer seeds and JAX's PRNG key system, providing a unified interface for random
number generation in scientific computing and machine learning applications.

Key Features:
    - Unified seed management for JAX and NumPy
    - Context managers for temporary seed changes
    - Key splitting for parallel computation
    - Automatic seed backup and restoration
    - Thread-safe random state management

Example:
    Basic usage for reproducible random number generation:

    >>> import brainstate
    >>> brainstate.random.seed(42)
    >>> print(brainstate.random.rand(3))
    [0.95598125 0.4032725  0.96086407]

    Using context managers for temporary seeds:

    >>> with brainstate.random.seed_context(123):
    ...     values = brainstate.random.rand(2)
    >>> print(values)  # Reproducible output

    Key splitting for parallel computation:

    >>> keys = brainstate.random.split_keys(4)  # Generate 4 independent keys
    >>> # Use keys for parallel random number generation
"""

from contextlib import contextmanager

import jax
import numpy as np

from brainstate._utils import set_module_as
from brainstate.typing import SeedOrKey
from ._state import RandomState, DEFAULT, use_prng_key

__all__ = [
    'seed',
    'set_key',
    'get_key',
    'default_rng',
    'split_key',
    'split_keys',
    'seed_context',
    'restore_key',
    'self_assign_multi_keys',
    'clone_rng',
]


@set_module_as('brainstate.random')
def restore_key() -> None:
    """
    Restore the default random key to its previous state.

    This function restores the global random state to a previously backed up state.
    It's useful for undoing changes to the random state or implementing checkpoint
    functionality in computational workflows.

    Note:
        This operation requires that a backup was previously created. If no backup
        exists, this function may not have any effect or may restore to an initial state.

    Example:
        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> original_key = brainstate.random.get_key()
        >>> brainstate.random.seed(123)  # Change the seed
        >>> brainstate.random.restore_key()  # Restore to previous state
        >>> assert np.array_equal(brainstate.random.get_key(), original_key)

    See Also:
        - :func:`set_key`: Set a new random key
        - :func:`get_key`: Get the current random key
        - :func:`seed_context`: Temporary seed changes with automatic restoration
    """
    DEFAULT.restore_key()


@set_module_as('brainstate.random')
def split_key(n: int = None, backup: bool = False):
    """
    Create new random key(s) from the current seed.

    This function generates one or more independent random keys by splitting the
    current global random state. It follows JAX's random paradigm, ensuring that
    each split key produces statistically independent random sequences.

    Args:
        n: The number of keys to generate. If None, returns a single key.
            If an integer, returns an array of n keys.
        backup: Whether to backup the current key before splitting. This allows
            restoration of the original state using :func:`restore_key`.

    Returns:
        If n is None: A single JAX PRNG key.
        If n is an integer: An array of n independent JAX PRNG keys.

    Example:
        Generate a single key:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> key = brainstate.random.split_key()
        >>> print(key.shape)
        (2,)

        Generate multiple keys for parallel computation:

        >>> keys = brainstate.random.split_key(4)
        >>> print(keys.shape)
        (4, 2)

        Use with backup for state restoration:

        >>> original_key = brainstate.random.get_key()
        >>> keys = brainstate.random.split_key(2, backup=True)
        >>> brainstate.random.restore_key()
        >>> assert np.array_equal(brainstate.random.get_key(), original_key)

    Note:
        This function advances the global random state. Each call produces
        different keys unless the state is reset.

    See Also:
        - :func:`split_keys`: Convenience function for multiple keys
        - :func:`seed`: Set the random seed
        - :func:`restore_key`: Restore backed up key
    """
    return DEFAULT.split_key(n=n, backup=backup)


@set_module_as('brainstate.random')
def split_keys(n: int, backup: bool = False):
    """
    Create multiple independent random keys from the current seed.

    This is a convenience function that generates exactly n independent random keys
    by splitting the current global random state. It's commonly used internally by
    parallel computation functions like `pmap` and `vmap` to ensure that each
    parallel thread gets a unique random key.

    Args:
        n: The number of independent keys to generate. Must be a positive integer.
        backup: Whether to backup the current key before splitting. If True,
            the original key can be restored using :func:`restore_key`.

    Returns:
        An array of n independent JAX PRNG keys with shape (n, 2).

    Raises:
        ValueError: If n is not a positive integer.

    Example:
        Generate keys for parallel computation:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> keys = brainstate.random.split_keys(4)
        >>> print(keys.shape)
        (4, 2)

        Use with vmap for parallel random number generation:

        >>> import jax
        >>> keys = brainstate.random.split_keys(8)
        >>> @jax.vmap
        ... def generate_random(key):
        ...     return jax.random.normal(key, (10,))
        >>> parallel_randoms = generate_random(keys)
        >>> print(parallel_randoms.shape)
        (8, 10)

        Use with backup for state preservation:

        >>> original_state = brainstate.random.get_key()
        >>> keys = brainstate.random.split_keys(3, backup=True)
        >>> # ... use keys for computation ...
        >>> brainstate.random.restore_key()  # Restore original state

    Note:
        This function is equivalent to calling :func:`split_key` with n as an argument.
        It's provided as a convenience function with a more explicit name for clarity.

    See Also:
        - :func:`split_key`: More general key splitting function
        - :func:`self_assign_multi_keys`: Assign multiple keys to global state
        - :func:`seed_context`: Temporary seed changes
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    return split_key(n, backup=backup)


@set_module_as('brainstate.random')
def self_assign_multi_keys(n: int, backup: bool = True) -> None:
    """
    Assign multiple keys to the global random state for parallel access.

    This function prepares the global random state for parallel computation by
    pre-generating n independent keys. It's particularly useful when you need
    to ensure that parallel computations have access to independent random
    sequences without the overhead of key splitting during computation.

    Args:
        n: The number of independent keys to pre-generate and assign.
            Must be a positive integer.
        backup: Whether to backup the current random state before assignment.
            If True, the original state can be restored using :func:`restore_key`.

    Raises:
        ValueError: If n is not a positive integer.

    Example:
        Prepare for parallel computation:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> # Prepare 4 independent keys for parallel access
        >>> brainstate.random.self_assign_multi_keys(4)

        Use in parallel context:

        >>> # The random state now has 4 independent keys ready for use
        >>> # Each parallel thread can access a different key

    Note:
        This is an advanced function primarily used internally for optimizing
        parallel random number generation. In most cases, :func:`split_keys`
        provides a more straightforward interface for parallel computation.

    See Also:
        - :func:`split_keys`: Generate multiple independent keys
        - :func:`restore_key`: Restore backed up state
        - :func:`seed_context`: Temporary state changes
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    DEFAULT.self_assign_multi_keys(n, backup=backup)


@set_module_as('brainstate.random')
def clone_rng(seed_or_key: SeedOrKey = None, clone: bool = True) -> RandomState:
    """
    Create a clone of the random state or a new random state.

    This function provides a flexible way to create independent random states,
    either by cloning the current global state or by creating a new state with
    a specific seed or key. Cloned states are independent and don't affect each
    other when used for random number generation.

    Args:
        seed_or_key: Optional seed (integer) or JAX random key to initialize
            the new random state. If None, uses the current global state.
        clone: Whether to clone the default random state. If False and
            seed_or_key is None, returns the global state directly (not recommended
            for most use cases as it shares state).

    Returns:
        A RandomState instance that can be used independently for random
        number generation.

    Example:
        Clone the current global state:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> rng1 = brainstate.random.clone_rng()
        >>> rng2 = brainstate.random.clone_rng()
        >>> # rng1 and rng2 are independent copies

        Create a new state with specific seed:

        >>> rng_fixed = brainstate.random.clone_rng(123)
        >>> # Always produces the same sequences when reset to seed 123

        Use for independent computations:

        >>> rng = brainstate.random.clone_rng(456)
        >>> values1 = rng.normal(size=5)
        >>> values2 = rng.normal(size=5)
        >>> # values1 and values2 are different but reproducible

    Note:
        Cloned random states are completely independent. Changes to one state
        (like advancing through random number generation) don't affect others.

    See Also:
        - :func:`default_rng`: Get or create a random state
        - :func:`seed`: Set the global random seed
        - :class:`RandomState`: The random state class
    """
    if seed_or_key is None:
        return DEFAULT.clone() if clone else DEFAULT
    else:
        return RandomState(seed_or_key)


@set_module_as('brainstate.random')
def default_rng(seed_or_key: SeedOrKey = None) -> RandomState:
    """
    Get the default random state or create a new one with specified seed.

    This function provides access to the global random state used throughout
    BrainState, or creates a new independent random state if a seed is provided.
    It's the primary interface for obtaining random state objects in BrainState.

    Args:
        seed_or_key: Optional seed (integer) or JAX random key. If None,
            returns the global default random state. If provided, creates
            a new independent RandomState with the specified seed.

    Returns:
        The default RandomState if seed_or_key is None, otherwise a new
        RandomState initialized with the provided seed or key.

    Example:
        Get the global random state:

        >>> import brainstate
        >>> rng = brainstate.random.default_rng()
        >>> # rng is the global random state used by brainstate.random functions

        Create a new independent random state:

        >>> rng_local = brainstate.random.default_rng(42)
        >>> values = rng_local.normal(size=10)

        Use for reproducible local computations:

        >>> def reproducible_computation():
        ...     local_rng = brainstate.random.default_rng(12345)
        ...     return local_rng.uniform(size=5)
        >>> result1 = reproducible_computation()
        >>> result2 = reproducible_computation()
        >>> assert np.allclose(result1, result2)  # Always the same

    Note:
        When seed_or_key is None, this returns the actual global state object.
        Modifications to this state (through random number generation) will
        affect all subsequent calls to global random functions.

    See Also:
        - :func:`clone_rng`: Create independent clones of random states
        - :func:`seed`: Set the global random seed
        - :class:`RandomState`: The underlying random state implementation
    """
    if seed_or_key is None:
        return DEFAULT
    else:
        return RandomState(seed_or_key)


@set_module_as('brainstate.random')
def set_key(seed_or_key: SeedOrKey) -> None:
    """
    Set a new random key for the global random state.

    This function updates the global random state with a new key, which can be
    either an integer seed or a JAX PRNG key. All subsequent calls to global
    random functions will use this new key state.

    Args:
        seed_or_key: The new random key to set. Can be:
            - An integer seed (will be converted to a JAX PRNG key)
            - A JAX PRNG key array
            - A numpy array representing a PRNG key

    Raises:
        ValueError: If the provided key is not in a valid format.

    Example:
        Set with integer seed:

        >>> import brainstate
        >>> brainstate.random.set_key(42)
        >>> values1 = brainstate.random.rand(3)

        Set with JAX key:

        >>> import jax
        >>> key = jax.random.key(123)
        >>> brainstate.random.set_key(key)
        >>> values2 = brainstate.random.rand(3)

        Restore reproducible state:

        >>> brainstate.random.set_key(42)
        >>> # Now random functions will produce the same sequences as first example

    Note:
        This function immediately changes the global random state. All threads
        and computations using the global random functions will be affected.

    See Also:
        - :func:`get_key`: Get the current random key
        - :func:`seed`: Set seed (also affects NumPy)
        - :func:`restore_key`: Restore a backed up key
    """
    if isinstance(seed_or_key, int):
        # Create key using appropriate JAX function based on version
        key = jax.random.PRNGKey(seed_or_key) if use_prng_key else jax.random.key(seed_or_key)
    elif isinstance(seed_or_key, (jax.numpy.ndarray, np.ndarray)):
        if jax.numpy.issubdtype(seed_or_key.dtype, jax.dtypes.prng_key):
            key = seed_or_key
        elif seed_or_key.size == 2 and seed_or_key.dtype == jax.numpy.uint32:
            key = seed_or_key
        else:
            raise ValueError(
                f"seed_or_key should be an integer, a JAX PRNG key, or a uint32 array of size 2. "
                f"Got array with dtype {seed_or_key.dtype} and size {seed_or_key.size}."
            )
    else:
        raise ValueError(
            f"seed_or_key must be an integer or a JAX-compatible array. "
            f"Got {type(seed_or_key)}."
        )
    DEFAULT.set_key(key)


@set_module_as('brainstate.random')
def get_key():
    """
    Get the current global random key.

    This function returns the current random key used by the global random state.
    The returned key represents the internal state of the JAX PRNG and can be used
    to restore the random state later or to create independent random number generators.

    Returns:
        The current JAX PRNG key as a numpy array. This is typically a 2-element
        uint32 array representing the internal state of the random number generator.

    Example:
        Get and store the current random state:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> current_key = brainstate.random.get_key()
        >>> print(current_key.shape)
        (2,)

        Use the key to restore state later:

        >>> # Generate some random numbers
        >>> values1 = brainstate.random.rand(3)
        >>> # Restore the previous state
        >>> brainstate.random.set_key(current_key)
        >>> values2 = brainstate.random.rand(3)
        >>> # values1 and values2 will be identical

        Compare keys for debugging:

        >>> brainstate.random.seed(123)
        >>> key1 = brainstate.random.get_key()
        >>> brainstate.random.seed(123)
        >>> key2 = brainstate.random.get_key()
        >>> assert jax.numpy.array_equal(key1, key2)  # Same seed gives same key

    Note:
        The returned key is a snapshot of the current state. Subsequent calls to
        random functions will advance the internal state, so calling get_key()
        again will return a different key unless the state is reset.

    See Also:
        - :func:`set_key`: Set a new random key
        - :func:`seed`: Set the random seed (also affects NumPy)
        - :func:`split_key`: Create new keys from current state
        - :func:`seed_context`: Temporary seed changes with automatic restoration

    """
    return DEFAULT.value


@set_module_as('brainstate.random')
def seed(seed_or_key: SeedOrKey = None):
    """
    Set the global random seed for both JAX and NumPy.

    This function initializes the global random state with a new seed, affecting
    both JAX and NumPy random number generators. It ensures reproducible random
    number generation across the entire BrainState ecosystem.

    Args:
        seed_or_key: The seed or key to set. Can be:
            - None: Generates a random seed automatically
            - int: An integer seed (0 to 2^32-1)
            - JAX PRNG key: A JAX random key array
            If None, a random seed is generated using NumPy's random generator.

    Raises:
        ValueError: If seed_or_key is not a valid seed format (not an integer,
            valid JAX key, or None).

    Example:
        Set a specific seed for reproducible results:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> values1 = brainstate.random.rand(3)
        >>> brainstate.random.seed(42)  # Reset to same seed
        >>> values2 = brainstate.random.rand(3)
        >>> assert np.allclose(values1, values2)  # Same values

        Use automatic random seeding:

        >>> brainstate.random.seed()  # Uses random seed
        >>> # Each call will produce different sequences

        Use with JAX keys:

        >>> import jax
        >>> key = jax.random.key(123)
        >>> brainstate.random.seed(key)
        >>> # Now both JAX and NumPy use consistent seeds

        Ensure reproducibility in scientific experiments:

        >>> def experiment():
        ...     brainstate.random.seed(12345)  # Fixed seed for reproducibility
        ...     data = brainstate.random.normal(size=(100, 10))
        ...     return data.mean()
        >>> result1 = experiment()
        >>> result2 = experiment()
        >>> assert result1 == result2  # Always same result

    Note:
        - This function affects the global random state used by all BrainState
          random functions and NumPy's global random state.
        - When using automatic seeding (seed_or_key=None), NumPy's seed is not
          set to maintain its current state.
        - JAX compilation is handled automatically with compile-time evaluation.
        - For JAX keys, only the first element is used to seed NumPy to maintain
          compatibility between the two random systems.

    See Also:
        - :func:`set_key`: Set only the JAX random key
        - :func:`get_key`: Get the current random key
        - :func:`seed_context`: Temporary seed changes
        - :func:`split_key`: Create independent random keys

    """
    with jax.ensure_compile_time_eval():
        _set_numpy_seed = True
        if seed_or_key is None:
            seed_or_key = np.random.randint(0, 100000)
            _set_numpy_seed = False

        # numpy random seed
        if _set_numpy_seed:
            try:
                if np.size(seed_or_key) == 1:  # seed
                    np.random.seed(seed_or_key)
                elif np.size(seed_or_key) == 2:  # jax random key
                    np.random.seed(seed_or_key[0])
                else:
                    raise ValueError(f"seed_or_key should be an integer or a tuple of two integers.")
            except jax.errors.TracerArrayConversionError:
                pass

    # jax random seed
    DEFAULT.seed(seed_or_key)


@contextmanager
@set_module_as('brainstate.random')
def seed_context(seed_or_key: SeedOrKey):
    """
    Context manager for temporary random seed changes with automatic restoration.

    This context manager temporarily changes the global random seed for the duration
    of the block, then automatically restores the previous random state when exiting.
    It's ideal for ensuring reproducible computations in specific code sections without
    permanently affecting the global random state.

    Args:
        seed_or_key: The temporary seed or key to use within the context. Can be:
            - int: An integer seed for reproducible sequences
            - JAX PRNG key: A JAX random key array
            The seed affects both JAX and NumPy random states during the context.

    Yields:
        None: The context manager doesn't yield any value, but provides a
        controlled random environment for the enclosed code block.

    Example:
        Reproducible computations without affecting global state:

        >>> import brainstate
        >>> # Global state remains unaffected
        >>> global_values1 = brainstate.random.rand(2)
        >>>
        >>> with brainstate.random.seed_context(42):
        ...     temp_values1 = brainstate.random.rand(2)
        ...     print(f"First run: {temp_values1}")
        [0.95598125 0.4032725 ]
        >>>
        >>> with brainstate.random.seed_context(42):
        ...     temp_values2 = brainstate.random.rand(2)
        ...     print(f"Second run: {temp_values2}")
        [0.95598125 0.4032725 ]
        >>>
        >>> # Values are identical within context
        >>> assert np.allclose(temp_values1, temp_values2)
        >>>
        >>> # Global state continues from where it left off
        >>> global_values2 = brainstate.random.rand(2)

        Nested contexts for complex scenarios:

        >>> with brainstate.random.seed_context(123):
        ...     outer_values = brainstate.random.rand(2)
        ...     with brainstate.random.seed_context(456):
        ...         inner_values = brainstate.random.rand(2)
        ...     # Outer context is restored here
        ...     outer_values2 = brainstate.random.rand(2)

        Exception safety - state is restored even on errors:

        >>> try:
        ...     with brainstate.random.seed_context(789):
        ...         some_values = brainstate.random.rand(3)
        ...         raise ValueError("Something went wrong")
        ... except ValueError:
        ...     pass
        >>> # Random state is properly restored

        Testing reproducible algorithms:

        >>> def test_algorithm():
        ...     with brainstate.random.seed_context(42):
        ...         data = brainstate.random.normal(size=(100,))
        ...         return data.mean()
        >>>
        >>> result1 = test_algorithm()
        >>> result2 = test_algorithm()
        >>> assert result1 == result2  # Always same result

    Note:
        - The context manager saves and restores the complete JAX random state
        - NumPy's random state is also temporarily modified during the context
        - Nested contexts work correctly - each level restores its own state
        - Exception safety is guaranteed - random state is restored even if
          exceptions occur within the context
        - This is more convenient than manually saving/restoring state with
          get_key() and set_key()

    See Also:
        - :func:`seed`: Permanently set the global random seed
        - :func:`get_key`: Get the current random key for manual state management
        - :func:`set_key`: Set the random key for manual state management
        - :func:`clone_rng`: Create independent random states

    """
    # get the old random key
    old_jrand_key = DEFAULT.value
    try:
        # set the seed of jax random state
        DEFAULT.seed(seed_or_key)
        yield
    finally:
        # restore the random state
        DEFAULT.seed(old_jrand_key)
