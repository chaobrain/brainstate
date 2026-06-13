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

Examples
--------
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

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import jax
import numpy as np

from brainstate._utils import set_module_as
from brainstate.typing import SeedOrKey
from ._impl import _format_key
from ._state import RandomState, DEFAULT

__all__ = [
    'seed',
    'set_key',
    'get_key',
    'get_key_data',
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

    Notes
    -----
        This operation requires that a backup was previously created. If no backup
        exists, this function may not have any effect or may restore to an initial state.

    Examples
    --------
        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> original_key = brainstate.random.get_key()
        >>> brainstate.random.seed(123)  # Change the seed
        >>> brainstate.random.restore_key()  # Restore to previous state
        >>> assert np.array_equal(brainstate.random.get_key(), original_key)

    See Also
    --------
    set_key : Set a new random key
    get_key : Get the current random key
    seed_context : Temporary seed changes with automatic restoration
    """
    DEFAULT.restore_key()


@set_module_as('brainstate.random')
def split_key(n: Optional[int] = None, backup: bool = False) -> jax.Array:
    """
    Create new random key(s) from the current seed.

    This function generates one or more independent random keys by splitting the
    current global random state. It follows JAX's random paradigm, ensuring that
    each split key produces statistically independent random sequences.

    Parameters
    ----------
    n
        The number of keys to generate. If None, returns a single key.
        If an integer, returns an array of n keys.
    backup
        Whether to backup the current key before splitting. This allows
        restoration of the original state using :func:`restore_key`.

    Returns
    -------
    If n is None: A single JAX PRNG key.
    If n is an integer: An array of n independent JAX PRNG keys.

    Examples
    --------
        Generate a single key:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> key = brainstate.random.split_key()
        >>> print(key.shape)
        ()

        Generate multiple keys for parallel computation:

        >>> keys = brainstate.random.split_key(4)
        >>> print(keys.shape)
        (4,)

        Use with backup for state restoration:

        >>> original_key = brainstate.random.get_key()
        >>> keys = brainstate.random.split_key(2, backup=True)
        >>> brainstate.random.restore_key()
        >>> assert np.array_equal(brainstate.random.get_key(), original_key)

    Notes
    -----
        This function advances the global random state. Each call produces
        different keys unless the state is reset.

    See Also
    --------
    split_keys : Convenience function for multiple keys
    seed : Set the random seed
    restore_key : Restore backed up key
    """
    return DEFAULT.split_key(n=n, backup=backup)


@set_module_as('brainstate.random')
def split_keys(n: int, backup: bool = False) -> jax.Array:
    """
    Create multiple independent random keys from the current seed.

    This is a convenience function that generates exactly n independent random keys
    by splitting the current global random state. It's commonly used internally by
    parallel computation functions like `pmap` and `vmap` to ensure that each
    parallel thread gets a unique random key.

    Parameters
    ----------
    n
        The number of independent keys to generate. Must be a positive integer.
    backup
        Whether to backup the current key before splitting. If True,
        the original key can be restored using :func:`restore_key`.

    Returns
    -------
    An array of n independent JAX typed PRNG keys with shape (n,).

    Raises
    ------
    ValueError
        If n is not a positive integer.

    Examples
    --------
        Generate keys for parallel computation:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> keys = brainstate.random.split_keys(4)
        >>> print(keys.shape)
        (4,)

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

    Notes
    -----
        This function is equivalent to calling :func:`split_key` with n as an argument.
        It's provided as a convenience function with a more explicit name for clarity.

    See Also
    --------
    split_key : More general key splitting function
    self_assign_multi_keys : Assign multiple keys to global state
    seed_context : Temporary seed changes
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

    Parameters
    ----------
    n
        The number of independent keys to pre-generate and assign.
        Must be a positive integer.
    backup
        Whether to backup the current random state before assignment.
        If True, the original state can be restored using :func:`restore_key`.

    Raises
    ------
    ValueError
        If n is not a positive integer.

    Examples
    --------
        Prepare for parallel computation:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> # Prepare 4 independent keys for parallel access
        >>> brainstate.random.self_assign_multi_keys(4)

        Use in parallel context:

        >>> # The random state now has 4 independent keys ready for use
        >>> # Each parallel thread can access a different key

    Notes
    -----
        This is an advanced function primarily used internally for optimizing
        parallel random number generation. In most cases, :func:`split_keys`
        provides a more straightforward interface for parallel computation.

    See Also
    --------
    split_keys : Generate multiple independent keys
    restore_key : Restore backed up state
    seed_context : Temporary state changes
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

    Parameters
    ----------
    seed_or_key
        Optional seed (integer) or JAX random key to initialize
        the new random state. If None, uses the current global state.
    clone
        Whether to clone the default random state. If False and
        seed_or_key is None, returns the global state directly (not recommended
        for most use cases as it shares state).

    Returns
    -------
    A RandomState instance that can be used independently for random
    number generation.

    Examples
    --------
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

    Notes
    -----
        Cloned random states are completely independent. Changes to one state
        (like advancing through random number generation) don't affect others.

    See Also
    --------
    default_rng : Get or create a random state
    seed : Set the global random seed
    RandomState : The random state class
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

    Parameters
    ----------
    seed_or_key
        Optional seed (integer) or JAX random key. If None,
        returns the global default random state. If provided, creates
        a new independent RandomState with the specified seed.

    Returns
    -------
    The default RandomState if seed_or_key is None, otherwise a new
    RandomState initialized with the provided seed or key.

    Examples
    --------
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

    Notes
    -----
        When seed_or_key is None, this returns the actual global state object.
        Modifications to this state (through random number generation) will
        affect all subsequent calls to global random functions.

    See Also
    --------
    clone_rng : Create independent clones of random states
    seed : Set the global random seed
    RandomState : The underlying random state implementation
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
    an integer seed, a JAX typed PRNG key, or a legacy ``uint32[2]`` key array
    (auto-wrapped into a typed key). All subsequent calls to global random
    functions will use this new key state.

    Parameters
    ----------
    seed_or_key
        The new random key to set. Can be:
        - An integer seed (converted to a typed JAX key via ``jax.random.key``)
        - A JAX typed PRNG key
        - A legacy ``uint32[2]`` array (auto-wrapped into a typed key)

    Raises
    ------
    TypeError
        If the provided key is not in a valid format.

    Examples
    --------
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

    Notes
    -----
        This function immediately changes the global random state. All threads
        and computations using the global random functions will be affected.

    See Also
    --------
    get_key : Get the current random key
    get_key_data : Get the current key as raw ``uint32[2]`` data
    seed : Set seed (also affects NumPy)
    restore_key : Restore a backed up key
    """
    DEFAULT.set_key(_format_key(seed_or_key))


@set_module_as('brainstate.random')
def get_key() -> jax.Array:
    """
    Get the current global random key.

    This function returns the current random key used by the global random state.
    The returned key represents the internal state of the JAX PRNG and can be used
    to restore the random state later or to create independent random number generators.

    Returns
    -------
    The current JAX typed PRNG key (a scalar array of dtype ``key<...>``).
    Use :func:`get_key_data` if you need the raw ``uint32[2]`` representation.

    Examples
    --------
        Get and store the current random state:

        >>> import brainstate
        >>> brainstate.random.seed(42)
        >>> current_key = brainstate.random.get_key()
        >>> print(current_key.shape)
        ()

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

    Notes
    -----
        The returned key is a snapshot of the current state. Subsequent calls to
        random functions will advance the internal state, so calling get_key()
        again will return a different key unless the state is reset.

    See Also
    --------
    set_key : Set a new random key
    seed : Set the random seed (also affects NumPy)
    split_key : Create new keys from current state
    seed_context : Temporary seed changes with automatic restoration

    """
    return DEFAULT.value


@set_module_as('brainstate.random')
def get_key_data() -> jax.Array:
    """
    Get the current global random key as raw ``uint32[2]`` data.

    This is the legacy-interop counterpart of :func:`get_key`. It returns the
    underlying key data (a 2-element ``uint32`` array) extracted from the current
    typed key via :func:`jax.random.key_data`. This is useful when interfacing with
    code that still expects the old ``uint32[2]`` key representation.

    Returns
    -------
    A ``uint32`` array of shape ``(2,)`` holding the current key's raw data.

    Examples
    --------
        >>> import brainstate
        >>> import jax.random as jr
        >>> brainstate.random.set_key(jr.key(11))
        >>> data = brainstate.random.get_key_data()
        >>> print(data.shape, data.dtype)
        (2,) uint32

    See Also
    --------
    get_key : Get the current typed key
    set_key : Set a new random key (accepts raw ``uint32[2]`` too)
    """
    return jax.random.key_data(DEFAULT.value)


@set_module_as('brainstate.random')
def seed(seed_or_key: Optional[SeedOrKey] = None) -> None:
    """
    Set the global random seed for both JAX and NumPy.

    This function initializes the global random state with a new seed, affecting
    both JAX and NumPy random number generators. It ensures reproducible random
    number generation across the entire BrainState ecosystem.

    Parameters
    ----------
    seed_or_key
        The seed or key to set. Can be:
        - None: Generates a random seed automatically
        - int: An integer seed. Any Python integer is accepted; it is reduced
          modulo ``2**32`` for NumPy (matching JAX, which reduces an integer
          seed to its low 32 bits), so values outside ``[0, 2**32-1]`` (e.g.
          ``hash(...)``, ``time.time_ns()`` or negative values) are valid.
        - JAX PRNG key: A JAX random key array
        If None, a random seed is generated without disturbing NumPy's global
        random state.

    Raises
    ------
    ValueError
        If seed_or_key is not a valid seed format (not an integer,
        valid JAX key, or None).

    Examples
    --------
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

    Notes
    -----
        - This function affects the global random state used by all BrainState
          random functions and NumPy's global random state.
        - When using automatic seeding (seed_or_key=None), NumPy's global random
          state is left untouched: the auto-key is drawn from an independent
          ``numpy.random.default_rng`` Generator, so the current state is
          maintained.
        - JAX compilation is handled automatically with compile-time evaluation.
        - The input is first normalized to a typed JAX key; the first element of
          that key's raw ``uint32[2]`` data is then used to seed NumPy, keeping
          the two random systems consistent. Because the JAX key is validated
          before any NumPy mutation, an invalid input raises without leaving the
          JAX and NumPy states out of sync.

    See Also
    --------
    set_key : Set only the JAX random key
    get_key : Get the current random key
    seed_context : Temporary seed changes
    split_key : Create independent random keys

    """
    with jax.ensure_compile_time_eval():
        if seed_or_key is None:
            # Automatic seeding: draw a full uint32[2] key from a *fresh*
            # ``default_rng`` Generator so the legacy global ``np.random`` state
            # is left untouched (honoring the docstring's "maintain its current
            # state") and the full 2**32 entropy range is used (matching
            # ``RandomState.seed(None)`` rather than a reduced range).
            seed_or_key = np.random.default_rng().integers(0, 2 ** 32, size=2, dtype=np.uint32)
            DEFAULT.seed(seed_or_key)
            return

        # Normalize a NumPy integer scalar (e.g. ``np.int64(...)``) to a Python
        # int so it follows the int-seed path below (``_format_key`` itself only
        # accepts python ints, typed keys, or uint32[2] arrays).
        if isinstance(seed_or_key, np.integer):
            seed_or_key = int(seed_or_key)

        # Validate/normalize the JAX key *first* so that, on invalid input, no
        # global state is mutated. ``_format_key`` raises TypeError/ValueError
        # for anything that is not an int, a typed key, or a uint32[2] array,
        # which is strictly stricter than the NumPy-seeding step below; doing it
        # first eliminates the partial-mutation window (NumPy seeded, JAX not).
        key = _format_key(seed_or_key)

        # Set the JAX state from the validated key.
        DEFAULT.set_key(key)

        # Seed NumPy. ``np.random.seed`` only accepts [0, 2**32-1]; reduce the
        # value modulo 2**32 (matching JAX, which reduces an integer seed to its
        # low 32 bits) so out-of-range / negative ints no longer raise.
        #
        #   * For an integer seed ``v`` we seed NumPy with ``v % 2**32`` -- this
        #     equals the low word of ``jax.random.key(v)`` so the two systems
        #     stay consistent, and it preserves the historical NumPy stream for
        #     in-range ints (``np.random.seed(v) == np.random.seed(v % 2**32)``).
        #   * For a typed / legacy ``uint32[2]`` key we seed NumPy from the first
        #     raw word of the key data, matching the documented "first element of
        #     the key is used to seed NumPy" behavior.
        try:
            if isinstance(seed_or_key, int):
                np_seed = seed_or_key % (2 ** 32)
            else:
                np_seed = int(jax.random.key_data(key)[0]) % (2 ** 32)
            np.random.seed(np_seed)
        except (jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError):
            # Inside a jit trace the key is abstract; skip NumPy seeding.
            pass


@contextmanager
@set_module_as('brainstate.random')
def seed_context(seed_or_key: SeedOrKey) -> Iterator[None]:
    """
    Context manager for temporary random seed changes with automatic restoration.

    This context manager temporarily changes the global random seed for the duration
    of the block, then automatically restores the previous random state when exiting.
    It's ideal for ensuring reproducible computations in specific code sections without
    permanently affecting the global random state.

    Parameters
    ----------
    seed_or_key
        The temporary seed or key to use within the context. Can be:
        - int: An integer seed for reproducible sequences
        - JAX PRNG key: A JAX random key array
        The seed affects both JAX and NumPy random states during the context.

    Yields
    ------
    None
        The context manager doesn't yield any value, but provides a
        controlled random environment for the enclosed code block.

    Examples
    --------
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

    Notes
    -----
        - The context manager saves and restores the complete JAX random state
        - NumPy's global random state is also seeded on entry and fully restored
          on exit (via ``numpy.random.get_state``/``set_state``), so NumPy-backed
          randomness inside the block is reproducible and leaves no side effect
        - Nested contexts work correctly - each level restores its own state
        - Exception safety is guaranteed - random state is restored even if
          exceptions occur within the context
        - This is more convenient than manually saving/restoring state with
          get_key() and set_key()

    See Also
    --------
    seed : Permanently set the global random seed
    get_key : Get the current random key for manual state management
    set_key : Set the random key for manual state management
    clone_rng : Create independent random states

    """
    # Snapshot BOTH the JAX key and the full NumPy generator state so the
    # documented "affects both JAX and NumPy" contract holds and both are
    # restored on exit. We restore the NumPy state with ``set_state`` (not a
    # re-seed) because seeding on entry advances NumPy's position; re-seeding
    # could not reproduce a mid-stream Mersenne-Twister state.
    old_jrand_key = DEFAULT.value
    old_np_state = np.random.get_state()
    try:
        # Seed both the JAX key and NumPy via the module-level ``seed`` so the
        # two systems stay consistent. ``seed(None)`` deliberately leaves
        # NumPy's state untouched, so feed it a concrete key in that case to
        # honor the documented NumPy-seeding behavior of this context manager.
        if seed_or_key is None:
            seed(int(np.random.default_rng().integers(0, 2 ** 32)))
        else:
            seed(seed_or_key)
        yield
    finally:
        # restore both the JAX key and the full NumPy MT state
        DEFAULT.seed(old_jrand_key)
        np.random.set_state(old_np_state)
