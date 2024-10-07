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

from contextlib import contextmanager

import jax
import numpy as np

from brainstate.typing import SeedOrKey
from ._rand_state import RandomState, DEFAULT

__all__ = [
  'seed', 'default_rng', 'split_key', 'split_keys', 'seed_context',
]


def split_key():
  """Create a new seed from the current seed.

  This function is useful for the consistency with JAX's random paradigm.

  Returns
  -------
  key : jax.random.PRNGKey
    A new random key.
  """
  return DEFAULT.split_key()


def split_keys(n: int):
  """Create multiple seeds from the current seed. This is used
  internally by `pmap` and `vmap` to ensure that random numbers
  are different in parallel threads.

  Parameters
  ----------
  n : int
    The number of seeds to generate.

  Returns
  -------
  keys : jax.random.PRNGKey
    A tuple of JAX random keys.
  """
  return DEFAULT.split_keys(n)


def clone_rng(seed_or_key=None, clone: bool = True) -> RandomState:
  """Clone the random state according to the given setting.

  Args:
    seed_or_key: The seed (an integer) or the random key.
    clone: Bool. Whether clone the default random state.

  Returns:
    The random state.
  """
  if seed_or_key is None:
    return DEFAULT.clone() if clone else DEFAULT
  else:
    return RandomState(seed_or_key)


def default_rng(seed_or_key=None) -> RandomState:
  """
  Get the default random state.

  Args:
    seed_or_key: The seed (an integer) or the jax random key.

  Returns:
    The random state.
  """
  if seed_or_key is None:
    return DEFAULT
  else:
    return RandomState(seed_or_key)


def seed(seed_or_key: SeedOrKey = None):
  """Sets a new random seed.

  Parameters
  ----------
  seed_or_key: int, optional
    The random seed (an integer) or jax random key.
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
def seed_context(seed_or_key: SeedOrKey):
  """
  A context manager that sets the random seed for the duration of the block.

  Examples:

  >>> import brainstate as bst
  >>> print(bst.random.rand(2))
  [0.57721865 0.9820676 ]
  >>> print(bst.random.rand(2))
  [0.8511752  0.95312667]
  >>> with bst.random.seed_context(42):
  ...     print(bst.random.rand(2))
  [0.95598125 0.4032725 ]
  >>> with bst.random.seed_context(42):
  ...     print(bst.random.rand(2))
  [0.95598125 0.4032725 ]

  .. note::

     The context manager does not only set the seed for the AX random state, but also for the numpy random state.

  Args:
    seed_or_key: The seed (an integer) or jax random key.

  """
  old_jrand_key = DEFAULT.value
  old_np_state = np.random.get_state()
  try:
    # step 1: set the seed of numpy random state
    try:
      if np.size(seed_or_key) == 1:  # seed
        np.random.seed(seed_or_key)
      elif np.size(seed_or_key) == 2:  # jax random key
        np.random.seed(seed_or_key[0])
      else:
        raise ValueError(f"seed_or_key should be an integer or a tuple of two integers.")
    except jax.errors.TracerArrayConversionError:
      pass

    # step 2: set the seed of jax random state
    DEFAULT.seed(seed_or_key)
    yield
  finally:
    # restore the random state
    np.random.set_state(old_np_state)
    DEFAULT.seed(old_jrand_key)
