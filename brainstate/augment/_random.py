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

import functools
from typing import Callable, Sequence, Union

from brainstate.random import DEFAULT, RandomState
from brainstate.typing import Missing
from brainstate.util import PrettyObject

__all__ = [
    'restore_rngs'
]


class RngRestore(PrettyObject):
    """
    Backup and restore the random state of a sequence of RandomState instances.

    This class provides functionality to save the current state of multiple
    RandomState instances and later restore them to their saved states.

    Attributes:
        rngs (Sequence[RandomState]): A sequence of RandomState instances to manage.
        rng_keys (list): A list to store the backed up random keys.
    """

    def __init__(self, rngs: Sequence[RandomState]):
        """
        Initialize the RngRestore instance.

        Args:
            rngs (Sequence[RandomState]): A sequence of RandomState instances
                whose states will be managed.
        """
        self.rngs: Sequence[RandomState] = rngs
        self.rng_keys = []

    def backup(self):
        """
        Backup the current random key of the RandomState instances.

        This method saves the current value (state) of each RandomState
        instance in the rngs sequence.
        """
        self.rng_keys = [rng.value for rng in self.rngs]

    def restore(self):
        """
        Restore the random key of the RandomState instances.

        This method restores each RandomState instance to its previously
        saved state. It raises an error if the number of saved keys doesn't
        match the number of RandomState instances.

        Raises:
            ValueError: If the number of saved random keys does not match
                the number of RandomState instances.
        """
        if len(self.rng_keys) != len(self.rngs):
            raise ValueError('The number of random keys does not match the number of random states.')
        for rng, key in zip(self.rngs, self.rng_keys):
            rng.restore_value(key)
        self.rng_keys.clear()


def _rng_backup(
    fn: Callable,
    rngs: Union[RandomState, Sequence[RandomState]]
) -> Callable:
    rng_restorer = RngRestore(rngs)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # backup the random state
        rng_restorer.backup()
        # call the function
        out = fn(*args, **kwargs)
        # restore the random state
        rng_restorer.restore()
        return out

    return wrapper


def restore_rngs(
    fn: Callable = Missing(),
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
) -> Callable:
    """
    Decorator to backup and restore the random state before and after a function call.

    This function can be used as a decorator or called directly. It ensures that the
    random state of the specified RandomState instances is preserved across function calls,
    which is useful for maintaining reproducibility in stochastic operations.

    Parameters
    ----------
    fn : Callable, optional
        The function to be wrapped. If not provided, the decorator can be used
        with parameters.
    rngs : Union[RandomState, Sequence[RandomState]], optional
        The random state(s) to be backed up and restored. This can be a single
        RandomState instance or a sequence of RandomState instances. If not provided,
        the default RandomState instance will be used.

    Returns
    -------
    Callable
        If `fn` is provided, returns the wrapped function that will backup the
        random state before execution and restore it afterwards.
        If `fn` is not provided, returns a partial function that can be used as
        a decorator with the specified `rngs`.

    Raises
    ------
    AssertionError
        If `rngs` is not a RandomState instance or a sequence of RandomState instances.

    Examples
    --------
    >>> @restore_rngs
    ... def my_random_function():
    ...     return random.random()

    >>> rng = RandomState(42)
    >>> @restore_rngs(rngs=rng)
    ... def another_random_function():
    ...     return rng.random()
    """
    if isinstance(fn, Missing):
        return functools.partial(restore_rngs, rngs=rngs)

    if isinstance(rngs, RandomState):
        rngs = [rngs]
    assert isinstance(rngs, Sequence), 'rngs must be a RandomState or a sequence of RandomState instances.'
    for rng in rngs:
        assert isinstance(rng, RandomState), 'rngs must be a RandomState or a sequence of RandomState instances.'
    return _rng_backup(fn, rngs=rngs)
