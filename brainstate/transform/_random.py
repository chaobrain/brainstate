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

import functools
from typing import Callable, Sequence, Union

from brainstate._utils import set_module_as
from brainstate.random import DEFAULT, RandomState
from brainstate.typing import Missing
from brainstate.util import PrettyObject

__all__ = [
    'restore_rngs'
]


class RngRestore(PrettyObject):
    """
    Manage backing up and restoring multiple random states.

    Parameters
    ----------
    rngs : Sequence[RandomState]
        Sequence of :class:`~brainstate.random.RandomState` instances whose
        states should be captured and restored.

    Attributes
    ----------
    rngs : Sequence[RandomState]
        Managed random-state instances.
    rng_keys : list
        Cached keys captured by :meth:`backup` until :meth:`restore` runs.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> rng = brainstate.random.RandomState(0)
        >>> restorer = brainstate.transform.RngRestore([rng])
        >>> restorer.backup()
        >>> _ = rng.random()
        >>> restorer.restore()
    """
    __module__ = 'brainstate.transform'

    def __init__(self, rngs: Sequence[RandomState]):
        """
        Initialize a restorer for the provided random states.

        Parameters
        ----------
        rngs : Sequence[RandomState]
            Random states that will be backed up and restored.
        """
        self.rngs: Sequence[RandomState] = rngs
        self.rng_keys = []

    def backup(self):
        """
        Cache the current key for each managed random state.

        Notes
        -----
        The cached keys persist until :meth:`restore` is called, after which the
        internal cache is cleared.
        """
        self.rng_keys = [rng.value for rng in self.rngs]

    def restore(self):
        """
        Restore each random state to the cached key.

        Raises
        ------
        ValueError
            Raised when the number of stored keys does not match ``rngs``.
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


@set_module_as('brainstate.transform')
def restore_rngs(
    fn: Callable = Missing(),
    rngs: Union[RandomState, Sequence[RandomState]] = DEFAULT,
) -> Callable:
    """
    Decorate a function so specified random states are restored after execution.

    Parameters
    ----------
    fn : Callable, optional
        Function to wrap. When omitted, :func:`restore_rngs` returns a decorator
        preconfigured with ``rngs``.
    rngs : Union[RandomState, Sequence[RandomState]], optional
        Random states whose keys should be backed up before running ``fn`` and
        restored afterwards. Defaults to :data:`brainstate.random.DEFAULT`.

    Returns
    -------
    Callable
        Wrapped callable that restores the random state or a partially applied
        decorator depending on how :func:`restore_rngs` is used.

    Raises
    ------
    AssertionError
        If ``rngs`` is neither a :class:`~brainstate.random.RandomState` instance nor
        a sequence of such instances.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>>
        >>> rng = brainstate.random.RandomState(0)
        >>>
        >>> @brainstate.transform.restore_rngs(rngs=rng)
        ... def sample_pair():
        ...     first = rng.random()
        ...     second = rng.random()
        ...     return first, second
        >>>
        >>> assert sample_pair()[0] == sample_pair()[0]
    """
    if isinstance(fn, Missing):
        return functools.partial(restore_rngs, rngs=rngs)

    if isinstance(rngs, RandomState):
        rngs = [rngs]
    assert isinstance(rngs, Sequence), 'rngs must be a RandomState or a sequence of RandomState instances.'
    for rng in rngs:
        assert isinstance(rng, RandomState), 'rngs must be a RandomState or a sequence of RandomState instances.'
    return _rng_backup(fn, rngs=rngs)
