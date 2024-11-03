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

from __future__ import annotations

import functools
from typing import Callable, Sequence, Union

from brainstate.random import DEFAULT, RandomState
from brainstate.typing import Missing

__all__ = [
    'restore_rngs'
]


class RngRestore:
    """
    Backup and restore the random state of a sequence of RandomState instances.
    """

    def __init__(self, rngs: Sequence[RandomState]):
        self.rngs: Sequence[RandomState] = rngs
        self.rng_keys = []

    def backup(self):
        """
        Backup the current random key of the RandomState instances.
        """
        self.rng_keys = [rng.value for rng in self.rngs]

    def restore(self):
        """
        Restore the random key of the RandomState instances.
        """
        for rng, key in zip(self.rngs, self.rng_keys):
            rng.restore_value(key)
        self.rng_keys = []


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
    Backup the current random state and restore it after the function call.

    Parameters
    ----------
    fn : Callable, optional
        The function to be wrapped.
    rngs : Union[RandomState, Sequence[RandomState]]
        The random state to be backed up and restored. If not provided, the default RandomState instance will be used.

    Returns
    -------
    Callable
        The wrapped function.
    """
    if isinstance(fn, Missing):
        return functools.partial(restore_rngs, rngs=rngs)

    if isinstance(rngs, RandomState):
        rngs = [rngs]
    assert isinstance(rngs, Sequence), 'rngs must be a RandomState or a sequence of RandomState instances.'
    for rng in rngs:
        assert isinstance(rng, RandomState), 'rngs must be a RandomState or a sequence of RandomState instances.'
    return _rng_backup(fn, rngs=rngs)
