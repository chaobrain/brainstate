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

import threading
from contextlib import contextmanager
from typing import Union

import numpy as np

__all__ = [
    'FloatScalar',
    'IntScalar',
    'set_numba_environ',
]

FloatScalar = Union[
    np.number,  # NumPy scalar types
    float,  # Python scalar types
]

IntScalar = Union[
    np.number,  # NumPy scalar types
    int,  # Python scalar types
]


class EnvironContext(threading.local):
    def __init__(self):
        # default environment settings
        self.numba_parallel: bool = False
        self.numba_setting: dict = dict(nogil=True, fastmath=True)


environ = EnvironContext()


@contextmanager
def set_numba_environ(
    parallel_if_possible: int | bool = None,
    **kwargs
) -> None:
    """
    Enable Numba parallel execution if possible.
    """
    old_parallel = environ.numba_parallel
    old_setting = environ.numba_setting.copy()

    try:
        environ.numba_setting.update(kwargs)
        if parallel_if_possible is not None:
            if isinstance(parallel_if_possible, bool):
                environ.numba_parallel = parallel_if_possible
            elif isinstance(parallel_if_possible, int):
                environ.numba_parallel = True
                assert parallel_if_possible > 0, 'The number of threads must be a positive integer.'
                import numba  # pylint: disable=import-outside-toplevel
                numba.set_num_threads(parallel_if_possible)
            else:
                raise ValueError('The argument `parallel_if_possible` must be a boolean or an integer.')
    finally:
        environ.numba_parallel = old_parallel
        environ.numba_setting = old_setting
