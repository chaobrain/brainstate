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

import copy
import importlib.util
from typing import Optional, Callable, Any, Tuple, Dict

import jax

tqdm_installed = importlib.util.find_spec('tqdm') is not None

__all__ = [
    'ProgressBar',
]

Index = int
Carray = Any
Output = Any


class ProgressBar(object):
    """
    A progress bar for tracking the progress of a jitted for-loop computation.
    """
    __module__ = "brainstate.compile"

    def __init__(
        self,
        freq: Optional[int] = None,
        count: Optional[int] = None,
        desc: Optional[Tuple[str, Callable[[Dict], Dict]]] = None,
        **kwargs
    ):
        # print rate
        self.print_freq = freq
        if isinstance(freq, int):
            assert freq > 0, "Print rate should be > 0."

        # print count
        self.print_count = count
        if self.print_freq is not None and self.print_count is not None:
            raise ValueError("Cannot specify both count and freq.")

        # other parameters
        for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
            kwargs.pop(kwarg, None)
        self.kwargs = kwargs

        # description
        if desc is not None:
            assert isinstance(desc, (tuple, list)), 'Description should be a tuple or list.'
            assert isinstance(desc[0], str), 'Description should be a string.'
            assert callable(desc[1]), 'Description should be a callable.'
        self.desc = desc

        # check if tqdm is installed
        if not tqdm_installed:
            raise ImportError("tqdm is not installed.")

    def init(self, n: int):
        kwargs = copy.copy(self.kwargs)
        freq = self.print_freq
        count = self.print_count
        if count is not None:
            freq, remainder = divmod(n, count)
            if freq == 0:
                raise ValueError(f"Count {count} is too large for n {n}.")
        elif freq is None:
            if n > 20:
                freq = int(n / 20)
            else:
                freq = 1
            remainder = n % freq
        else:
            if freq < 1:
                raise ValueError(f"Print rate should be > 0 got {freq}")
            elif freq > n:
                raise ValueError("Print rate should be less than the "
                                 f"number of steps {n}, got {freq}")
            remainder = n % freq

        message = f"Running for {n:,} iterations" if self.desc is None else self.desc
        return ProgressBarRunner(n, freq, remainder, message, **kwargs)


class ProgressBarRunner(object):
    __module__ = "brainstate.compile"

    def __init__(
        self,
        n: int,
        print_freq: int,
        remainder: int,
        message: str | Tuple[str, Callable[[Dict], Dict]],
        **kwargs
    ):
        self.tqdm_bars = {}
        self.kwargs = kwargs
        self.n = n
        self.print_freq = print_freq
        self.remainder = remainder
        self.message = message

    def _define_tqdm(self, x: dict):
        from tqdm.auto import tqdm
        self.tqdm_bars[0] = tqdm(range(self.n), **self.kwargs)
        if isinstance(self.message, str):
            self.tqdm_bars[0].set_description(self.message, refresh=False)
        else:
            self.tqdm_bars[0].set_description(self.message[0].format(**x), refresh=True)

    def _update_tqdm(self, x: dict):
        self.tqdm_bars[0].update(self.print_freq)
        if not isinstance(self.message, str):
            self.tqdm_bars[0].set_description(self.message[0].format(**x), refresh=True)

    def _close_tqdm(self, x: dict):
        if self.remainder > 0:
            self.tqdm_bars[0].update(self.remainder)
            if not isinstance(self.message, str):
                self.tqdm_bars[0].set_description(self.message[0].format(**x), refresh=True)
        self.tqdm_bars[0].close()

    def __call__(self, iter_num, **kwargs):
        data = dict(i=iter_num, **kwargs)
        data = dict() if isinstance(self.message, str) else self.message[1](data)
        assert isinstance(data, dict), 'Description function should return a dictionary.'

        _ = jax.lax.cond(
            iter_num == 0,
            lambda x: jax.debug.callback(self._define_tqdm, x, ordered=True),
            lambda x: None,
            data
        )
        _ = jax.lax.cond(
            iter_num % self.print_freq == (self.print_freq - 1),
            lambda x: jax.debug.callback(self._update_tqdm, x, ordered=True),
            lambda x: None,
            data
        )
        _ = jax.lax.cond(
            iter_num == self.n - 1,
            lambda x: jax.debug.callback(self._close_tqdm, x, ordered=True),
            lambda x: None,
            data
        )
