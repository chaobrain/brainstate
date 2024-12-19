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
from typing import Optional

import jax

try:
    from tqdm.auto import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = None

__all__ = [
    'ProgressBar',
]


class ProgressBar(object):
    __module__ = "brainstate.compile"

    def __init__(self, freq: Optional[int] = None, count: Optional[int] = None, **kwargs):
        self.print_freq = freq
        if isinstance(freq, int):
            assert freq > 0, "Print rate should be > 0."
        self.print_count = count
        if self.print_freq is not None and self.print_count is not None:
            raise ValueError("Cannot specify both count and freq.")
        for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
            kwargs.pop(kwarg, None)
        self.kwargs = kwargs
        if tqdm is None:
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
        desc = kwargs.pop("desc", f"Running for {n:,} iterations")
        message = kwargs.pop("message", desc)
        return ProgressBarRunner(n, message, freq, remainder, **kwargs)


class ProgressBarRunner(object):
    __module__ = "brainstate.compile"

    def __init__(self, n: int, message, print_freq: int, remainder: int, **kwargs):
        self.tqdm_bars = {}
        self.kwargs = kwargs
        self.n = n
        self.print_freq = print_freq
        self.remainder = remainder
        self.message = message

    def _define_tqdm(self):
        self.tqdm_bars[0] = tqdm(range(self.n), **self.kwargs)
        self.tqdm_bars[0].set_description(self.message, refresh=False)

    def _update_tqdm(self):
        self.tqdm_bars[0].update(self.print_freq)

    def _close_tqdm(self):
        if self.remainder > 0:
            self.tqdm_bars[0].update(self.remainder)
        self.tqdm_bars[0].close()

    def _tqdm(self, is_init, is_print, is_final):
        if is_init:
            self.tqdm_bars[0] = tqdm(range(self.n), **self.kwargs)
            self.tqdm_bars[0].set_description(self.message, refresh=False)
        if is_print:
            self.tqdm_bars[0].update(self.print_freq)
        if is_final:
            if self.remainder > 0:
                self.tqdm_bars[0].update(self.remainder)
            self.tqdm_bars[0].close()

    def __call__(self, iter_num, *args, **kwargs):
        # jax.debug.callback(
        #     self._tqdm,
        #     iter_num == 0,
        #     (iter_num + 1) % self.print_freq == 0,
        #     iter_num == self.n - 1
        # )

        _ = jax.lax.cond(
            iter_num == 0,
            lambda: jax.debug.callback(self._define_tqdm, ordered=True),
            lambda: None,
        )
        _ = jax.lax.cond(
            iter_num % self.print_freq == (self.print_freq - 1),
            lambda: jax.debug.callback(self._update_tqdm, ordered=True),
            lambda: None,
        )
        _ = jax.lax.cond(
            iter_num == self.n - 1,
            lambda: jax.debug.callback(self._close_tqdm, ordered=True),
            lambda: None,
        )

