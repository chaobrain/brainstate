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

    It can be used in :py:func:`for_loop`, :py:func:`checkpointed_for_loop`, :py:func:`scan`,
    and :py:func:`checkpointed_scan` functions. Or any other jitted function that uses
    a for-loop.

    The message displayed in the progress bar can be customized by the following two methods:

    1. By passing a string to the `desc` argument. For example:

    .. code-block:: python

            ProgressBar(desc="Running 1000 iterations")

    2. By passing a tuple with a string and a callable function to the `desc` argument. The callable
       function should take a dictionary as input and return a dictionary. The returned dictionary
       will be used to format the string. For example:

    .. code-block:: python

                a = brainstate.State(1.)
                def loop_fn(x):
                    a.value = x.value + 1.
                    return jnp.sum(x ** 2)

                pbar = ProgressBar(desc=("Running {i} iterations, loss = {loss}",
                                         lambda i_carray_y: {"i": i_carray_y["i"], "loss": i_carray_y["y"]}))

                brainstate.compile.for_loop(loop_fn, xs, pbar=pbar)

    In this example, ``"i"`` denotes the iteration number and ``"loss"`` is computed from the output,
    the ``"carry"`` is the dynamic state in the loop, for example ``a.value`` in this case.


    Args:
        freq: The frequency at which to print the progress bar. If not specified, the progress
            bar will be printed every 5% of the total iterations.
        count: The number of times to print the progress bar. If not specified, the progress
            bar will be printed every 5% of the total iterations.
        desc: A description of the progress bar. If not specified, a default message will be
            displayed.
        kwargs: Additional keyword arguments to pass to the progress bar.
    """
    __module__ = "brainstate.compile"

    def __init__(
        self,
        freq: Optional[int] = None,
        count: Optional[int] = None,
        desc: Optional[Tuple[str, Callable[[Dict], Dict]] | str] = None,
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
            if isinstance(desc, str):
                pass
            else:
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
        data = dict() if isinstance(self.message, str) else self.message[1](dict(i=iter_num, **kwargs))
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
