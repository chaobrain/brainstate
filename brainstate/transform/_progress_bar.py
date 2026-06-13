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


import copy
import importlib.util
import shutil
import sys
import time
from typing import Optional, Callable, Any, Tuple, Dict

import jax

tqdm_installed = importlib.util.find_spec('tqdm') is not None

__all__ = [
    'ProgressBar',
]

Index = int
Carray = Any
Output = Any


class _FallbackProgressBar:
    """
    A minimal, dependency-free progress bar used when :mod:`tqdm` is unavailable.

    This class reproduces the small slice of the :mod:`tqdm` API that
    :class:`ProgressBarRunner` relies on (construction from an iterable,
    :meth:`set_description`, :meth:`update`, and :meth:`close`), so it can be
    used as a drop-in replacement. It renders a single-line, tqdm-like progress
    display containing a description, percentage, a Unicode bar, the
    iteration count, elapsed time, an ETA, and an iteration rate.

    The renderer is terminal aware. When the output stream is an interactive
    terminal, the line is redrawn in place using a carriage return. When the
    output is redirected (a file, a CI log, or a notebook), each update is
    written on its own line so no carriage-return control characters leak into
    the captured text.

    Parameters
    ----------
    iterable : iterable, optional
        Iterable whose length provides ``total`` when ``total`` is not given.
        Only its length is used; it is not consumed.
    total : int, optional
        Total number of iterations. Falls back to ``len(iterable)`` and finally
        to ``None`` (an unknown-total display) when neither is available.
    desc : str, optional
        Description shown as a prefix (``"desc: "``).
    file : file-like, optional
        Output stream. Defaults to :data:`sys.stderr` (matching :mod:`tqdm`).
    disable : bool, default False
        When ``True`` every method becomes a no-op and nothing is written.
    ncols : int, optional
        Fixed total width in columns. When omitted the terminal width is queried
        via :func:`shutil.get_terminal_size`.
    _time : callable, optional
        Zero-argument clock returning seconds, used for elapsed/rate/ETA. Defaults
        to :func:`time.monotonic`. Exposed mainly to make tests deterministic.
    **ignored
        Any further :mod:`tqdm`-specific keyword arguments (for example
        ``colour``, ``leave``, ``unit``, ``position``). They are accepted and
        ignored so code written for :mod:`tqdm` does not break.

    See Also
    --------
    ProgressBar : The public progress-bar configuration object.

    Notes
    -----
    Only a single bar is supported; nested/positioned bars are not. The
    iteration rate is an exponential moving average (smoothing ``0.3``) of the
    instantaneous rate, matching the feel of :mod:`tqdm`, and falls back to the
    overall average until enough data is available.

    Examples
    --------
    .. code-block:: python

        >>> import io
        >>> from brainstate.transform._progress_bar import _FallbackProgressBar
        >>> buf = io.StringIO()
        >>> bar = _FallbackProgressBar(total=4, desc="Train", file=buf)
        >>> for _ in range(4):
        ...     bar.update(1)
        >>> bar.close()
        >>> "4/4" in buf.getvalue()
        True
    """
    __module__ = "brainstate.transform"

    def __init__(
        self,
        iterable=None,
        *,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        file=None,
        disable: bool = False,
        ncols: Optional[int] = None,
        _time: Optional[Callable[[], float]] = None,
        **ignored,
    ):
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.total = total
        self.n = 0
        self.desc = desc or ''
        self.file = file if file is not None else sys.stderr
        self.disable = bool(disable)
        self.ncols = ncols
        self._time = _time if _time is not None else time.monotonic

        self._start = self._time()
        self._last_time = self._start
        self._last_n = 0
        self._ema_rate: Optional[float] = None
        self._last_line_len = 0
        self._closed = False

        self._is_tty = bool(getattr(self.file, 'isatty', lambda: False)())
        enc = getattr(self.file, 'encoding', None)
        self._ascii = bool(enc) and 'utf' not in enc.lower()

    # -- public tqdm-compatible API --

    def set_description(self, desc=None, refresh: bool = True):
        """Set the description prefix, optionally redrawing immediately."""
        self.desc = desc or ''
        if refresh and not self.disable and not self._closed:
            self._render()

    def update(self, n: int = 1):
        """Advance the counter by ``n`` iterations and redraw."""
        if self.disable or self._closed:
            return
        self.n += n
        self._render()

    def close(self):
        """Render the final state and terminate the line."""
        if self.disable or self._closed:
            return
        self._closed = True
        self._render(final=True)

    # -- internals --

    def _update_rate(self):
        if self.n <= self._last_n:
            return
        now = self._time()
        dt = now - self._last_time
        if dt <= 0:
            return
        inst = (self.n - self._last_n) / dt
        if self._ema_rate is None:
            self._ema_rate = inst
        else:
            self._ema_rate = 0.3 * inst + 0.7 * self._ema_rate
        self._last_time = now
        self._last_n = self.n

    def _current_rate(self) -> Optional[float]:
        if self._ema_rate is not None:
            return self._ema_rate
        elapsed = self._time() - self._start
        if elapsed > 0 and self.n > 0:
            return self.n / elapsed
        return None

    @staticmethod
    def _format_interval(seconds: float) -> str:
        # Guard against NaN / +-inf, then clamp negatives to zero.
        if seconds != seconds or seconds in (float('inf'), float('-inf')):
            return '?'
        total = max(int(seconds), 0)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f'{h:d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

    def _make_bar(self, frac: float, reserved: int) -> str:
        if self.ncols is not None:
            cols = self.ncols
        else:
            cols = shutil.get_terminal_size(fallback=(80, 20)).columns
        width = cols - reserved
        if width <= 0:
            return ''
        charset = ' 123456789#' if self._ascii else ' ▏▎▍▌▋▊▉█'
        full_char = charset[-1]
        n_parts = len(charset) - 1
        filled = frac * width
        full = int(filled)
        if full >= width:
            return full_char * width
        bar = full_char * full
        idx = int((filled - full) * n_parts)
        bar += charset[idx]
        bar += ' ' * (width - full - 1)
        return bar

    def _format_line(self) -> str:
        prefix = f'{self.desc}: ' if self.desc else ''
        elapsed = self._format_interval(self._time() - self._start)
        rate = self._current_rate()
        rate_str = f'{rate:.2f}it/s' if rate else '?it/s'
        if self.total:
            frac = min(max(self.n / self.total, 0.0), 1.0)
            pct = int(frac * 100)
            if rate and rate > 0:
                eta = self._format_interval((self.total - self.n) / rate)
            else:
                eta = '?'
            right = f'{self.n}/{self.total} [{elapsed}<{eta}, {rate_str}]'
            left = f'{prefix}{pct:3d}%|'
            bar = self._make_bar(frac, len(left) + len(right) + 2)
            return f'{left}{bar}| {right}'
        return f'{prefix}{self.n}it [{elapsed}, {rate_str}]'

    def _write(self, text: str):
        try:
            self.file.write(text)
        except UnicodeEncodeError:
            enc = getattr(self.file, 'encoding', None) or 'ascii'
            self.file.write(text.encode(enc, errors='replace').decode(enc))
        flush = getattr(self.file, 'flush', None)
        if flush is not None:
            flush()

    def _render(self, final: bool = False):
        if self.disable:
            return
        self._update_rate()
        line = self._format_line()
        if self._is_tty:
            pad = max(self._last_line_len - len(line), 0)
            self._write('\r' + line + ' ' * pad)
            self._last_line_len = len(line)
            if final:
                self._write('\n')
        else:
            self._write(line + '\n')


class ProgressBar(object):
    """
    A progress bar for tracking the progress of a jitted for-loop computation.

    It can be used in :py:func:`for_loop`, :py:func:`checkpointed_for_loop`, :py:func:`scan`,
    and :py:func:`checkpointed_scan` functions. Or any other jitted function that uses
    a for-loop.

    The message displayed in the progress bar can be customized by the following two methods:

    1. By passing a string to the `desc` argument.
    2. By passing a tuple with a string and a callable function to the `desc` argument. The callable
       function should take a dictionary as input and return a dictionary. The returned dictionary
       will be used to format the string.

    In the second case, ``"i"`` denotes the iteration number and other keys can be computed from the
    loop outputs and carry values.

    Parameters
    ----------
    freq : int, optional
        The frequency at which to print the progress bar. If not specified, the progress
        bar will be printed every 5% of the total iterations.
    count : int, optional
        The number of times to print the progress bar. If not specified, the progress
        bar will be printed every 5% of the total iterations. Cannot be used together with `freq`.
    desc : str or tuple, optional
        A description of the progress bar. If not specified, a default message will be
        displayed. Can be either a string or a tuple of (format_string, format_function).
    **kwargs
        Additional keyword arguments to pass to the progress bar.

    Notes
    -----
    ``tqdm`` is an optional dependency. When it is installed, the bar is rendered
    with :mod:`tqdm`. When it is not installed, a built-in pure-Python fallback
    (:class:`_FallbackProgressBar`) renders an equivalent terminal progress bar,
    so :class:`ProgressBar` works in either environment with no code changes.

    Examples
    --------
    Basic usage with default description:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> def loop_fn(x):
        ...     return x ** 2
        >>>
        >>> xs = jnp.arange(100)
        >>> pbar = brainstate.transform.ProgressBar()
        >>> results = brainstate.transform.for_loop(loop_fn, xs, pbar=pbar)

    With custom description string:

    .. code-block:: python

        >>> pbar = brainstate.transform.ProgressBar(desc="Running 1000 iterations")
        >>> results = brainstate.transform.for_loop(loop_fn, xs, pbar=pbar)

    With frequency control:

    .. code-block:: python

        >>> # Update every 10 iterations
        >>> pbar = brainstate.transform.ProgressBar(freq=10)
        >>> results = brainstate.transform.for_loop(loop_fn, xs, pbar=pbar)
        >>>
        >>> # Update exactly 20 times during execution
        >>> pbar = brainstate.transform.ProgressBar(count=20)
        >>> results = brainstate.transform.for_loop(loop_fn, xs, pbar=pbar)

    With dynamic description based on loop variables:

    .. code-block:: python

        >>> state = brainstate.State(1.0)
        >>>
        >>> def loop_fn(x):
        ...     state.value += x
        ...     loss = jnp.sum(x ** 2)
        ...     return loss
        >>>
        >>> def format_desc(data):
        ...     return {"i": data["i"], "loss": data["y"], "state": data["carry"]}
        >>>
        >>> pbar = brainstate.transform.ProgressBar(
        ...     desc=("Iteration {i}, loss = {loss:.4f}, state = {state:.2f}", format_desc)
        ... )
        >>> results = brainstate.transform.for_loop(loop_fn, xs, pbar=pbar)

    With scan function:

    .. code-block:: python

        >>> def scan_fn(carry, x):
        ...     new_carry = carry + x
        ...     return new_carry, new_carry ** 2
        >>>
        >>> init_carry = 0.0
        >>> pbar = brainstate.transform.ProgressBar(freq=5)
        >>> final_carry, ys = brainstate.transform.scan(scan_fn, init_carry, xs, pbar=pbar)
    """
    __module__ = "brainstate.transform"

    def __init__(
        self,
        freq: Optional[int] = None,
        count: Optional[int] = None,
        desc: Optional[Tuple[str, Callable[[Dict], Dict]] | str] = None,
        **kwargs
    ):
        # print rate
        self.print_freq = freq
        if isinstance(freq, int) and freq <= 0:
            raise ValueError(f"Print rate (freq) should be > 0, but got {freq}.")

        # print count
        self.print_count = count
        if self.print_freq is not None and self.print_count is not None:
            raise ValueError("Cannot specify both count and freq.")

        # other parameters
        for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
            kwargs.pop(kwarg, None)
        self.kwargs = kwargs

        # description -- ``desc`` is user-supplied, so validate with real
        # exceptions: ``assert`` is stripped under ``python -O``, which would let
        # a malformed ``desc`` reach the formatting callbacks and fail cryptically.
        if desc is not None:
            if isinstance(desc, str):
                pass
            else:
                if not isinstance(desc, (tuple, list)):
                    raise TypeError(
                        f'desc must be a string or a (format_string, callable) '
                        f'tuple/list, but got {type(desc).__name__}.'
                    )
                if len(desc) != 2:
                    raise ValueError(
                        f'desc tuple/list must have exactly two elements '
                        f'(format_string, callable), but got {len(desc)}.'
                    )
                if not isinstance(desc[0], str):
                    raise TypeError(
                        f'The first element of a desc tuple must be a format '
                        f'string, but got {type(desc[0]).__name__}.'
                    )
                if not callable(desc[1]):
                    raise TypeError(
                        f'The second element of a desc tuple must be callable, '
                        f'but got {type(desc[1]).__name__}.'
                    )
        self.desc = desc

    def init(self, n: int):
        kwargs = copy.copy(self.kwargs)
        freq = self.print_freq
        count = self.print_count
        if count is not None:
            freq = n // count
            if freq == 0:
                raise ValueError(f"Count {count} is too large for n {n}.")
            # The leftover added by ``_close_tqdm`` must be ``n % freq`` (the
            # iterations not covered by the floor(n/freq) regular ``update(freq)``
            # ticks), NOT ``n % count``. Using ``n % count`` overshoots the bar
            # past 100% whenever ``count`` does not evenly divide ``n``. Match the
            # other two branches, which correctly use ``n % freq``.
            remainder = n % freq
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
    __module__ = "brainstate.transform"

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
        # Read the module flag at call time so it can be monkeypatched in tests.
        if tqdm_installed:
            from tqdm.auto import tqdm as bar_cls
        else:
            bar_cls = _FallbackProgressBar
        self.tqdm_bars[0] = bar_cls(range(self.n), **self.kwargs)
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
        # The description callback is user-supplied; validate its return with a
        # real exception (``assert`` is stripped under ``python -O``), which would
        # otherwise let a non-dict reach ``str.format(**data)`` and fail cryptically.
        if not isinstance(data, dict):
            raise TypeError(
                f'The desc format function must return a dict, but got '
                f'{type(data).__name__}.'
            )

        _ = jax.lax.cond(
            iter_num == 0,
            lambda x: jax.debug.callback(self._define_tqdm, x, ordered=True),
            lambda x: None,
            data
        )
        # Guard the update on ``iter_num < self.n``. ``checkpointed_scan`` runs
        # through ``_bounded_while_loop``, whose skip path advances the counter
        # PAST ``self.n - 1`` (by whole sub-blocks) after the loop is done and
        # still invokes this runner. Without the guard those post-completion
        # calls can satisfy ``iter_num % print_freq == print_freq - 1`` and fire
        # ``update()`` after ``close()`` already ran, pushing the bar past 100%.
        _ = jax.lax.cond(
            (iter_num < self.n) & (iter_num % self.print_freq == (self.print_freq - 1)),
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
