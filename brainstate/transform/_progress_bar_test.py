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

"""Tests for :class:`brainstate.transform.ProgressBar` and its loop integration.

The bar has two layers: construction-time validation (``freq``/``count``/``desc``)
and the ``init`` -> runner machinery that fires ``tqdm`` callbacks during a
:func:`for_loop`. Both layers are exercised here. Loops are tiny so the tests
stay fast; the assertions check the numerical loop result, which also forces the
``tqdm`` callbacks (define/update/close) to actually execute.
"""

import contextlib
import io
import unittest
from unittest import mock

import jax.numpy as jnp

import brainstate
from brainstate.transform import ProgressBar
from brainstate.transform import _progress_bar as _pbmod
from brainstate.transform._progress_bar import _FallbackProgressBar, ProgressBarRunner


class _TTYStringIO(io.StringIO):
    """A ``StringIO`` that reports itself as an interactive terminal."""

    def isatty(self):
        return True


class _AsciiFile:
    """A minimal non-UTF output stream used to drive the ASCII bar charset."""

    encoding = "ascii"

    def __init__(self):
        self.parts = []

    def write(self, s):
        self.parts.append(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def getvalue(self):
        return "".join(self.parts)


def _make_clock(step=0.1, start=0.0):
    """Return a deterministic monotonic-like clock that advances by ``step``.

    The first call returns ``start``; each subsequent call advances by ``step``.
    """
    state = {"t": start, "first": True}

    def clock():
        if state["first"]:
            state["first"] = False
            return state["t"]
        state["t"] += step
        return state["t"]

    return clock


class TestProgressBarConstruction(unittest.TestCase):
    """Validation performed in ``ProgressBar.__init__``."""

    def test_freq_must_be_positive(self):
        """A non-positive ``freq`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            ProgressBar(freq=0)

    def test_cannot_specify_both_freq_and_count(self):
        """Specifying both ``freq`` and ``count`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            ProgressBar(freq=2, count=5)

    def test_desc_string_is_stored(self):
        """A string ``desc`` is stored verbatim."""
        pbar = ProgressBar(desc="Running")
        self.assertEqual(pbar.desc, "Running")

    def test_desc_tuple_is_stored(self):
        """A ``(format_string, callable)`` ``desc`` is accepted."""
        fn = lambda d: {"i": d["i"]}
        pbar = ProgressBar(desc=("step {i}", fn))
        self.assertEqual(pbar.desc[0], "step {i}")

    def test_desc_tuple_requires_callable(self):
        """A ``desc`` tuple whose second element is not callable raises.

        Uses a real ``TypeError`` (not ``AssertionError``) so the validation
        survives ``python -O`` (audit Tier C).
        """
        with self.assertRaises(TypeError):
            ProgressBar(desc=("step {i}", "not-callable"))

    def test_desc_tuple_first_must_be_string(self):
        """A ``desc`` tuple whose first element is not a string raises TypeError."""
        with self.assertRaises(TypeError):
            ProgressBar(desc=(123, lambda d: d))

    def test_desc_must_be_str_or_sequence(self):
        """A ``desc`` that is neither a string nor a tuple/list raises TypeError."""
        with self.assertRaises(TypeError):
            ProgressBar(desc=123)

    def test_desc_tuple_wrong_length_raises(self):
        """A ``desc`` tuple without exactly two elements raises ValueError."""
        with self.assertRaises(ValueError):
            ProgressBar(desc=("only-one",))


class TestProgressBarInitFrequency(unittest.TestCase):
    """``ProgressBar.init`` frequency/count resolution via ``for_loop``."""

    def _run(self, n, pbar):
        ys = brainstate.transform.for_loop(lambda x: x * 2, jnp.arange(float(n)), pbar=pbar)
        ys.block_until_ready()
        return ys

    def test_count_resolves_frequency(self):
        """``count`` divides the length into update steps."""
        ys = self._run(4, ProgressBar(count=2))
        self.assertEqual(ys.shape, (4,))
        self.assertTrue(bool(jnp.allclose(ys, jnp.arange(4.0) * 2)))

    def test_count_too_large_raises(self):
        """A ``count`` larger than the length raises ``ValueError``."""
        with self.assertRaises(ValueError):
            self._run(4, ProgressBar(count=100))

    def test_default_small_n_uses_freq_one(self):
        """With no freq/count and ``n <= 20`` the bar updates every step."""
        ys = self._run(10, ProgressBar())
        self.assertEqual(ys.shape, (10,))

    def test_default_large_n_uses_five_percent(self):
        """With no freq/count and ``n > 20`` the bar updates ~ every 5%."""
        ys = self._run(40, ProgressBar())
        self.assertEqual(ys.shape, (40,))

    def test_explicit_freq_with_remainder(self):
        """An explicit ``freq`` not dividing ``n`` leaves a closing remainder."""
        ys = self._run(10, ProgressBar(freq=3))
        self.assertEqual(ys.shape, (10,))

    def test_freq_greater_than_n_raises(self):
        """A ``freq`` exceeding the length raises ``ValueError``."""
        with self.assertRaises(ValueError):
            self._run(4, ProgressBar(freq=100))

    def test_int_pbar_shortcut(self):
        """An int ``pbar`` is treated as ``ProgressBar(freq=int)``."""
        ys = brainstate.transform.for_loop(lambda x: x + 1, jnp.arange(6.0), pbar=2)
        self.assertTrue(bool(jnp.allclose(ys, jnp.arange(6.0) + 1)))


class TestProgressBarRunnerMessages(unittest.TestCase):
    """The runner's define/update/close callbacks for both message kinds."""

    def test_string_desc_runs_to_completion(self):
        """A string ``desc`` drives the str branch of every callback."""
        ys = brainstate.transform.for_loop(
            lambda x: x * 2, jnp.arange(5.0), pbar=ProgressBar(freq=2, desc="run")
        )
        ys.block_until_ready()
        self.assertEqual(ys.shape, (5,))

    def test_dynamic_desc_runs_to_completion(self):
        """A ``(fmt, fn)`` ``desc`` drives the formatted branch of each callback."""

        def fmt(data):
            return {"i": data["i"]}

        pbar = ProgressBar(freq=2, desc=("iter {i}", fmt))
        ys = brainstate.transform.for_loop(lambda x: x * 2, jnp.arange(5.0), pbar=pbar)
        ys.block_until_ready()
        self.assertEqual(ys.shape, (5,))


class TestFallbackProgressBar(unittest.TestCase):
    """Unit tests for the dependency-free fallback bar (``_FallbackProgressBar``)."""

    def test_basic_progression_to_100(self):
        """Driving the bar to ``total`` shows a full count and 100%."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=4, file=buf, ncols=80)
        for _ in range(4):
            bar.update(1)
        bar.close()
        out = buf.getvalue()
        self.assertIn("4/4", out)
        self.assertIn("100%", out)

    def test_midpoint_percentage(self):
        """Half-way progress renders ``50%``."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=10, file=buf, ncols=80)
        bar.update(5)
        self.assertIn(" 50%", buf.getvalue())

    def test_tty_uses_carriage_return_and_single_trailing_newline(self):
        """On a TTY: ``\\r`` redraws, no newline until ``close`` adds exactly one."""
        buf = _TTYStringIO()
        bar = _FallbackProgressBar(total=4, file=buf, ncols=60)
        bar.update(1)
        mid = buf.getvalue()
        self.assertIn("\r", mid)
        self.assertNotIn("\n", mid)
        bar.update(3)
        bar.close()
        out = buf.getvalue()
        self.assertTrue(out.endswith("\n"))
        self.assertEqual(out.count("\n"), 1)

    def test_non_tty_writes_newline_per_update_no_cr(self):
        """Off a TTY: one line per update, never a carriage return."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=3, file=buf, ncols=60)
        bar.update(1)
        bar.update(1)
        out = buf.getvalue()
        self.assertNotIn("\r", out)
        self.assertEqual(out.count("\n"), 2)

    def test_rate_eta_elapsed_present_with_advancing_clock(self):
        """An advancing clock yields a concrete rate and an elapsed<ETA field."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=10, file=buf, ncols=80, _time=_make_clock(step=0.1))
        for _ in range(5):
            bar.update(1)
        last = buf.getvalue().splitlines()[-1]
        self.assertIn("it/s", last)
        self.assertNotIn("?it/s", last)
        self.assertIn("<", last)

    def test_rate_unknown_when_no_time_elapses(self):
        """With a frozen clock the rate is unknown (``?it/s``)."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=10, file=buf, ncols=80, _time=lambda: 5.0)
        bar.update(1)
        self.assertIn("?it/s", buf.getvalue())

    def test_unknown_total_branch(self):
        """No total -> count/elapsed form with no percentage or bar."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(file=buf)
        bar.update(3)
        bar.close()
        out = buf.getvalue()
        self.assertIn("3it", out)
        self.assertNotIn("%", out)

    def test_total_inferred_from_iterable_length(self):
        """``total`` defaults to ``len(iterable)`` when not given."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(range(7), file=buf, ncols=80)
        bar.update(7)
        self.assertIn("7/7", buf.getvalue())

    def test_disable_produces_no_output(self):
        """``disable=True`` makes every method a no-op."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=5, file=buf, disable=True)
        bar.update(5)
        bar.close()
        self.assertEqual(buf.getvalue(), "")

    def test_description_prefix_and_set_description(self):
        """The description is shown as a prefix and can be changed."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=4, desc="Train", file=buf, ncols=80)
        bar.update(1)
        self.assertIn("Train:", buf.getvalue())
        bar.set_description("Eval")
        self.assertIn("Eval:", buf.getvalue())

    def test_unknown_tqdm_kwargs_ignored(self):
        """tqdm-only kwargs are accepted and ignored without error."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(
            range(3), file=buf, ncols=80,
            colour="green", leave=False, unit="it", position=0,
        )
        bar.update(3)
        bar.close()
        self.assertIn("3/3", buf.getvalue())

    def test_narrow_terminal_drops_bar_but_keeps_counts(self):
        """A tiny ``ncols`` drops the bar yet still shows the counts."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=10, file=buf, ncols=5)
        bar.update(5)
        out = buf.getvalue()
        self.assertIn("5/10", out)
        self.assertNotIn("█", out)

    def test_ascii_charset_for_non_utf_stream(self):
        """A non-UTF stream uses the ASCII bar charset."""
        af = _AsciiFile()
        bar = _FallbackProgressBar(total=4, file=af, ncols=50)
        bar.update(2)
        bar.close()
        out = af.getvalue()
        self.assertIn("#", out)
        self.assertNotIn("█", out)

    def test_close_is_idempotent(self):
        """Calling ``close`` twice does not write a second time."""
        buf = _TTYStringIO()
        bar = _FallbackProgressBar(total=2, file=buf, ncols=60)
        bar.update(2)
        bar.close()
        first = buf.getvalue()
        bar.close()
        self.assertEqual(buf.getvalue(), first)

    def test_update_after_close_is_noop(self):
        """Updating after ``close`` writes nothing further."""
        buf = io.StringIO()
        bar = _FallbackProgressBar(total=4, file=buf, ncols=60)
        bar.update(4)
        bar.close()
        snapshot = buf.getvalue()
        bar.update(1)
        self.assertEqual(buf.getvalue(), snapshot)

    def test_format_interval(self):
        """``_format_interval`` formats seconds, hours, and guards bad input."""
        self.assertEqual(_FallbackProgressBar._format_interval(5), "00:05")
        self.assertEqual(_FallbackProgressBar._format_interval(3725), "1:02:05")
        self.assertEqual(_FallbackProgressBar._format_interval(-3), "00:00")
        self.assertEqual(_FallbackProgressBar._format_interval(float("inf")), "?")
        self.assertEqual(_FallbackProgressBar._format_interval(float("nan")), "?")

    def test_unicode_encode_error_falls_back(self):
        """A unicode description on an ASCII stream is replaced, not raised."""

        class _StrictAsciiFile(io.StringIO):
            encoding = "ascii"

            def isatty(self):
                return False

            def write(self, s):
                s.encode("ascii")  # raises UnicodeEncodeError on non-ascii
                return super().write(s)

        buf = _StrictAsciiFile()
        bar = _FallbackProgressBar(total=4, desc="Δelta", file=buf, ncols=60)
        bar.update(2)  # line carries the unicode desc -> exercises the except path
        bar.close()
        self.assertIn("2/4", buf.getvalue())


class TestProgressBarFallbackIntegration(unittest.TestCase):
    """Force the fallback path (``tqdm`` reported absent) through real loops."""

    def test_construction_does_not_raise_without_tqdm(self):
        """``ProgressBar`` constructs even when ``tqdm`` is unavailable."""
        with mock.patch.object(_pbmod, "tqdm_installed", False):
            ProgressBar(freq=2)

    def test_for_loop_uses_fallback(self):
        """``for_loop`` runs correctly and emits to the fallback bar."""
        with mock.patch.object(_pbmod, "tqdm_installed", False):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                ys = brainstate.transform.for_loop(
                    lambda x: x * 2, jnp.arange(10.0), pbar=ProgressBar(freq=2)
                )
                ys.block_until_ready()
            self.assertTrue(bool(jnp.allclose(ys, jnp.arange(10.0) * 2)))
            self.assertNotEqual(err.getvalue(), "")

    def test_scan_with_dynamic_desc_uses_fallback(self):
        """``scan`` with a ``(fmt, fn)`` desc renders through the fallback."""
        with mock.patch.object(_pbmod, "tqdm_installed", False):

            def step(c, x):
                return c + x, c + x

            def fmt(d):
                return {"i": d["i"]}

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                _, ys = brainstate.transform.scan(
                    step, 0.0, jnp.arange(6.0),
                    pbar=ProgressBar(freq=2, desc=("iter {i}", fmt)),
                )
                ys.block_until_ready()
            out = err.getvalue()
            self.assertNotEqual(out, "")
            self.assertIn("iter", out)


class TestProgressBarFreqValidation(unittest.TestCase):
    """Invalid ``freq`` must raise ValueError (not AssertionError, which
    disappears under ``python -O``) (audit Tier C)."""

    def test_negative_freq_raises_value_error(self):
        with self.assertRaises(ValueError):
            ProgressBar(freq=-3)

    def test_positive_freq_accepted(self):
        pbar = ProgressBar(freq=2)
        self.assertEqual(pbar.print_freq, 2)


class _CountingBar:
    """A minimal fake bar that just accumulates ``update`` deltas, used to
    measure the final bar position the runner drives it to."""

    def __init__(self, *args, **kwargs):
        self.n = 0

    def update(self, delta):
        self.n += delta

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass


class TestProgressBarCountNoOvershoot(unittest.TestCase):
    """``ProgressBar(count=K)`` over a loop of length ``n`` must finish at
    exactly ``n`` (100%) and never overshoot, even when ``K`` does not divide
    ``n`` (audit L24: the close remainder was ``n % count`` instead of
    ``n % freq``)."""

    def _final_position(self, n, count):
        """Replicate the runner's __call__ dispatch with a counting bar and
        return the final accumulated position."""
        runner = ProgressBar(count=count).init(n)
        bar = _CountingBar()
        runner.tqdm_bars[0] = bar
        freq = runner.print_freq
        for iter_num in range(n):
            if iter_num % freq == (freq - 1):
                runner._update_tqdm({})
            if iter_num == n - 1:
                runner._close_tqdm({})
        return bar.n

    def test_count_not_dividing_n_reaches_exactly_n(self):
        # Audit example: count=20, n=50 used to push the bar to 60/50.
        self.assertEqual(self._final_position(50, 20), 50)
        # Audit example: count=99, n=100 used to give 101/100.
        self.assertEqual(self._final_position(100, 99), 100)

    def test_sweep_no_overshoot_no_undershoot(self):
        # Exhaustive-ish sweep: for every valid (n, count) the bar must end at
        # exactly n -- never over, never under.
        for n in range(2, 60):
            for count in range(1, n + 1):
                pos = self._final_position(n, count)
                self.assertEqual(
                    pos, n,
                    msg=f"count={count}, n={n}: bar ended at {pos}, expected {n}",
                )

    def test_remainder_consistent_with_freq(self):
        # The init-time remainder must be n % freq (matching the other branches).
        runner = ProgressBar(count=20).init(50)
        self.assertEqual(runner.print_freq, 2)
        self.assertEqual(runner.remainder, 50 % runner.print_freq)


class TestCheckpointedScanNoPostCloseUpdate(unittest.TestCase):
    """``checkpointed_scan`` drives its bar through ``_bounded_while_loop``,
    whose skip path advances the counter PAST ``n - 1`` after the loop is done.
    ``ProgressBarRunner.__call__`` must guard the update on ``iter_num < n`` so
    those post-completion calls never fire ``update()`` after ``close()`` and
    push the bar past 100% (audit item 18)."""

    def _run_and_collect(self, n, base, freq):
        created = []

        class _RecordingBar:
            def __init__(self, *args, **kwargs):
                self.n = 0
                self.closed = False
                self.updates_after_close = 0
                created.append(self)

            def update(self, delta):
                if self.closed:
                    self.updates_after_close += 1
                self.n += delta

            def set_description(self, *args, **kwargs):
                pass

            def close(self):
                self.closed = True

        with mock.patch.object(_pbmod, "tqdm_installed", False):
            with mock.patch.object(_pbmod, "_FallbackProgressBar", _RecordingBar):
                def step(c, x):
                    return c + x, c + x

                _, ys = brainstate.transform.checkpointed_scan(
                    step, 0.0, jnp.arange(float(n)), base=base,
                    pbar=ProgressBar(freq=freq),
                )
                ys.block_until_ready()
        return created

    def test_no_post_close_update_and_exact_position(self):
        # base=2 forces several skip-call counter bumps past n-1.
        n = 5
        created = self._run_and_collect(n, base=2, freq=1)
        self.assertTrue(created, "expected the runner to create a progress bar")
        bar = created[0]
        # No update() may land after close()...
        self.assertEqual(bar.updates_after_close, 0)
        # ...and the bar must end at exactly n (never past 100%).
        self.assertEqual(bar.n, n)

    def test_no_overshoot_various_lengths(self):
        for n in (3, 4, 7, 8, 16):
            created = self._run_and_collect(n, base=2, freq=1)
            self.assertTrue(created)
            bar = created[0]
            self.assertEqual(bar.updates_after_close, 0, msg=f"n={n}")
            self.assertEqual(bar.n, n, msg=f"n={n}")


class TestProgressBarRunnerDescReturnValidation(unittest.TestCase):
    """``ProgressBarRunner.__call__`` validates the user desc callback's return.

    A ``(fmt, fn)`` desc whose ``fn`` returns a non-dict would otherwise reach
    ``str.format(**data)`` and fail cryptically. The runner raises a clear
    ``TypeError`` (a real exception, not an ``assert`` stripped under ``python
    -O``) before dispatching the ``jax.lax.cond``."""

    def test_non_dict_desc_return_raises_type_error(self):
        """A desc callback returning a non-dict raises TypeError naming the type."""
        def bad_fmt(data):
            return "not-a-dict"  # callable (passes construction) but wrong return

        runner = ProgressBar(freq=2, desc=("iter {i}", bad_fmt)).init(10)
        with self.assertRaises(TypeError) as ctx:
            runner(0)
        msg = str(ctx.exception)
        self.assertIn("must return a dict", msg)
        # The error reports the offending return type.
        self.assertIn("str", msg)

    def test_dict_desc_return_does_not_raise_type_error(self):
        """A desc callback returning a dict passes the validation guard.

        ``__call__`` proceeds into ``jax.lax.cond``; we only assert that the
        non-dict guard does not trip for a well-behaved callback."""
        def good_fmt(data):
            return {"i": data["i"]}

        runner = ProgressBar(freq=2, desc=("iter {i}", good_fmt)).init(10)
        # Must not raise TypeError from the desc-return guard.
        runner(0)


if __name__ == "__main__":
    unittest.main()
