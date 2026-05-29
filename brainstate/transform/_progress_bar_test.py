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

import unittest

import jax.numpy as jnp

import brainstate
from brainstate.transform import ProgressBar


class TestProgressBarConstruction(unittest.TestCase):
    """Validation performed in ``ProgressBar.__init__``."""

    def test_freq_must_be_positive(self):
        """A non-positive ``freq`` raises ``AssertionError``."""
        with self.assertRaises(AssertionError):
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
        """A ``desc`` tuple whose second element is not callable raises."""
        with self.assertRaises(AssertionError):
            ProgressBar(desc=("step {i}", "not-callable"))


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


if __name__ == "__main__":
    unittest.main()
