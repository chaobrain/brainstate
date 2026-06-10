# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest

import jax
import jax.numpy as jnp
from jax.experimental import checkify as _cfy

import brainstate


class TestExports(unittest.TestCase):
    def test_symbols_exist(self):
        self.assertTrue(callable(brainstate.transform.checkify))
        self.assertTrue(callable(brainstate.transform.check))
        self.assertTrue(callable(brainstate.transform.check_error))

    def test_error_sets_are_frozensets(self):
        self.assertIsInstance(brainstate.transform.user_checks, frozenset)
        self.assertIsInstance(brainstate.transform.all_checks, frozenset)
        self.assertIsInstance(brainstate.transform.float_checks, frozenset)


class TestCheckify(unittest.TestCase):
    def test_passing_check_returns_value(self):
        def safe(x):
            brainstate.transform.check(x > 0, 'x must be positive')
            return x * 2.0

        checked = brainstate.transform.checkify(safe)
        err, out = checked(jnp.array(3.0))
        self.assertIsNone(err.get())   # no error
        err.throw()                    # must not raise
        self.assertTrue(jnp.allclose(out, 6.0))

    def test_failing_check_records_error(self):
        def safe(x):
            brainstate.transform.check(x > 0, 'x must be positive, got {v}', v=x)
            return x * 2.0

        checked = brainstate.transform.checkify(safe)
        err, out = checked(jnp.array(-1.0))
        self.assertIsNotNone(err.get())
        self.assertIn('positive', err.get())
        with self.assertRaises(_cfy.JaxRuntimeError):
            err.throw()

    def test_state_writeback_and_readonly_unchanged(self):
        w = brainstate.State(jnp.array(2.0))          # read-only
        acc = brainstate.State(jnp.array(0.0))         # written

        def fun(x):
            brainstate.transform.check(x > 0, 'x must be positive')
            acc.value = acc.value + x * w.value
            return x * w.value

        checked = brainstate.transform.checkify(fun)
        err, out = checked(jnp.array(3.0))
        err.throw()
        self.assertTrue(jnp.allclose(out, 6.0))
        self.assertTrue(jnp.allclose(acc.value, 6.0))   # write applied
        self.assertTrue(jnp.allclose(w.value, 2.0))     # read-only unchanged

    def test_repeatable_no_tracer_leak(self):
        acc = brainstate.State(jnp.array(0.0))

        def fun(x):
            acc.value = acc.value + x
            return x

        checked = brainstate.transform.checkify(fun)
        err1, _ = checked(jnp.array(1.0))
        err1.throw()
        self.assertTrue(jnp.allclose(acc.value, 1.0))
        err2, _ = checked(jnp.array(2.0))   # second call must not see a leaked tracer
        err2.throw()
        self.assertTrue(jnp.allclose(acc.value, 3.0))

    def test_under_jit(self):
        def safe(x):
            brainstate.transform.check(x > 0, 'x must be positive')
            return x * 2.0

        checked = brainstate.transform.checkify(safe)
        err, out = jax.jit(checked)(jnp.array(4.0))
        err.throw()
        self.assertTrue(jnp.allclose(out, 8.0))

    def test_float_checks_div_by_zero(self):
        def fun(x):
            return x / 0.0

        checked = brainstate.transform.checkify(fun, errors=brainstate.transform.float_checks)
        err, _ = checked(jnp.array(1.0))
        self.assertIsNotNone(err.get())
        self.assertIn('division', err.get().lower())

    def test_empty_state_function(self):
        def fun(x):
            brainstate.transform.check(x >= 0, 'x must be non-negative')
            return x + 1.0

        checked = brainstate.transform.checkify(fun)
        err, out = checked(jnp.array(5.0))
        err.throw()
        self.assertTrue(jnp.allclose(out, 6.0))


if __name__ == '__main__':
    unittest.main()


class TestFailedCheckifyRestoresStates(unittest.TestCase):
    """A failure inside the checkify pass must not leave tracers in states
    (audit M5: restoration only ran on success)."""

    def test_failure_inside_checkify_restores_states(self):
        s = brainstate.State(jnp.asarray(0.0))
        calls = {'n': 0}

        def f(x):
            s.value = s.value + x
            calls['n'] += 1
            if calls['n'] >= 2:  # first call: discovery trace; second: checkify pass
                raise RuntimeError('boom')
            return x * 2.0

        checked = brainstate.transform.checkify(f)
        with self.assertRaises(RuntimeError):
            checked(jnp.asarray(3.0))
        self.assertFalse(isinstance(s.value, jax.core.Tracer))
        self.assertEqual(float(s.value), 0.0)
