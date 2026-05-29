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
import numpy as np

import brainstate


class TestExports(unittest.TestCase):
    def test_symbols_exist(self):
        self.assertTrue(callable(brainstate.transform.pure_callback))
        self.assertTrue(callable(brainstate.transform.io_callback))


class TestPureCallback(unittest.TestCase):
    def test_basic(self):
        def host_fn(x):
            return np.sin(np.asarray(x))

        x = jnp.array([0.0, np.pi / 2])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = brainstate.transform.pure_callback(host_fn, spec, x)
        self.assertTrue(jnp.allclose(out, jnp.sin(x)))

    def test_read_states(self):
        w = brainstate.State(jnp.array([2.0, 3.0]))

        def host_fn(x, w_val):
            return np.asarray(x) * np.asarray(w_val)

        x = jnp.array([5.0, 7.0])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = brainstate.transform.pure_callback(host_fn, spec, x, read_states=w)
        self.assertTrue(jnp.allclose(out, x * w.value))

    def test_under_jit(self):
        def host_fn(x):
            return np.asarray(x) ** 2

        @jax.jit
        def f(x):
            spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
            return brainstate.transform.pure_callback(host_fn, spec, x)

        x = jnp.array([1.0, 2.0, 3.0])
        self.assertTrue(jnp.allclose(f(x), x ** 2))


class TestIoCallback(unittest.TestCase):
    def test_basic_return(self):
        def host_fn(x):
            return np.asarray(x) + 1.0

        x = jnp.array([1.0, 2.0])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = brainstate.transform.io_callback(host_fn, spec, x)
        self.assertTrue(jnp.allclose(out, x + 1.0))

    def test_read_states(self):
        bias = brainstate.State(jnp.array([10.0, 20.0]))

        def host_fn(x, b):
            return np.asarray(x) + np.asarray(b)

        x = jnp.array([1.0, 2.0])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = brainstate.transform.io_callback(host_fn, spec, x, read_states=bias)
        self.assertTrue(jnp.allclose(out, x + bias.value))

    def test_writeback_single_state(self):
        state = brainstate.State(jnp.array([0.0, 0.0]))

        def host_fn(x):
            return np.asarray(x) + 1.0

        x = jnp.array([1.0, 2.0])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = brainstate.transform.io_callback(host_fn, spec, x, write_states=state)
        self.assertTrue(jnp.allclose(out, x + 1.0))
        self.assertTrue(jnp.allclose(state.value, x + 1.0))

    def test_writeback_multiple_states(self):
        s1 = brainstate.State(jnp.array(0.0))
        s2 = brainstate.State(jnp.array(0.0))

        def host_fn(x):
            xv = np.asarray(x)
            return (xv + 1.0, xv + 2.0)

        x = jnp.array(5.0)
        specs = (jax.ShapeDtypeStruct((), x.dtype), jax.ShapeDtypeStruct((), x.dtype))
        out = brainstate.transform.io_callback(host_fn, specs, x, write_states=[s1, s2])
        self.assertTrue(jnp.allclose(s1.value, 6.0))
        self.assertTrue(jnp.allclose(s2.value, 7.0))

    def test_ordered_side_effect_under_jit(self):
        log = []

        def host_fn(x):
            log.append(float(np.asarray(x)))
            return np.asarray(x)

        @jax.jit
        def f(x):
            spec = jax.ShapeDtypeStruct((), x.dtype)
            return brainstate.transform.io_callback(host_fn, spec, x)

        f(jnp.array(3.0))
        self.assertEqual(log, [3.0])


class TestStateNormalization(unittest.TestCase):
    """Validate the ``read_states``/``write_states`` normalization helper."""

    def test_read_states_as_list(self):
        """A list of read_states is normalized and appended in order."""
        a = brainstate.State(jnp.array([2.0, 3.0]))
        b = brainstate.State(jnp.array([10.0, 20.0]))

        def host(x, a_val, b_val):
            return np.asarray(x) + np.asarray(a_val) + np.asarray(b_val)

        x = jnp.array([1.0, 1.0])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = brainstate.transform.pure_callback(host, spec, x, read_states=[a, b])
        self.assertTrue(bool(jnp.allclose(out, jnp.array([13.0, 24.0]))))

    def test_read_states_non_state_raises(self):
        """A non-State entry in ``read_states`` raises ``TypeError``."""
        a = brainstate.State(jnp.array([1.0]))
        x = jnp.array([1.0])
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        with self.assertRaises(TypeError):
            brainstate.transform.pure_callback(
                lambda *xs: np.asarray(xs[0]), spec, x, read_states=[a, object()]
            )

    def test_write_states_non_state_raises(self):
        """A non-State entry in ``write_states`` raises ``TypeError``."""
        st = brainstate.State(jnp.array([0.0]))
        x = jnp.array([1.0])
        spec = (jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct(x.shape, x.dtype))

        def host(v):
            return np.asarray(v), np.asarray(v) + 1.0

        with self.assertRaises(TypeError):
            brainstate.transform.io_callback(
                host, spec, x, write_states=[st, object()]
            )


if __name__ == '__main__':
    unittest.main()
