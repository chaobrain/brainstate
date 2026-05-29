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

import os
# Fake multiple CPU devices when this module is imported before JAX initializes.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

import unittest

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import brainstate


class TestExports(unittest.TestCase):
    def test_symbol_exists(self):
        self.assertTrue(callable(brainstate.transform.shard_map))


class TestShardMap(unittest.TestCase):
    def setUp(self):
        self.n = jax.device_count()
        self.mesh = jax.make_mesh((self.n,), ('x',))

    def test_sharded_data_replicated_state(self):
        w = brainstate.State(jnp.array(3.0))               # replicated scalar
        out_state = brainstate.State(jnp.zeros(self.n * 2))  # sharded write state

        def fun(data):
            out_state.value = data * w.value
            return out_state.value

        f = brainstate.transform.shard_map(
            fun, self.mesh, in_specs=(P('x'),), out_specs=P('x'),
            state_out_specs=P('x'),
        )
        data = jnp.arange(self.n * 2, dtype=jnp.float32)
        out = f(data)
        self.assertTrue(jnp.allclose(out, data * 3.0))
        self.assertTrue(jnp.allclose(out_state.value, data * 3.0))
        self.assertTrue(jnp.allclose(w.value, 3.0))  # replicated read state unchanged

    def test_no_state_function(self):
        def fun(data):
            return data + 1.0

        f = brainstate.transform.shard_map(
            fun, self.mesh, in_specs=(P('x'),), out_specs=P('x'),
        )
        data = jnp.arange(self.n * 2, dtype=jnp.float32)
        self.assertTrue(jnp.allclose(f(data), data + 1.0))

    def test_repeatable_no_tracer_leak(self):
        w = brainstate.State(jnp.array(2.0))

        def fun(data):
            return data * w.value

        f = brainstate.transform.shard_map(
            fun, self.mesh, in_specs=(P('x'),), out_specs=P('x'),
        )
        d1 = jnp.arange(self.n * 2, dtype=jnp.float32)
        out1 = f(d1)
        self.assertTrue(jnp.allclose(out1, d1 * 2.0))
        # Second call must not see a leaked tracer in w.
        d2 = jnp.ones(self.n * 2)
        out2 = f(d2)
        self.assertTrue(jnp.allclose(out2, d2 * 2.0))
        self.assertTrue(jnp.allclose(w.value, 2.0))

    def test_under_jit(self):
        w = brainstate.State(jnp.array(5.0))

        def fun(data):
            return data * w.value

        f = brainstate.transform.shard_map(
            fun, self.mesh, in_specs=(P('x'),), out_specs=P('x'),
        )
        data = jnp.arange(self.n * 2, dtype=jnp.float32)
        out = jax.jit(f)(data)
        self.assertTrue(jnp.allclose(out, data * 5.0))

    def test_in_specs_length_mismatch_raises(self):
        def fun(a, b):
            return a + b

        f = brainstate.transform.shard_map(
            fun, self.mesh, in_specs=(P('x'),), out_specs=P('x'),  # only 1 spec for 2 args
        )
        d = jnp.arange(self.n * 2, dtype=jnp.float32)
        with self.assertRaises(ValueError):
            f(d, d)


if __name__ == '__main__':
    unittest.main()
