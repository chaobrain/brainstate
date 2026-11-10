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

"""``checkpointed_scan`` / ``checkpointed_for_loop`` on sharded values (chaobrain/brainstate#255).

The collected outputs keep the type of the body's outputs: their sharding under an
explicit mesh (forward and reverse mode), and their varying manual axes inside
``shard_map``. Each case is compared with ``jax.lax.scan`` on the same body.
"""

import os

os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

import brainstate

_N_DEV = len(jax.devices())
_EXPLICIT = getattr(getattr(jax.sharding, 'AxisType', None), 'Explicit', None)
_HAS_EXPLICIT_MESH = _EXPLICIT is not None and 'axis_types' in inspect.signature(jax.make_mesh).parameters


def _mesh(shape, names):
    return jax.make_mesh(shape, names, axis_types=(_EXPLICIT,) * len(shape))


@unittest.skipIf(_N_DEV < 8 or not _HAS_EXPLICIT_MESH, 'needs 8 devices and explicit mesh axes')
class TestCheckpointedScanSharded(unittest.TestCase):

    def setUp(self):
        self.mesh = _mesh((2, 4), ('trial', 'model'))
        self.x = jax.device_put(jnp.arange(8.), NamedSharding(self.mesh, P('model')))
        self.w = jax.device_put(jnp.linspace(0.3, 0.7, 8), NamedSharding(self.mesh, P('model')))

        def local(w, c):
            c = jnp.tanh(c * w + 1.0)
            return c, (c * 2.0, jnp.sum(c)[None])  # a split output and a per-shard one

        self.step = jax.shard_map(local, mesh=self.mesh, in_specs=(P('model'), P('model')),
                                  out_specs=(P('model'), (P('model'), P('model'))))

    def _loss(self, scan):
        def loss(w, x):
            c, ys = scan(lambda c, i: self.step(w, c), x, jnp.arange(5))
            return jnp.sum(c) + jnp.sum(ys[0] * jnp.arange(5.)[:, None]) + jnp.sum(ys[1])

        return loss

    def _check(self, value_and_grad):
        ck = lambda f, c, xs: brainstate.transform.checkpointed_scan(f, c, xs, base=2)
        v, g = value_and_grad(self._loss(ck))
        v0, g0 = value_and_grad(self._loss(jax.lax.scan))
        np.testing.assert_allclose(v, v0, rtol=1e-6)
        np.testing.assert_allclose(g, g0, rtol=1e-5, atol=1e-6)
        self.assertEqual(jax.typeof(g).sharding.spec, P('model'))

    def test_shard_map_body_reverse_mode(self):
        self._check(lambda f: jax.jit(jax.value_and_grad(f))(self.w, self.x))

    def test_shard_map_body_reverse_mode_under_set_mesh(self):
        with jax.set_mesh(self.mesh):
            self._check(lambda f: jax.jit(jax.value_and_grad(f))(self.w, self.x))

    def test_shard_map_body_brainstate_jit(self):
        self._check(lambda f: brainstate.transform.jit(jax.value_and_grad(f))(self.w, self.x))

    def test_outputs_keep_their_sharding(self):
        def f(x):
            return brainstate.transform.checkpointed_scan(lambda c, i: self.step(self.w, c), x,
                                                          jnp.arange(5), base=2)

        c, (ys, sums) = jax.jit(f)(self.x)
        self.assertEqual(ys.shape, (5, 8))
        self.assertEqual(jax.typeof(ys).sharding.spec, P(None, 'model'))
        self.assertEqual(jax.typeof(sums).sharding.spec, P(None, 'model'))
        c0, (ys0, sums0) = jax.jit(lambda x: jax.lax.scan(lambda c, i: self.step(self.w, c), x,
                                                          jnp.arange(5)))(self.x)
        np.testing.assert_array_equal(ys, ys0)
        np.testing.assert_array_equal(sums, sums0)

    def test_checkpointed_for_loop(self):
        def loss(w, x):
            st = brainstate.State(x)

            def body(i):
                st.value, ys = self.step(w, st.value)
                return ys[0]

            ys = brainstate.transform.checkpointed_for_loop(body, jnp.arange(5), base=2)
            return jnp.sum(ys * jnp.arange(5.)[:, None])

        def ref(w, x):
            return jnp.sum(jax.lax.scan(lambda c, i: (lambda r: (r[0], r[1][0]))(self.step(w, c)), x,
                                        jnp.arange(5))[1] * jnp.arange(5.)[:, None])

        g = jax.jit(jax.grad(loss))(self.w, self.x)
        g0 = jax.jit(jax.grad(ref))(self.w, self.x)
        np.testing.assert_allclose(g, g0, rtol=1e-5, atol=1e-6)


@unittest.skipIf(_N_DEV < 4 or not _HAS_EXPLICIT_MESH, 'needs 4 devices and explicit mesh axes')
class TestCheckpointedScanInShardMap(unittest.TestCase):

    def test_varying_outputs(self):
        mesh = _mesh((4,), ('model',))
        x = jax.device_put(jnp.arange(8.), NamedSharding(mesh, P('model')))
        w = jnp.float32(0.5)

        def make(scan):
            def local(w, c):
                c, ys = scan(lambda c, i: (jnp.tanh(c * w + 1.0), c * 2.0), c, jnp.arange(5))
                return jax.lax.psum(jnp.sum(c) + jnp.sum(ys), 'model')

            return jax.shard_map(local, mesh=mesh, in_specs=(P(), P('model')), out_specs=P())

        ck = lambda f, c, xs: brainstate.transform.checkpointed_scan(f, c, xs, base=2)
        v, g = jax.jit(jax.value_and_grad(make(ck)))(w, x)
        v0, g0 = jax.jit(jax.value_and_grad(make(jax.lax.scan)))(w, x)
        np.testing.assert_allclose(v, v0, rtol=1e-6)
        np.testing.assert_allclose(g, g0, rtol=1e-5)


class TestCheckpointedScanUnsharded(unittest.TestCase):

    def test_unchanged_without_a_mesh(self):
        def f(c):
            return brainstate.transform.checkpointed_scan(lambda c, i: (c * 1.1 + i, c), c, jnp.arange(7.),
                                                          base=2)

        c, ys = jax.jit(f)(jnp.ones(3))
        c0, ys0 = jax.lax.scan(lambda c, i: (c * 1.1 + i, c), jnp.ones(3), jnp.arange(7.))
        np.testing.assert_allclose(ys, ys0, rtol=1e-6)
        self.assertEqual(jax.vmap(lambda c: f(c)[1])(jnp.ones((2, 3))).shape, (2, 7, 3))


if __name__ == '__main__':
    unittest.main()
