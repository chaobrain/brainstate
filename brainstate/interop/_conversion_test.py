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

"""Forward-equivalence + round-trip conversion tests across nnx / linen / equinox.

Each test builds a model in one framework, converts it, and asserts the converted model produces
numerically identical output. ``pytest.importorskip`` skips a framework's tests when it is not
installed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest

import brainstate
import brainstate.nn as bnn
from brainstate import interop
from brainstate.interop._errors import (ConversionError, MissingDependencyError,
                                        MissingShapeError, UnmappedLayerError,
                                        UnsupportedLayerError,
                                        UnsupportedStructureError)

TOL = dict(rtol=1e-4, atol=1e-4)


def _key():
    return brainstate.random.split_key()


def assert_close(a, b, msg=''):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), err_msg=msg, **TOL)


def bst_forward(layer, x, *, lstm=False, batch=None):
    """Run a brainstate layer in eval mode and return its output."""
    with brainstate.environ.context(fit=False):
        if lstm:
            layer.init_state(batch)
            return layer.update(x)
        return layer(x)


# ===========================================================================
# flax.nnx
# ===========================================================================

class NnxConversionTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def _rngs(self):
        return self.nnx.Rngs(_key())

    # --- Linear / Embedding ------------------------------------------------

    def test_linear_import_and_export(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 3)
        src = nnx.Linear(3, 5, rngs=self._rngs())
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'linear import')
        # export round-trip
        back = interop.to_nnx(dst)
        assert_close(src(x), back(x), 'linear export')

    def test_linear_no_bias(self):
        nnx = self.nnx
        x = brainstate.random.randn(2, 3)
        src = nnx.Linear(3, 5, use_bias=False, rngs=self._rngs())
        dst = interop.from_nnx(src)
        self.assertNotIn('bias', dst.weight.value)
        assert_close(src(x), bst_forward(dst, x), 'linear no-bias')

    def test_embedding(self):
        nnx = self.nnx
        src = nnx.Embed(7, 4, rngs=self._rngs())
        idx = jnp.array([0, 3, 6, 1])
        dst = interop.from_nnx(src)
        assert_close(src(idx), bst_forward(dst, idx), 'embed import')
        assert_close(src(idx), interop.to_nnx(dst)(idx), 'embed export')

    # --- Normalization -----------------------------------------------------

    def test_layernorm(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 6)
        src = nnx.LayerNorm(6, rngs=self._rngs())
        # give scale/bias non-trivial values
        src.scale[...] = brainstate.random.randn(6)
        src.bias[...] = brainstate.random.randn(6)
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'layernorm import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'layernorm export')

    def test_rmsnorm(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 6)
        src = nnx.RMSNorm(6, rngs=self._rngs())
        src.scale[...] = brainstate.random.randn(6)
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'rmsnorm import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'rmsnorm export')

    def test_groupnorm(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 8)
        src = nnx.GroupNorm(8, num_groups=2, rngs=self._rngs())
        src.scale[...] = brainstate.random.randn(8)
        src.bias[...] = brainstate.random.randn(8)
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'groupnorm import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'groupnorm export')

    # --- Conv --------------------------------------------------------------

    def test_conv2d(self):
        nnx = self.nnx
        x = brainstate.random.randn(2, 8, 8, 3)
        src = nnx.Conv(3, 4, (3, 3), rngs=self._rngs())
        dst = interop.from_nnx(src, sample_input=(8, 8, 3))
        assert_close(src(x), bst_forward(dst, x), 'conv2d import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'conv2d export')

    def test_conv1d(self):
        nnx = self.nnx
        x = brainstate.random.randn(2, 16, 3)
        src = nnx.Conv(3, 5, (3,), rngs=self._rngs())
        dst = interop.from_nnx(src, sample_input=(16, 3))
        assert_close(src(x), bst_forward(dst, x), 'conv1d import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'conv1d export')

    # --- BatchNorm ---------------------------------------------------------

    def test_batchnorm_import_export(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 5, 3)
        src = nnx.BatchNorm(3, use_running_average=True, rngs=self._rngs())
        src.scale[...] = brainstate.random.randn(3)
        src.bias[...] = brainstate.random.randn(3)
        src.mean[...] = brainstate.random.randn(3)
        src.var[...] = brainstate.random.rand(3) + 0.5
        dst = interop.from_nnx(src, sample_input=(5, 3))
        assert_close(src(x), bst_forward(dst, x), 'batchnorm import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'batchnorm export')

    # --- LSTM --------------------------------------------------------------

    def test_lstm(self):
        nnx = self.nnx
        x = brainstate.random.randn(2, 3)
        src = nnx.LSTMCell(3, 4, rngs=self._rngs())
        dst = interop.from_nnx(src)
        h_bst = bst_forward(dst, x, lstm=True, batch=2)
        carry = (jnp.zeros((2, 4)), jnp.zeros((2, 4)))
        (_, h_src), _ = src(carry, x)
        assert_close(h_src, h_bst, 'lstm import')
        # export
        back = interop.to_nnx(dst)
        (_, h_back), _ = back(carry, x)
        assert_close(h_src, h_back, 'lstm export')

    # --- Dropout / Sequential ----------------------------------------------

    def test_dropout_keep_prob(self):
        nnx = self.nnx
        src = nnx.Dropout(0.3, rngs=self._rngs())
        dst = interop.from_nnx(src)
        self.assertAlmostEqual(float(dst.prob), 0.7, places=6)
        back = interop.to_nnx(dst)
        self.assertAlmostEqual(float(back.rate), 0.3, places=6)

    def test_sequential(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 3)
        src = nnx.Sequential(
            nnx.Linear(3, 6, rngs=self._rngs()),
            nnx.LayerNorm(6, rngs=self._rngs()),
            nnx.Linear(6, 2, rngs=self._rngs()),
        )
        dst = interop.from_nnx(src)
        self.assertIsInstance(dst, bnn.Sequential)
        assert_close(src(x), bst_forward(dst, x), 'sequential import')
        assert_close(src(x), interop.to_nnx(dst)(x), 'sequential export')

    # --- error paths -------------------------------------------------------

    def test_gru_unsupported_both_directions(self):
        nnx = self.nnx
        with self.assertRaises(UnsupportedLayerError):
            interop.from_nnx(nnx.GRUCell(3, 4, rngs=self._rngs()))
        with self.assertRaises(UnsupportedLayerError):
            interop.to_nnx(bnn.GRUCell(3, 4))

    def test_unmapped_leaf(self):
        nnx = self.nnx

        class Custom(nnx.Module):
            def __init__(self):
                self.scalar = 1.0

        with self.assertRaises(UnmappedLayerError):
            interop.from_nnx(Custom())

    def test_unsupported_structure(self):
        nnx = self.nnx
        rngs = self._rngs()

        class TwoBranch(nnx.Module):
            def __init__(self, rngs):
                self.a = nnx.Linear(3, 3, rngs=rngs)
                self.b = nnx.Linear(3, 3, rngs=rngs)

        with self.assertRaises(UnsupportedStructureError):
            interop.from_nnx(TwoBranch(rngs))

    def test_missing_shape_for_conv(self):
        nnx = self.nnx
        with self.assertRaises(MissingShapeError):
            interop.from_nnx(nnx.Conv(3, 4, (3, 3), rngs=self._rngs()))


# ===========================================================================
# flax.linen
# ===========================================================================

class LinenConversionTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nn = pytest.importorskip('flax.linen', exc_type=ImportError)

    def _apply(self, module, variables, *args, **kw):
        return module.apply(variables, *args, **kw)

    def test_linear(self):
        nn = self.nn
        x = brainstate.random.randn(4, 3)
        m = nn.Dense(5)
        v = m.init(_key(), jnp.ones((1, 3)))
        dst = interop.from_linen(m, v)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'dense import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), 'dense export')

    def test_embedding(self):
        nn = self.nn
        m = nn.Embed(7, 4)
        v = m.init(_key(), jnp.array([[0, 1]]))
        idx = jnp.array([0, 3, 6])
        dst = interop.from_linen(m, v)
        assert_close(m.apply(v, idx), bst_forward(dst, idx), 'embed import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, idx), m2.apply(v2, idx), 'embed export')

    def test_layernorm(self):
        nn = self.nn
        x = brainstate.random.randn(4, 6)
        m = nn.LayerNorm()
        v = m.init(_key(), jnp.ones((1, 6)))
        v = jax.tree_util.tree_map(lambda a: brainstate.random.randn(*a.shape)
                                   if a.ndim else a, v)
        dst = interop.from_linen(m, v)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'layernorm import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), 'layernorm export')

    def test_rmsnorm(self):
        nn = self.nn
        x = brainstate.random.randn(4, 6)
        m = nn.RMSNorm()
        v = m.init(_key(), jnp.ones((1, 6)))
        dst = interop.from_linen(m, v)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'rmsnorm import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), 'rmsnorm export')

    def test_groupnorm(self):
        nn = self.nn
        x = brainstate.random.randn(4, 8)
        m = nn.GroupNorm(num_groups=2)
        v = m.init(_key(), jnp.ones((1, 8)))
        dst = interop.from_linen(m, v)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'groupnorm import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), 'groupnorm export')

    def test_conv2d(self):
        nn = self.nn
        x = brainstate.random.randn(2, 8, 8, 3)
        m = nn.Conv(features=4, kernel_size=(3, 3))
        v = m.init(_key(), jnp.ones((1, 8, 8, 3)))
        dst = interop.from_linen(m, v, sample_input=(8, 8, 3))
        assert_close(m.apply(v, x), bst_forward(dst, x), 'conv2d import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), 'conv2d export')

    def test_batchnorm(self):
        nn = self.nn
        x = brainstate.random.randn(4, 5, 3)
        m = nn.BatchNorm(use_running_average=True)
        v = m.init(_key(), jnp.ones((1, 5, 3)))
        v = jax.tree_util.tree_map(
            lambda a: brainstate.random.rand(*a.shape) + 0.5 if a.ndim == 1 else a, v)
        dst = interop.from_linen(m, v, sample_input=(5, 3))
        assert_close(m.apply(v, x), bst_forward(dst, x), 'batchnorm import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), 'batchnorm export')

    def test_lstm(self):
        nn = self.nn
        x = brainstate.random.randn(2, 3)
        m = nn.LSTMCell(features=4)
        carry0 = (jnp.zeros((2, 4)), jnp.zeros((2, 4)))
        v = m.init(_key(), carry0, jnp.ones((2, 3)))
        dst = interop.from_linen(m, v)
        h_bst = bst_forward(dst, x, lstm=True, batch=2)
        (_, h_src), _ = m.apply(v, carry0, x)
        assert_close(h_src, h_bst, 'lstm import')
        m2, v2 = interop.to_linen(dst)
        (_, h_back), _ = m2.apply(v2, carry0, x)
        assert_close(h_src, h_back, 'lstm export')

    def test_sequential_import(self):
        nn = self.nn
        x = brainstate.random.randn(4, 3)
        m = nn.Sequential([nn.Dense(6), nn.LayerNorm(), nn.Dense(2)])
        v = m.init(_key(), jnp.ones((1, 3)))
        dst = interop.from_linen(m, v)
        self.assertIsInstance(dst, bnn.Sequential)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'linen seq import')

    def test_sequential_with_dropout(self):
        nn = self.nn
        x = brainstate.random.randn(4, 3)
        src = bnn.Sequential(bnn.Linear(3, 6), bnn.Dropout(1.0), bnn.Linear(6, 2))
        m, v = interop.to_linen(src)
        # Dropout slot carries no params.
        self.assertEqual(sorted(v['params'].keys()), ['layers_0', 'layers_2'])
        assert_close(bst_forward(src, x), m.apply(v, x), 'seq+dropout export')

    def test_unsupported_structure(self):
        nn = self.nn

        class TwoDense(nn.Module):
            @nn.compact
            def __call__(self, x):
                return nn.Dense(2)(nn.Dense(4)(x))

        m = TwoDense()
        v = m.init(_key(), jnp.ones((1, 3)))
        with self.assertRaises(UnsupportedStructureError):
            interop.from_linen(m, v)

    def test_gru_unsupported(self):
        nn = self.nn
        m = nn.GRUCell(features=4)
        v = m.init(_key(), jnp.zeros((1, 4)), jnp.ones((1, 3)))
        with self.assertRaises(UnsupportedLayerError):
            interop.from_linen(m, v)

    def test_missing_shape_for_conv(self):
        nn = self.nn
        m = nn.Conv(features=4, kernel_size=(3, 3))
        v = m.init(_key(), jnp.ones((1, 8, 8, 3)))
        with self.assertRaises(MissingShapeError):
            interop.from_linen(m, v)


# ===========================================================================
# equinox
# ===========================================================================

class EquinoxConversionTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eqx = pytest.importorskip('equinox', exc_type=ImportError)

    def test_linear(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 3)
        src = eqx.nn.Linear(3, 5, key=_key())
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'linear import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'linear export')

    def test_embedding(self):
        eqx = self.eqx
        src = eqx.nn.Embedding(7, 4, key=_key())
        idx = jnp.array([0, 3, 6])
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(idx), bst_forward(dst, idx), 'embed import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(idx), jax.vmap(back)(idx), 'embed export')

    def test_layernorm(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 6)
        src = eqx.nn.LayerNorm(6)
        # set random weight/bias
        src = eqx.tree_at(lambda m: (m.weight, m.bias), src,
                          (brainstate.random.randn(6), brainstate.random.randn(6)))
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'layernorm import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'layernorm export')

    def test_rmsnorm(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 6)
        src = eqx.nn.RMSNorm(6)
        src = eqx.tree_at(lambda m: m.weight, src, brainstate.random.randn(6))
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'rmsnorm import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'rmsnorm export')

    def test_groupnorm(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 8)
        src = eqx.nn.GroupNorm(groups=2, channels=8)
        src = eqx.tree_at(lambda m: (m.weight, m.bias), src,
                          (brainstate.random.randn(8), brainstate.random.randn(8)))
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'groupnorm import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'groupnorm export')

    def test_conv2d(self):
        eqx = self.eqx
        # equinox conv is NCHW, unbatched
        x_nchw = brainstate.random.randn(2, 3, 8, 8)
        src = eqx.nn.Conv2d(3, 4, 3, key=_key())
        dst = interop.from_equinox(src, sample_input=(8, 8, 3))
        # brainstate is NHWC
        x_nhwc = jnp.transpose(x_nchw, (0, 2, 3, 1))
        y_src = jax.vmap(src)(x_nchw)               # (N, out, H, W)
        y_src_nhwc = jnp.transpose(y_src, (0, 2, 3, 1))
        assert_close(y_src_nhwc, bst_forward(dst, x_nhwc), 'conv2d import')
        back = interop.to_equinox(dst)
        y_back = jnp.transpose(jax.vmap(back)(x_nchw), (0, 2, 3, 1))
        assert_close(y_src_nhwc, y_back, 'conv2d export')

    def test_lstm(self):
        eqx = self.eqx
        x = brainstate.random.randn(2, 3)
        src = eqx.nn.LSTMCell(3, 4, key=_key())

        def run_src(xi):
            h, c = src(xi, (jnp.zeros(4), jnp.zeros(4)))
            return h

        dst = interop.from_equinox(src)
        h_bst = bst_forward(dst, x, lstm=True, batch=2)
        assert_close(jax.vmap(run_src)(x), h_bst, 'lstm import')
        back = interop.to_equinox(dst)

        def run_back(xi):
            h, c = back(xi, (jnp.zeros(4), jnp.zeros(4)))
            return h

        assert_close(jax.vmap(run_src)(x), jax.vmap(run_back)(x), 'lstm export')

    def test_dropout_keep_prob(self):
        eqx = self.eqx
        src = eqx.nn.Dropout(0.3)
        dst = interop.from_equinox(src)
        self.assertAlmostEqual(float(dst.prob), 0.7, places=6)
        back = interop.to_equinox(dst)
        self.assertAlmostEqual(float(back.p), 0.3, places=6)

    def test_sequential(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 3)
        src = eqx.nn.Sequential([
            eqx.nn.Linear(3, 6, key=_key()),
            eqx.nn.Linear(6, 2, key=_key()),
        ])
        dst = interop.from_equinox(src)
        self.assertIsInstance(dst, bnn.Sequential)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'sequential import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'sequential export')

    def test_to_equinox_with_explicit_key(self):
        # Regression: ``to_equinox(model, key=<PRNGKey>)`` must not evaluate the
        # truthiness of the key array. ``ctx.key or new_key()`` raised
        # "truth value of an array ... is ambiguous" for any weight-bearing layer.
        eqx = self.eqx
        x = brainstate.random.randn(4, 3)
        model = bnn.Sequential(bnn.Linear(3, 6), bnn.LayerNorm(6), bnn.Linear(6, 2))
        y = model(x)
        # Single layer and a Sequential, both with an explicit user key.
        single = interop.to_equinox(bnn.Linear(3, 2), key=jax.random.PRNGKey(0))
        self.assertIsInstance(single, eqx.nn.Linear)
        exported = interop.to_equinox(model, key=jax.random.PRNGKey(0))
        assert_close(y, jax.vmap(exported)(x), 'export with explicit key')

    def test_batchnorm_unsupported_both_directions(self):
        eqx = self.eqx
        # export bst BatchNorm -> equinox
        with self.assertRaises(UnsupportedLayerError):
            interop.to_equinox(bnn.BatchNorm1d((5, 3)))
        # import equinox BatchNorm -> bst
        bn = eqx.nn.BatchNorm(input_size=3, axis_name='batch')
        with self.assertRaises(UnsupportedLayerError):
            interop.from_equinox(bn)

    def test_gru_unsupported(self):
        eqx = self.eqx
        with self.assertRaises(UnsupportedLayerError):
            interop.from_equinox(eqx.nn.GRUCell(3, 4, key=_key()))

    def test_unsupported_structure(self):
        eqx = self.eqx

        class TwoLinear(eqx.Module):
            a: eqx.nn.Linear
            b: eqx.nn.Linear

            def __init__(self, key):
                k1, k2 = jax.random.split(key)
                self.a = eqx.nn.Linear(3, 3, key=k1)
                self.b = eqx.nn.Linear(3, 3, key=k2)

            def __call__(self, x):
                return self.b(self.a(x))

        with self.assertRaises(UnsupportedStructureError):
            interop.from_equinox(TwoLinear(_key()))


# ===========================================================================
# custom mapping + public API + missing dependency
# ===========================================================================

class CustomMappingTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def test_register_custom_layer_mapping(self):
        nnx = self.nnx
        from brainstate.interop import LayerMapping, register_layer_mapping

        # A user-defined nnx leaf wrapping a scale factor.
        class ScaleLayer(nnx.Module):
            def __init__(self, factor):
                self.factor = factor

            def __call__(self, x):
                return x * self.factor

        # A brainstate counterpart.
        class BstScale(bnn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor

            def update(self, x):
                return x * self.factor

        register_layer_mapping(LayerMapping(
            BstScale, 'nnx', ScaleLayer,
            to_bst=lambda m, ctx: BstScale(m.factor),
            to_foreign=lambda layer, ctx: ScaleLayer(layer.factor),
        ))
        x = brainstate.random.randn(3)
        dst = interop.from_nnx(ScaleLayer(2.0))
        self.assertIsInstance(dst, BstScale)
        assert_close(dst.update(x), x * 2.0)
        back = interop.to_nnx(dst)
        self.assertIsInstance(back, ScaleLayer)


class EngineErrorPathsTest(absltest.TestCase):
    """Container-structure error paths shared by the generic engine (exercised via nnx)."""

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def test_empty_sequential_import(self):
        nnx = self.nnx
        with self.assertRaises(UnsupportedStructureError):
            interop.from_nnx(nnx.Sequential())

    def test_bst_custom_container_attr(self):
        class Container(bnn.Module):
            def __init__(self):
                super().__init__()
                self.a = bnn.Linear(3, 3)

            def update(self, x):
                return x

        with self.assertRaises(UnsupportedStructureError):
            interop.to_nnx(Container())

    def test_bst_custom_container_list(self):
        # Children held only inside a list (exercises the list branch of child detection).
        class Container(bnn.Module):
            def __init__(self):
                super().__init__()
                self.items = [bnn.Linear(3, 3), bnn.Linear(3, 3)]

            def update(self, x):
                return x

        with self.assertRaises(UnsupportedStructureError):
            interop.to_nnx(Container())

    def test_bst_custom_container_dict(self):
        # Children held only inside a dict (exercises the dict branch of child detection).
        class Container(bnn.Module):
            def __init__(self):
                super().__init__()
                self.mapped = {'x': bnn.Linear(3, 3)}

            def update(self, x):
                return x

        with self.assertRaises(UnsupportedStructureError):
            interop.to_nnx(Container())

    def test_bst_unmapped_leaf_export(self):
        class Leaf(bnn.Module):
            def __init__(self):
                super().__init__()
                self.scalar = 1.0

            def update(self, x):
                return x

        with self.assertRaises(UnmappedLayerError):
            interop.to_nnx(Leaf())


class PublicApiTest(absltest.TestCase):

    def test_supported_layers_lists_core_types(self):
        pytest.importorskip('flax.nnx', exc_type=ImportError)
        table = interop.supported_layers('nnx')
        self.assertIn('nnx', table)
        self.assertIn('Linear', table['nnx'])
        self.assertIn('LSTMCell', table['nnx'])

    def test_missing_dependency_error(self):
        from brainstate.interop._common import lazy_import
        with self.assertRaises(MissingDependencyError):
            lazy_import('a_framework_that_does_not_exist_xyz')


# ===========================================================================
# Edge-case tests for audit-discovered bugs
# ===========================================================================

class NnxNormNoAffineTest(absltest.TestCase):
    """Norm layers with all affine params disabled must not crash."""

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def _rngs(self):
        return self.nnx.Rngs(brainstate.random.split_key())

    def test_layernorm_no_affine_import_export(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 6)
        src = nnx.LayerNorm(6, use_scale=False, use_bias=False, rngs=self._rngs())
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'layernorm no-affine import')
        back = interop.to_nnx(dst)
        assert_close(src(x), back(x), 'layernorm no-affine export')

    def test_groupnorm_no_affine_import_export(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 8)
        src = nnx.GroupNorm(8, num_groups=2, use_scale=False, use_bias=False, rngs=self._rngs())
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'groupnorm no-affine import')
        back = interop.to_nnx(dst)
        assert_close(src(x), back(x), 'groupnorm no-affine export')

    def test_layernorm_scale_only(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 6)
        src = nnx.LayerNorm(6, use_scale=True, use_bias=False, rngs=self._rngs())
        src.scale[...] = brainstate.random.randn(6)
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'layernorm scale-only import')
        back = interop.to_nnx(dst)
        assert_close(src(x), back(x), 'layernorm scale-only export')

    def test_layernorm_bias_only(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 6)
        src = nnx.LayerNorm(6, use_scale=False, use_bias=True, rngs=self._rngs())
        src.bias[...] = brainstate.random.randn(6)
        dst = interop.from_nnx(src)
        assert_close(src(x), bst_forward(dst, x), 'layernorm bias-only import')
        back = interop.to_nnx(dst)
        assert_close(src(x), back(x), 'layernorm bias-only export')


class EquinoxNormNoAffineTest(absltest.TestCase):
    """Norm layers with no affine params in equinox."""

    @classmethod
    def setUpClass(cls):
        cls.eqx = pytest.importorskip('equinox', exc_type=ImportError)

    def test_layernorm_no_affine_import_export(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 6)
        src = eqx.nn.LayerNorm(6, use_weight=False, use_bias=False)
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'eqx layernorm no-affine import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'eqx layernorm no-affine export')

    def test_groupnorm_no_affine_import_export(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 8)
        src = eqx.nn.GroupNorm(groups=2, channels=8, channelwise_affine=False)
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'eqx groupnorm no-affine import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'eqx groupnorm no-affine export')

    def test_rmsnorm_no_scale_import_export(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 6)
        src = eqx.nn.RMSNorm(6, use_weight=False, use_bias=False)
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'eqx rmsnorm no-scale import')
        back = interop.to_equinox(dst)
        assert_close(jax.vmap(src)(x), jax.vmap(back)(x), 'eqx rmsnorm no-scale export')


class NnxConvInputDilationTest(absltest.TestCase):
    """Conv with input_dilation != 1 should raise ConversionError, not silently succeed."""

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def test_input_dilation_raises(self):
        nnx = self.nnx
        src = nnx.Conv(3, 4, (3, 3), input_dilation=(2, 2), rngs=nnx.Rngs(0))
        with self.assertRaises(ConversionError):
            interop.from_nnx(src, sample_input=(8, 8, 3))


class BiasOnlyBatchNormTest(absltest.TestCase):
    """H2/H5: bias-only BatchNorm (use_scale=False, use_bias=True) must keep its bias."""

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)
        cls.nn = pytest.importorskip('flax.linen', exc_type=ImportError)

    def test_bias_only_batchnorm_from_nnx(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 5, 3)
        src = nnx.BatchNorm(3, use_scale=False, use_bias=True,
                            use_running_average=True, rngs=nnx.Rngs(brainstate.random.split_key()))
        src.bias[...] = brainstate.random.randn(3)
        src.mean[...] = brainstate.random.randn(3)
        src.var[...] = brainstate.random.rand(3) + 0.5
        dst = interop.from_nnx(src, sample_input=(5, 3))
        # bias must survive (affine present)
        self.assertIsNotNone(dst.weight)
        self.assertIn('bias', dst.weight.value)
        self.assertNotIn('scale', dst.weight.value)
        assert_close(src(x), bst_forward(dst, x), 'bias-only batchnorm nnx import')

    def test_bias_only_batchnorm_from_linen(self):
        nn = self.nn
        x = brainstate.random.randn(4, 5, 3)
        m = nn.BatchNorm(use_running_average=True, use_scale=False, use_bias=True)
        v = m.init(_key(), jnp.ones((1, 5, 3)))
        v = jax.tree_util.tree_map(
            lambda a: brainstate.random.rand(*a.shape) + 0.5 if a.ndim == 1 else a, v)
        dst = interop.from_linen(m, v, sample_input=(5, 3))
        self.assertIsNotNone(dst.weight)
        self.assertIn('bias', dst.weight.value)
        self.assertNotIn('scale', dst.weight.value)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'bias-only batchnorm linen import')


class TwoDBatchNormRoundTripTest(absltest.TestCase):
    """H3: a BatchNorm over a 2-D (batch, features) input resolves to nd==0 (BatchNorm0d).

    Pre-fix this raised ``KeyError(0)`` because the ``_BN_CLS`` map and the nnx/linen
    registration pairs started at spatial-dim 1. It must round-trip through nnx and linen.
    """

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)
        cls.nn = pytest.importorskip('flax.linen', exc_type=ImportError)

    def test_2d_batchnorm_nnx_round_trip(self):
        nnx = self.nnx
        # (N, C) input -> nd == 0 -> BatchNorm0d
        x = brainstate.random.randn(4, 3)
        src = nnx.BatchNorm(3, use_running_average=True,
                            rngs=nnx.Rngs(brainstate.random.split_key()))
        src.scale[...] = brainstate.random.randn(3)
        src.bias[...] = brainstate.random.randn(3)
        src.mean[...] = brainstate.random.randn(3)
        src.var[...] = brainstate.random.rand(3) + 0.5
        dst = interop.from_nnx(src, sample_input=(3,))
        self.assertIsInstance(dst, bnn.BatchNorm0d)
        assert_close(src(x), bst_forward(dst, x), '2d batchnorm nnx import')
        assert_close(src(x), interop.to_nnx(dst)(x), '2d batchnorm nnx export')

    def test_2d_batchnorm_linen_round_trip(self):
        nn = self.nn
        x = brainstate.random.randn(4, 3)
        m = nn.BatchNorm(use_running_average=True)
        v = m.init(_key(), jnp.ones((1, 3)))
        v = jax.tree_util.tree_map(
            lambda a: brainstate.random.rand(*a.shape) + 0.5 if a.ndim == 1 else a, v)
        dst = interop.from_linen(m, v, sample_input=(3,))
        self.assertIsInstance(dst, bnn.BatchNorm0d)
        assert_close(m.apply(v, x), bst_forward(dst, x), '2d batchnorm linen import')
        m2, v2 = interop.to_linen(dst)
        assert_close(m.apply(v, x), m2.apply(v2, x), '2d batchnorm linen export')


class LinenGroupNormGroupSizeTest(absltest.TestCase):
    """M8: linen GroupNorm with num_groups=None + group_size must not crash on int(None)."""

    @classmethod
    def setUpClass(cls):
        cls.nn = pytest.importorskip('flax.linen', exc_type=ImportError)

    def test_groupnorm_group_size_only(self):
        nn = self.nn
        x = brainstate.random.randn(1, 12)
        m = nn.GroupNorm(num_groups=None, group_size=4)
        v = m.init(_key(), jnp.ones((1, 12)))
        v = jax.tree_util.tree_map(lambda a: brainstate.random.randn(*a.shape)
                                   if a.ndim else a, v)
        dst = interop.from_linen(m, v)
        # 12 channels / group_size 4 == 3 groups
        self.assertEqual(int(dst.num_groups), 3)
        assert_close(m.apply(v, x), bst_forward(dst, x), 'groupnorm group_size-only import')


class EquinoxRmsNormBiasTest(absltest.TestCase):
    """H4: equinox RMSNorm with a non-zero bias cannot be represented and must raise."""

    @classmethod
    def setUpClass(cls):
        cls.eqx = pytest.importorskip('equinox', exc_type=ImportError)

    def test_nonzero_bias_raises(self):
        eqx = self.eqx
        src = eqx.nn.RMSNorm(6, use_bias=True)
        src = eqx.tree_at(lambda m: m.bias, src, brainstate.random.randn(6))
        with self.assertRaises(ConversionError):
            interop.from_equinox(src)

    def test_zero_bias_still_converts(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 6)
        # default RMSNorm has a zero (not None) bias; conversion must still succeed.
        src = eqx.nn.RMSNorm(6)
        src = eqx.tree_at(lambda m: m.weight, src, brainstate.random.randn(6))
        dst = interop.from_equinox(src)
        assert_close(jax.vmap(src)(x), bst_forward(dst, x), 'eqx zero-bias rmsnorm import')


class NnxDtypePreservationTest(absltest.TestCase):
    """M9: non-float32 weights must keep their dtype through from_nnx/to_nnx round-trips.

    Sources are built directly with ``param_dtype`` so their weights carry a non-float32 dtype.
    ``from_nnx`` must preserve it into the brainstate layer, and ``to_nnx`` must thread it back
    into the rebuilt nnx layer (whose constructors otherwise default param_dtype to float32).
    """

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def _rngs(self):
        return self.nnx.Rngs(brainstate.random.split_key())

    def test_linear_bf16_round_trip(self):
        nnx = self.nnx
        dt = jnp.bfloat16
        src = nnx.Linear(3, 5, param_dtype=dt, rngs=self._rngs())
        dst = interop.from_nnx(src)
        self.assertEqual(dst.weight.value['weight'].dtype, dt)
        back = interop.to_nnx(dst)
        self.assertEqual(back.kernel.value.dtype, dt)
        self.assertEqual(back.bias.value.dtype, dt)

    def test_conv_fp16_round_trip(self):
        nnx = self.nnx
        dt = jnp.float16
        src = nnx.Conv(3, 4, (3, 3), param_dtype=dt, rngs=self._rngs())
        dst = interop.from_nnx(src, sample_input=(8, 8, 3))
        self.assertEqual(dst.weight.value['weight'].dtype, dt)
        back = interop.to_nnx(dst)
        self.assertEqual(back.kernel.value.dtype, dt)

    def test_batchnorm_bf16_round_trip(self):
        nnx = self.nnx
        dt = jnp.bfloat16
        # nnx forces running stats to float32 on construction; build the brainstate side with
        # bf16 running stats explicitly so the export path is what must preserve the dtype.
        layer = bnn.BatchNorm1d((5, 3), affine=True)
        # Cast every stored array to bf16.
        layer.weight.value = {k: v.astype(dt) for k, v in layer.weight.value.items()}
        layer.running_mean.value = layer.running_mean.value.astype(dt)
        layer.running_var.value = layer.running_var.value.astype(dt)
        back = interop.to_nnx(layer)
        self.assertEqual(back.scale.value.dtype, dt)
        # running stats: nnx forces float32 on construction; the export must rebuild them.
        self.assertEqual(back.mean.value.dtype, dt)
        self.assertEqual(back.var.value.dtype, dt)

    def test_lstm_bf16_round_trip(self):
        nnx = self.nnx
        dt = jnp.bfloat16
        src = nnx.LSTMCell(3, 4, param_dtype=dt, rngs=self._rngs())
        dst = interop.from_nnx(src)
        self.assertEqual(dst.W.weight.value['weight'].dtype, dt)
        back = interop.to_nnx(dst)
        self.assertEqual(back.ii.kernel.value.dtype, dt)
        self.assertEqual(back.hi.bias.value.dtype, dt)


class DropoutEvalEquivalenceTest(absltest.TestCase):
    """M10: exported Dropout must be deterministic (eval-equivalent), not stochastic."""

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)
        cls.eqx = pytest.importorskip('equinox', exc_type=ImportError)

    def test_to_nnx_dropout_deterministic(self):
        nnx = self.nnx
        x = brainstate.random.randn(4, 6)
        src = bnn.Sequential(bnn.Linear(6, 6), bnn.Dropout(0.5), bnn.Linear(6, 6))
        out = interop.to_nnx(src)
        # Output-equivalent to brainstate eval-mode forward (deterministic Dropout = identity).
        assert_close(bst_forward(src, x), out(x), 'to_nnx dropout eval-equivalent')

    def test_to_equinox_dropout_inference(self):
        eqx = self.eqx
        x = brainstate.random.randn(4, 6)
        src = bnn.Sequential(bnn.Linear(6, 6), bnn.Dropout(0.5), bnn.Linear(6, 6))
        out = interop.to_equinox(src)
        assert_close(bst_forward(src, x), jax.vmap(out)(x), 'to_equinox dropout eval-equivalent')


class NnxConvChannelMismatchTest(absltest.TestCase):
    """L8: from_nnx of a Conv with a sample_input channel count that mismatches the kernel."""

    @classmethod
    def setUpClass(cls):
        cls.nnx = pytest.importorskip('flax.nnx', exc_type=ImportError)

    def test_channel_mismatch_raises(self):
        nnx = self.nnx
        # kernel expects 3 input channels; sample_input declares 5 -> clear ConversionError.
        src = nnx.Conv(3, 4, (3, 3), rngs=nnx.Rngs(0))
        with self.assertRaises(ConversionError):
            interop.from_nnx(src, sample_input=(8, 8, 5))


if __name__ == '__main__':
    absltest.main()
