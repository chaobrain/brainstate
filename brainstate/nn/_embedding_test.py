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

# -*- coding: utf-8 -*-

import unittest

import jax
import jax.numpy as jnp

import brainstate as bs


class TestEmbedding(unittest.TestCase):
    """Comprehensive tests for the Embedding module."""

    def setUp(self):
        settings = bs.environ.all()
        self._prev_fit = settings.get('fit', None)
        bs.environ.set(fit=True)

    def tearDown(self):
        if self._prev_fit is None:
            bs.environ.pop('fit', None)
        else:
            bs.environ.set(fit=self._prev_fit)

    def test_padding_idx_zero_gradient(self):
        embedding = bs.nn.Embedding(num_embeddings=4, embedding_size=3, padding_idx=0)
        lookup = embedding._lookup
        weight = jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3)
        indices = jnp.array([0, 1, 0, 2], dtype=jnp.int32)

        def loss_fn(w):
            return jnp.sum(lookup(w, indices))

        grad = jax.grad(loss_fn)(weight)
        self.assertTrue(jnp.allclose(grad[0], 0.0))
        self.assertFalse(jnp.allclose(grad[1], 0.0))

    def test_scale_grad_by_freq(self):
        base = bs.nn.Embedding(num_embeddings=5, embedding_size=2)
        scaled = bs.nn.Embedding(num_embeddings=5, embedding_size=2, scale_grad_by_freq=True)
        base_lookup = base._lookup
        scaled_lookup = scaled._lookup

        weight = jnp.arange(10.0, dtype=jnp.float32).reshape(5, 2)
        indices = jnp.array([1, 1, 2], dtype=jnp.int32)

        def loss_base(w):
            return jnp.sum(base_lookup(w, indices))

        def loss_scaled(w):
            return jnp.sum(scaled_lookup(w, indices))

        base_grad = jax.grad(loss_base)(weight)
        scaled_grad = jax.grad(loss_scaled)(weight)

        counts = jnp.bincount(indices, length=weight.shape[0])
        ones = jnp.ones((indices.shape[0], weight.shape[1]), dtype=weight.dtype)
        expected_base = jnp.zeros_like(weight).at[indices].add(ones)
        expected_scaled = jnp.where(counts[:, None] > 0, jnp.ones_like(weight), 0.0)

        self.assertTrue(jnp.allclose(base_grad, expected_base))
        self.assertTrue(jnp.allclose(scaled_grad, expected_scaled))

    def test_lookup_grad_jit_consistent(self):
        embedding = bs.nn.Embedding(num_embeddings=4, embedding_size=2)
        lookup = embedding._lookup
        weight = jnp.arange(8.0, dtype=jnp.float32).reshape(4, 2)
        indices = jnp.array([0, 1, 1, 3], dtype=jnp.int32)

        def loss_fn(w):
            return jnp.sum(lookup(w, indices))

        grad_eager = jax.grad(loss_fn)(weight)
        grad_jitted = jax.grad(jax.jit(loss_fn))(weight)

        expected = jnp.zeros_like(weight).at[indices].add(jnp.ones((indices.shape[0], weight.shape[1]), dtype=weight.dtype))

        self.assertTrue(jnp.allclose(grad_eager, grad_jitted))
        self.assertTrue(jnp.allclose(grad_eager, expected))

    def test_jit_forward_with_max_norm(self):
        embedding = bs.nn.Embedding(num_embeddings=3, embedding_size=3, max_norm=0.5)
        heavy = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [0.3, -0.4, 0.5]], dtype=jnp.float32)
        embedding.weight.value = heavy
        indices = jnp.array([1, 2, 1], dtype=jnp.int32)

        compiled = jax.jit(lambda ids: embedding(ids))
        out = compiled(indices)
        self.assertEqual(out.shape, (3, 3))
        output_norms = jnp.linalg.norm(out, axis=-1)
        self.assertTrue(jnp.all(output_norms <= 0.5 + 1e-6))
        # Weight remains unclipped during JIT execution but must be usable without tracer leaks
        self.assertGreater(float(jnp.linalg.norm(embedding.weight.value[1])), 0.5)

    def test_freeze_disables_gradients(self):
        embedding = bs.nn.Embedding(num_embeddings=4, embedding_size=2, freeze=True)
        indices = jnp.array([1, 2, 3], dtype=jnp.int32)

        def loss_fn(weight):
            embedding.weight.value = weight
            return jnp.sum(embedding(indices))

        grad = jax.grad(loss_fn)(embedding.weight.value)
        self.assertTrue(jnp.allclose(grad, 0.0))

    def test_from_pretrained_defaults_to_freeze(self):
        pretrained = jnp.arange(12.0, dtype=jnp.float32).reshape(4, 3)
        embedding = bs.nn.Embedding.from_pretrained(pretrained)
        self.assertTrue(embedding.freeze)

        def loss_fn(weight):
            embedding.weight.value = weight
            return jnp.sum(embedding(jnp.array([1, 2], dtype=jnp.int32)))

        grad = jax.grad(loss_fn)(embedding.weight.value)
        self.assertTrue(jnp.allclose(grad, 0.0))

    def test_from_pretrained_unfrozen_gradients(self):
        pretrained = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
        embedding = bs.nn.Embedding.from_pretrained(pretrained, freeze=False)

        def loss_fn(weight):
            embedding.weight.value = weight
            return jnp.sum(embedding(jnp.array([0, 1], dtype=jnp.int32)))

        grad = jax.grad(loss_fn)(embedding.weight.value)
        self.assertFalse(jnp.allclose(grad, 0.0))

    def test_max_norm_renormalizes_weights(self):
        embedding = bs.nn.Embedding(num_embeddings=3, embedding_size=3, max_norm=1.0, norm_type=2.0)
        custom_weight = jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.5, 0.5, 0.5]], dtype=jnp.float32)
        embedding.weight.value = custom_weight
        _ = embedding(jnp.array([1, 2], dtype=jnp.int32))

        row_norm = jnp.linalg.norm(embedding.weight.value[1])
        self.assertLessEqual(float(row_norm), 1.0 + 1e-6)
        self.assertTrue(jnp.allclose(embedding.weight.value[0], custom_weight[0]))


if __name__ == '__main__':
    unittest.main()
