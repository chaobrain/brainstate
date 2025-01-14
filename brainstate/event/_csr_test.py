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
# -*- coding: utf-8 -*-


import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainstate as bst


class TestCSR(unittest.TestCase):
    def test_event_homo_bool(self):
        for dat in [1., 2., 3.]:
            mask = (bst.random.rand(10, 20) < 0.1).astype(float) * dat
            csr = u.sparse.CSR.fromdense(mask)
            csr = bst.event.CSR((dat, csr.indices, csr.indptr), shape=mask.shape)

            v = bst.random.rand(20) < 0.5
            self.assertTrue(
                u.math.allclose(
                    mask.astype(float) @ v.astype(float),
                    csr @ v
                )
            )

            v = bst.random.rand(10) < 0.5
            self.assertTrue(
                u.math.allclose(
                    v.astype(float) @ mask.astype(float),
                    v @ csr
                )
            )

    def test_event_homo_heter(self):
        mat = bst.random.rand(10, 20)
        mask = (bst.random.rand(10, 20) < 0.1) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = bst.event.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = bst.random.rand(20) < 0.5
        self.assertTrue(
            u.math.allclose(
                mask.astype(float) @ v.astype(float),
                csr @ v
            )
        )

        v = bst.random.rand(10) < 0.5
        self.assertTrue(
            u.math.allclose(
                v.astype(float) @ mask.astype(float),
                v @ csr
            )
        )

    def test_event_heter_float_as_bool(self):
        mat = bst.random.rand(10, 20)
        mask = (mat < 0.1).astype(float) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = bst.event.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = (bst.random.rand(20) < 0.5).astype(float)
        self.assertTrue(
            u.math.allclose(
                mask.astype(float) @ v.astype(float),
                csr @ v
            )
        )

        v = (bst.random.rand(10) < 0.5).astype(float)
        self.assertTrue(
            u.math.allclose(
                v.astype(float) @ mask.astype(float),
                v @ csr
            )
        )


def _get_csr(n_pre, n_post, prob):
    n_conn = int(n_post * prob)
    indptr = np.arange(n_pre + 1) * n_conn
    indices = np.random.randint(0, n_post, (n_pre * n_conn,))
    return indptr, indices


def vector_csr(x, w, indices, indptr, shape):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((shape[1],))
    for i_pre in range(x.shape[0]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        post = post.at[ids].add(w * x[i_pre] if homo_w else w[indptr[i_pre]: indptr[i_pre + 1]] * x[i_pre])
    return post


def matrix_csr(xs, w, indices, indptr, shape):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))
    for i_pre in range(xs.shape[1]):
        ids = indices[indptr[i_pre]: indptr[i_pre + 1]]
        post = post.at[:, ids].add(
            w * xs[:, i_pre: i_pre + 1]
            if homo_w else
            (w[indptr[i_pre]: indptr[i_pre + 1]] * xs[:, i_pre: i_pre + 1])
        )
    return post


def csr_vector(x, w, indices, indptr, shape):
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0]])
    for i in range(shape[0]):
        ids = indices[indptr[i]: indptr[i + 1]]
        ws = w if homo_w else w[indptr[i]: indptr[i + 1]]
        out = out.at[i].set(jnp.sum(x[ids] * ws))
    return out


def csr_matrix(xs, w, indices, indptr, shape):
    # CSR @ matrix
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])
    for i in range(shape[0]):
        ids = indices[indptr[i]: indptr[i + 1]]
        ws = w if homo_w else jnp.expand_dims(w[indptr[i]: indptr[i + 1]], axis=1)
        out = out.at[i].set(jnp.sum(xs[ids] * ws, axis=0))
    return out


class TestVectorCSR(unittest.TestCase):
    def test_vector_csr(self, ):
        m, n = 20, 40
        x = bst.random.rand(m) < 0.1
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            print(f'homo_w = {homo_w}')
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = bst.event.CSR([data, indices, indptr], shape=(m, n))
            y = x @ csr
            y2 = vector_csr(x, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))

    def test_vector_csr_vmap_vector(self):
        n_batch, m, n = 10, 20, 40
        xs = bst.random.rand(n_batch, m) < 0.1
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = bst.event.CSR([data, indices, indptr], shape=(m, n))
            y = jax.vmap(lambda x: x @ csr)(xs)
            y2 = jax.vmap(lambda x: vector_csr(x, csr.data, indices, indptr, [m, n]))(xs)
            self.assertTrue(jnp.allclose(y, y2))


class TestMatrixCSR(unittest.TestCase):
    def test_matrix_csr(self):
        k, m, n = 10, 20, 40
        x = bst.random.rand(k, m) < 0.1
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = bst.event.CSR([data, indices, indptr], shape=(m, n))
            y = x @ csr
            y2 = matrix_csr(x, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))


class TestCSRVector(unittest.TestCase):
    def test_csr_vector(self):
        m, n = 20, 40
        v = bst.random.rand(n) < 0.1
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = bst.event.CSR([data, indices, indptr], shape=(m, n))
            y = csr @ v
            y2 = csr_vector(v, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))


class TestCSRMatrix(unittest.TestCase):
    def test_csr_matrix(self):
        m, n, k = 20, 40, 10
        matrix = bst.random.rand(n, k) < 0.1
        indptr, indices = _get_csr(m, n, 0.1)

        for homo_w in [True, False]:
            data = 1.5 if homo_w else bst.init.Normal()(indices.shape)
            csr = bst.event.CSR([data, indices, indptr], shape=(m, n))
            y = csr @ matrix
            y2 = csr_matrix(matrix, csr.data, indices, indptr, [m, n])
            self.assertTrue(jnp.allclose(y, y2))

    # @parameterized.product(
    #     bool_x=[True, False],
    #     homo_w=[True, False]
    # )
    # def test_vjp(self, bool_x, homo_w):
    #     n_in = 20
    #     n_out = 30
    #     if bool_x:
    #         x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
    #     else:
    #         x = bst.random.rand(n_in)
    #
    #     indptr, indices = _get_csr(n_in, n_out, 0.1)
    #     fn = bst.event.CSRLinear(n_in, n_out, indptr, indices, 1.5 if homo_w else bst.init.Normal())
    #     w = fn.weight.value
    #
    #     def f(x, w):
    #         fn.weight.value = w
    #         return fn(x).sum()
    #
    #     r = jax.grad(f, argnums=(0, 1))(x, w)
    #
    #     # -------------------
    #     # TRUE gradients
    #
    #     def f2(x, w):
    #         return true_fn(x, w, indices, indptr, n_out).sum()
    #
    #     r2 = jax.grad(f2, argnums=(0, 1))(x, w)
    #     self.assertTrue(jnp.allclose(r[0], r2[0]))
    #     self.assertTrue(jnp.allclose(r[1], r2[1]))
    #
    # @parameterized.product(
    #     bool_x=[True, False],
    #     homo_w=[True, False]
    # )
    # def test_jvp(self, bool_x, homo_w):
    #     n_in = 20
    #     n_out = 30
    #     if bool_x:
    #         x = jax.numpy.asarray(bst.random.rand(n_in) < 0.3, dtype=float)
    #     else:
    #         x = bst.random.rand(n_in)
    #
    #     indptr, indices = _get_csr(n_in, n_out, 0.1)
    #     fn = bst.event.CSRLinear(n_in, n_out, indptr, indices,
    #                              1.5 if homo_w else bst.init.Normal(), grad_mode='jvp')
    #     w = fn.weight.value
    #
    #     def f(x, w):
    #         fn.weight.value = w
    #         return fn(x)
    #
    #     o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
    #
    #     # -------------------
    #     # TRUE gradients
    #
    #     def f2(x, w):
    #         return true_fn(x, w, indices, indptr, n_out)
    #
    #     o2, r2 = jax.jvp(f2, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
    #     self.assertTrue(jnp.allclose(r1, r2))
    #     self.assertTrue(jnp.allclose(o1, o2))
