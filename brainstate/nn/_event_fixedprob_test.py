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


import jax.numpy
import jax.numpy as jnp
import pytest

import brainstate
import braintools


class TestFixedProbCSR:
    @pytest.mark.parametrize('allow_multi_conn', [True, False, ])
    def test1(self, allow_multi_conn):
        x = brainstate.random.rand(20) < 0.1
        # x = brainstate.random.rand(20)
        m = brainstate.nn.EventFixedProb(20, 40, 0.1, 1.0, seed=123, allow_multi_conn=allow_multi_conn)
        y = m(x)
        print(y)

        m2 = brainstate.nn.EventFixedProb(20, 40, 0.1, braintools.init.KaimingUniform(), seed=123)
        print(m2(x))

    def test_grad_bool(self):
        n_in = 20
        n_out = 30
        x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
        fn = brainstate.nn.EventFixedProb(n_in, n_out, 0.1, braintools.init.KaimingUniform(), seed=123)

        def f(x):
            return fn(x).sum()

        print(jax.grad(f)(x))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vjp(self, homo_w):
        n_in = 20
        n_out = 30
        x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)

        if homo_w:
            fn = brainstate.nn.EventFixedProb(n_in, n_out, 0.1, 1.5, seed=123)
        else:
            fn = brainstate.nn.EventFixedProb(n_in, n_out, 0.1, braintools.init.KaimingUniform(), seed=123)
        w = fn.weight.value

        def f(x, w):
            fn.weight.value = w
            return fn(x).sum()

        r = brainstate.transform.grad(f, argnums=(0, 1))(x, w)

        # -------------------
        # TRUE gradients

        def true_fn(x, w, indices, n_post):
            post = jnp.zeros((n_post,))
            for i in range(n_in):
                post = post.at[indices[i]].add(w * x[i] if homo_w else w[i] * x[i])
            return post

        def f2(x, w):
            return true_fn(x, w, fn.conn.indices, n_out).sum()

        r2 = jax.grad(f2, argnums=(0, 1))(x, w)
        assert (jnp.allclose(r[0], r2[0]))
        assert (jnp.allclose(r[1], r2[1]))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_jvp(self, homo_w):
        n_in = 20
        n_out = 30
        x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)

        fn = brainstate.nn.EventFixedProb(
            n_in, n_out, 0.1, 1.5 if homo_w else braintools.init.KaimingUniform(),
            seed=123,
        )
        w = fn.weight.value

        def f(x, w):
            fn.weight.value = w
            return fn(x)

        o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        # -------------------
        # TRUE gradients

        def true_fn(x, w, indices, n_post):
            post = jnp.zeros((n_post,))
            for i in range(n_in):
                post = post.at[indices[i]].add(w * x[i] if homo_w else w[i] * x[i])
            return post

        def f2(x, w):
            return true_fn(x, w, fn.conn.indices, n_out)

        o2, r2 = jax.jvp(f2, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert (jnp.allclose(o1, o2))
        # assert jnp.allclose(r1, r2), f'r1={r1}, r2={r2}'
        assert (jnp.allclose(r1, r2, rtol=1e-4, atol=1e-4))
