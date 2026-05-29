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

import brainstate


class TestExports(unittest.TestCase):
    def test_symbols_exist(self):
        self.assertTrue(callable(brainstate.transform.custom_vjp))
        self.assertTrue(callable(brainstate.transform.custom_jvp))


class TestCustomVjp(unittest.TestCase):
    def test_value(self):
        w = brainstate.State(jnp.array(2.0))
        cf = brainstate.transform.custom_vjp(lambda x: w.value * x, grad_states=[w])

        @cf.def_fwd
        def fwd(x):
            return w.value * x, (x, w.value)

        @cf.def_bwd
        def bwd(res, ct):
            x, wv = res
            return [ct * x], ct * wv  # (state_grads, arg_grads)

        self.assertTrue(jnp.allclose(cf(jnp.array(3.0)), 6.0))

    def test_custom_grad_wrt_args(self):
        w = brainstate.State(jnp.array(2.0))
        cf = brainstate.transform.custom_vjp(lambda x: w.value * x, grad_states=[w])

        @cf.def_fwd
        def fwd(x):
            return w.value * x, (x, w.value)

        @cf.def_bwd
        def bwd(res, ct):
            x, wv = res
            return [100.0 * ct * x], 100.0 * ct * wv  # deliberately ×100

        gx = jax.grad(lambda x: jnp.sum(cf(x)))(jnp.array(3.0))
        self.assertTrue(jnp.allclose(gx, 100.0 * 2.0))  # 100 * w

    def test_custom_grad_wrt_grad_states(self):
        w = brainstate.State(jnp.array(2.0))
        cf = brainstate.transform.custom_vjp(lambda x: w.value * x, grad_states=[w])

        @cf.def_fwd
        def fwd(x):
            return w.value * x, (x, w.value)

        @cf.def_bwd
        def bwd(res, ct):
            x, wv = res
            return [100.0 * ct * x], 100.0 * ct * wv

        gw = brainstate.transform.grad(lambda x: jnp.sum(cf(x)), grad_states=[w])(jnp.array(3.0))
        gw_val = gw[0] if isinstance(gw, (list, tuple)) else gw
        self.assertTrue(jnp.allclose(gw_val, 100.0 * 3.0))  # 100 * x

    def test_no_grad_states_args_only(self):
        cf = brainstate.transform.custom_vjp(lambda x: x ** 2)

        @cf.def_fwd
        def fwd(x):
            return x ** 2, (x,)

        @cf.def_bwd
        def bwd(res, ct):
            (x,) = res
            return (), 100.0 * ct * x  # empty state_grads

        gx = jax.grad(lambda x: jnp.sum(cf(x)))(jnp.array(3.0))
        self.assertTrue(jnp.allclose(gx, 100.0 * 3.0))

    def test_error_without_rules(self):
        cf = brainstate.transform.custom_vjp(lambda x: x ** 2)
        with self.assertRaises(RuntimeError):
            jax.grad(lambda x: jnp.sum(cf(x)))(jnp.array(3.0))


class TestCustomJvp(unittest.TestCase):
    def test_value(self):
        w = brainstate.State(jnp.array(2.0))
        cf = brainstate.transform.custom_jvp(lambda x: w.value * x)

        @cf.def_jvp
        def jvp_rule(primals, tangents):
            (x,) = primals
            (xt,) = tangents
            return w.value * x, w.value * xt

        self.assertTrue(jnp.allclose(cf(jnp.array(3.0)), 6.0))

    def test_custom_tangent(self):
        w = brainstate.State(jnp.array(2.0))
        cf = brainstate.transform.custom_jvp(lambda x: w.value * x)

        @cf.def_jvp
        def jvp_rule(primals, tangents):
            (x,) = primals
            (xt,) = tangents
            return w.value * x, 100.0 * w.value * xt  # deliberately ×100

        out, tan = jax.jvp(lambda x: cf(x), (jnp.array(3.0),), (jnp.array(1.0),))
        self.assertTrue(jnp.allclose(out, 6.0))
        self.assertTrue(jnp.allclose(tan, 100.0 * 2.0))  # 100 * w

    def test_grad_from_jvp(self):
        w = brainstate.State(jnp.array(2.0))
        cf = brainstate.transform.custom_jvp(lambda x: w.value * x)

        @cf.def_jvp
        def jvp_rule(primals, tangents):
            (x,) = primals
            (xt,) = tangents
            return w.value * x, 100.0 * w.value * xt

        gx = jax.grad(lambda x: jnp.sum(cf(x)))(jnp.array(3.0))
        self.assertTrue(jnp.allclose(gx, 100.0 * 2.0))

    def test_decorator_form(self):
        @brainstate.transform.custom_jvp
        def square(x):
            return x ** 2

        @square.def_jvp
        def square_jvp(primals, tangents):
            (x,) = primals
            (xt,) = tangents
            return x ** 2, 2.0 * x * xt

        self.assertTrue(jnp.allclose(square(jnp.array(4.0)), 16.0))
        out, tan = jax.jvp(square, (jnp.array(4.0),), (jnp.array(1.0),))
        self.assertTrue(jnp.allclose(tan, 8.0))

    def test_error_without_rule(self):
        cf = brainstate.transform.custom_jvp(lambda x: x ** 2)
        with self.assertRaises(RuntimeError):
            jax.jvp(lambda x: cf(x), (jnp.array(3.0),), (jnp.array(1.0),))


if __name__ == '__main__':
    unittest.main()
