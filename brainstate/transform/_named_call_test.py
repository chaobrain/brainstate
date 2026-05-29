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


def _name_stacks(jaxpr):
    """Collect the stringified name stacks of every equation in a jaxpr."""
    out = set()
    for eqn in jaxpr.eqns:
        si = getattr(eqn, 'source_info', None)
        ns = getattr(si, 'name_stack', None)
        if ns is not None:
            out.add(str(ns))
    return out


class TestExports(unittest.TestCase):
    def test_symbol_exists(self):
        self.assertTrue(callable(brainstate.transform.named_call))


class TestNamedCall(unittest.TestCase):
    def test_decorator_with_name(self):
        @brainstate.transform.named_call(name='myblock')
        def block(x):
            return jnp.sin(x) * 2.0

        x = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(block(x), jnp.sin(x) * 2.0))
        names = _name_stacks(jax.make_jaxpr(block)(x))
        self.assertTrue(any('myblock' in n for n in names))

    def test_bare_decorator_uses_fn_name(self):
        @brainstate.transform.named_call
        def my_layer(x):
            return jnp.cos(x)

        x = jnp.array([0.0, 1.0])
        self.assertTrue(jnp.allclose(my_layer(x), jnp.cos(x)))
        names = _name_stacks(jax.make_jaxpr(my_layer)(x))
        self.assertTrue(any('my_layer' in n for n in names))

    def test_direct_call_form(self):
        def f(x):
            return x + 1.0

        inc = brainstate.transform.named_call(f, name='inc')
        self.assertTrue(jnp.allclose(inc(jnp.array(1.0)), 2.0))
        names = _name_stacks(jax.make_jaxpr(inc)(jnp.array(1.0)))
        self.assertTrue(any('inc' in n for n in names))

    def test_state_transparency(self):
        counter = brainstate.State(jnp.array(0.0))

        @brainstate.transform.named_call(name='step')
        def step(x):
            counter.value = counter.value + 1.0
            return x * 2.0

        out = step(jnp.array(3.0))
        self.assertTrue(jnp.allclose(out, 6.0))
        self.assertTrue(jnp.allclose(counter.value, 1.0))  # state write preserved

    def test_composes_under_grad(self):
        @brainstate.transform.named_call(name='loss')
        def loss(x):
            return jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        g = jax.grad(loss)(x)
        self.assertTrue(jnp.allclose(g, 2.0 * x))

    def test_composes_under_jit(self):
        @brainstate.transform.named_call(name='scaled')
        def scaled(x):
            return x * 10.0

        x = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(jax.jit(scaled)(x), x * 10.0))


if __name__ == '__main__':
    unittest.main()
