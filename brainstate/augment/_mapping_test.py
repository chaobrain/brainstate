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

from __future__ import annotations

import unittest

import jax.core
import jax.numpy as jnp

import brainstate as bst


class TestVmap(unittest.TestCase):
    def test_vmap_return_keep_reference_return(self):
        @bst.augment.vmap(in_axes=0, out_axes=0)
        def create_model(key):
            bst.random.set_key(key)
            m1 = bst.nn.Linear(2, 3)

            m2 = bst.nn.Linear(3, 4)
            m2.a = m1
            m3 = bst.nn.Linear(3, 5)
            m3.a = m1
            self.assertTrue(id(m2.a) == id(m3.a))
            return m2, m3

        m2, m3 = create_model(bst.random.split_key(10))
        self.assertTrue(id(m2.a) == id(m3.a))
        jax.core.concrete_or_error(None, bst.random.DEFAULT.value)

    def test_vmap_return_keep_reference_pass_into_fun(self):
        @bst.augment.vmap(in_axes=(None, None, 0), out_axes=0)
        def run_model(m2, m3, x):
            self.assertTrue(id(m2.a) == id(m3.a))
            self.assertTrue(id(m2) != m2_id)
            self.assertTrue(id(m3) != m3_id)
            return m2(x), m3(x)

        m1 = bst.nn.Linear(2, 3)
        m2 = bst.nn.Linear(4, 3)
        m2.a = m1
        m3 = bst.nn.Linear(4, 5)
        m3.a = m1
        m3_id = id(m3)
        m2_id = id(m2)
        r1, r2 = run_model(m2, m3, jnp.ones((4, 3, 4)))

    def test_vmap_set_key(self):
        @bst.augment.vmap(in_axes=0, out_axes=0)
        def create_model(key):
            bst.random.set_key(key)
            return bst.nn.Linear(2, 3)

        model = create_model(bst.random.split_keys(10))
        print(model.weight.value_call(jnp.shape))
        model.weight.value_call(lambda x: jax.core.concrete_or_error(None, x))
        bst.random.seed()

    def test_vmap_input(self):
        model = bst.nn.Linear(2, 3)
        print(id(model), id(model.weight))
        model_id = id(model)
        weight_id = id(model.weight)

        x = jnp.ones((5, 2))

        @bst.augment.vmap
        def forward(x):
            self.assertTrue(id(model) == model_id)
            self.assertTrue(id(model.weight) == weight_id)
            return model(x)

        y = forward(x)
        self.assertTrue(y.shape == (5, 3))
        print(y.shape)
        print(model.weight.value_call(jnp.shape))
        print(model.weight.value)

    def test_vmap_model(self):
        model = bst.nn.Linear(2, 3)
        model_id = id(model)
        weight_id = id(model.weight)
        print(id(model), id(model.weight))
        x = jnp.ones((5, 2))

        @bst.augment.vmap(in_axes=(None, 0), out_axes=0)
        def forward(model, x):
            self.assertTrue(id(model) != model_id)
            self.assertTrue(id(model.weight) != weight_id)
            print(id(model), id(model.weight))
            return model(x)

        y = forward(model, x)
        print(y.shape)
        print(model.weight.value_call(jnp.shape))
        print(model.weight.value)

    def test_vmap1(self):
        model = bst.nn.Linear(2, 3)
        x = jnp.ones((5, 2))

        @bst.augment.vmap(in_axes=(None, 0), out_axes=0)
        def forward(model, x):
            return model(x)

        y = forward(model, x)
        print(y.shape)

    def test_vmap2(self):
        class LinearEnsemble(bst.nn.Module):
            def __init__(self, num):
                super().__init__()
                self.w = bst.ParamState(bst.random.random((num, 2, 3)))

        model = LinearEnsemble(5)
        x = jnp.ones((2,))

        @bst.augment.vmap(in_axes=(0, None), out_axes=0)
        def forward(model, x):
            return jnp.dot(x, model.w.value)

        y = forward(model, x)
        print(y.shape)

    def test_vmap3(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

        state_axes = bst.augment.StateAxes({bst.ParamState: 0, bst.ShortTermState: None})

        @bst.augment.vmap(in_axes=(state_axes,), out_axes=0)
        def mul(foo):
            return foo.a.value * foo.b.value

        foo = Foo()
        y = mul(foo)
        print(y.shape)

    def test_vmap4(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        @bst.augment.vmap
        def mul(foo):
            foo()
            return foo

        foo = Foo()
        with bst.StateTraceStack() as trace:
            m = mul(foo)

        self.assertTrue(m is foo)
        print(m.a.value, foo.a.value)
        self.assertTrue(jnp.allclose(m.a.value, foo.a.value))
        print(m.b.value, foo.b.value)
        self.assertTrue(jnp.allclose(m.b.value, foo.b.value))
        print(trace.get_write_states())
        self.assertTrue(len(trace.get_write_states()) == 1)
        print(trace.get_read_states())
        self.assertTrue(len(trace.get_read_states()) == 2)

