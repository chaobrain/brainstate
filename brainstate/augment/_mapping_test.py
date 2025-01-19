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

import jax.numpy as jnp

import brainstate as bst
from brainstate.augment._mapping import BatchAxisError


class TestVmap(unittest.TestCase):
    def test_vmap_1(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.State(bst.random.randn(5))
                self.b = bst.State(bst.random.randn(5))

            def __call__(self, *args, **kwargs):
                return self.a.value * self.b.value

        model = Model()
        r1 = model.a.value * model.b.value
        r2 = bst.augment.vmap(model, in_states=model.states())()
        self.assertTrue(jnp.allclose(r1, r2))

    def test_vmap_2(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))

            def __call__(self, *args, **kwargs):
                self.c.value = self.a.value * self.b.value
                return self.c.value + 1.

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = bst.augment.vmap(model, in_states=model.states(bst.ShortTermState))()

        model = Model()
        r2 = bst.augment.vmap(model, in_states=model.states(bst.ShortTermState), out_states=model.c)()

    def test_vmap_3(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.State(bst.random.randn(5))
                self.b = bst.State(bst.random.randn(5))

            def __call__(self, *args, **kwargs):
                return self.a.value * self.b.value

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = bst.augment.vmap(model, in_states=model.states(), out_states={1: model.states()})()

    def test_vmap_with_random(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))

            def __call__(self, key):
                bst.random.set_key(key)
                self.c.value = self.a.value * self.b.value
                return self.c.value + bst.random.randn(1)

        model = Model()
        r2 = bst.augment.vmap(
            model,
            in_states=model.states(bst.ShortTermState),
            out_states=model.c
        )(
            bst.random.split_key(5)
        )
        print(bst.random.DEFAULT)

    def test_vmap_with_random_2(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))
                self.rng = bst.random.RandomState(1)

            def __call__(self, key):
                self.rng.set_key(key)
                self.c.value = self.a.value * self.b.value
                return self.c.value + bst.random.randn(1)

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = bst.augment.vmap(
                model,
                in_states=model.states(bst.ShortTermState),
                out_states=model.c
            )(
                bst.random.split_key(5)
            )

        model = Model()
        r2 = bst.augment.vmap(
            model,
            in_states=model.states(bst.ShortTermState),
            out_states=model.c,
            rngs=model.rng,
        )(
            bst.random.split_key(5)
        )

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
            self.assertTrue(id(model) == model_id)
            self.assertTrue(id(model.weight) == weight_id)
            print(id(model), id(model.weight))
            return model(x)

        y = forward(model, x)
        print(y.shape)
        print(model.weight.value_call(jnp.shape))
        print(model.weight.value)

    def test_vmap_jit(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        foo = Foo()

        @bst.augment.vmap(in_states=foo.states())
        def mul():
            foo()

        @bst.compile.jit
        def mul_jit(inp):
            mul()
            foo.a.value += inp

        with bst.StateTraceStack() as trace:
            mul_jit(1.)

        print(foo.a.value)
        print(foo.b.value)
        self.assertTrue(jnp.allclose(foo.a.value, jnp.arange(4) + 1.))
        self.assertTrue(jnp.allclose(foo.b.value, jnp.arange(4) * jnp.arange(4)))

        write_state_ids = [id(st) for st in trace.get_write_states()]
        read_state_ids = [id(st) for st in trace.get_read_states()]

        assert id(foo.a) in write_state_ids
        assert id(foo.b) in write_state_ids

        print(trace.get_write_states())
        print(trace.get_read_states())

    def test_vmap_jit_2(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        foo = Foo()

        @bst.augment.vmap(in_states=foo.states())
        def mul():
            foo()

        @bst.compile.jit
        def mul_jit(inp):
            mul()
            foo.b.value += inp

        with bst.StateTraceStack() as trace:
            mul_jit(1.)

        print(foo.a.value)
        print(foo.b.value)
        self.assertTrue(jnp.allclose(foo.a.value, jnp.arange(4)))
        self.assertTrue(jnp.allclose(foo.b.value, jnp.arange(4) * jnp.arange(4) + 1.))

        write_state_ids = [id(st) for st in trace.get_write_states()]
        read_state_ids = [id(st) for st in trace.get_read_states()]

        assert id(foo.a) in read_state_ids
        assert id(foo.b) in write_state_ids

        print(trace.get_write_states())
        print(trace.get_read_states())


class TestMap(unittest.TestCase):
    def test_map(self):
        for dim in [(10,), (10, 10), (10, 10, 10)]:
            x = bst.random.rand(*dim)
            r1 = bst.augment.map(lambda a: a + 1, x, batch_size=None)
            r2 = bst.augment.map(lambda a: a + 1, x, batch_size=2)
            r3 = bst.augment.map(lambda a: a + 1, x, batch_size=4)
            r4 = bst.augment.map(lambda a: a + 1, x, batch_size=5)
            true_r = x + 1

            self.assertTrue(jnp.allclose(r1, true_r))
            self.assertTrue(jnp.allclose(r2, true_r))
            self.assertTrue(jnp.allclose(r3, true_r))
            self.assertTrue(jnp.allclose(r4, true_r))
