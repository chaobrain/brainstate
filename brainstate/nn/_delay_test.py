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

import jax.numpy as jnp

import brainstate


class TestDelay(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1)

    def tearDown(self):
        brainstate.environ.pop('dt')

    def test_delay1(self):
        a = brainstate.State(brainstate.random.random(10, 20))
        delay = brainstate.nn.Delay(a.value)
        delay.register_entry('a', 1.)
        delay.register_entry('b', 2.)
        delay.register_entry('c', None)

        delay.init_state()
        with self.assertRaises(KeyError):
            delay.register_entry('c', 10.)

    def test_rotation_delay(self):
        rotation_delay = brainstate.nn.Delay(jnp.ones((1,)))
        t0 = 0.
        t1, n1 = 1., 10
        t2, n2 = 2., 20

        rotation_delay.register_entry('a', t0)
        rotation_delay.register_entry('b', t1)
        rotation_delay.register_entry('c2', 1.9)
        rotation_delay.register_entry('c', t2)

        rotation_delay.init_state()

        print()
        # print(rotation_delay)
        # print(rotation_delay.max_length)

        for i in range(100):
            brainstate.environ.set(i=i)
            rotation_delay.update(jnp.ones((1,)) * i)
            # print(i, rotation_delay.at('a'), rotation_delay.at('b'), rotation_delay.at('c2'), rotation_delay.at('c'))
            self.assertTrue(jnp.allclose(rotation_delay.at('a'), jnp.ones((1,)) * i))
            self.assertTrue(jnp.allclose(rotation_delay.at('b'), jnp.maximum(jnp.ones((1,)) * i - n1, 0.)))
            self.assertTrue(jnp.allclose(rotation_delay.at('c'), jnp.maximum(jnp.ones((1,)) * i - n2, 0.)))

    def test_concat_delay(self):
        with brainstate.environ.context(dt=0.1) as env:
            rotation_delay = brainstate.nn.Delay(jnp.ones([1]), delay_method='concat')
            t0 = 0.
            t1, n1 = 1., 10
            t2, n2 = 2., 20

            rotation_delay.register_entry('a', t0)
            rotation_delay.register_entry('b', t1)
            rotation_delay.register_entry('c', t2)

            rotation_delay.init_state()

            print()
            for i in range(100):
                brainstate.environ.set(i=i)
                rotation_delay.update(jnp.ones((1,)) * i)
                print(i, rotation_delay.at('a'), rotation_delay.at('b'), rotation_delay.at('c'))
                self.assertTrue(jnp.allclose(rotation_delay.at('a'), jnp.ones((1,)) * i))
                self.assertTrue(jnp.allclose(rotation_delay.at('b'), jnp.maximum(jnp.ones((1,)) * i - n1, 0.)))
                self.assertTrue(jnp.allclose(rotation_delay.at('c'), jnp.maximum(jnp.ones((1,)) * i - n2, 0.)))
            # brainstate.util.clear_buffer_memory()

    def test_jit_erro(self):
        rotation_delay = brainstate.nn.Delay(jnp.ones([1]), time=2., delay_method='concat', interp_method='round')
        rotation_delay.init_state()

        with brainstate.environ.context(i=0, t=0, jit_error_check=True):
            rotation_delay.retrieve_at_time(-2.0)
            with self.assertRaises(Exception):
                rotation_delay.retrieve_at_time(-2.1)
            rotation_delay.retrieve_at_time(-2.01)
            with self.assertRaises(Exception):
                rotation_delay.retrieve_at_time(-2.09)
            with self.assertRaises(Exception):
                rotation_delay.retrieve_at_time(0.1)
            with self.assertRaises(Exception):
                rotation_delay.retrieve_at_time(0.01)

    def test_round_interp(self):
        for shape in [(1,), (1, 1), (1, 1, 1)]:
            for delay_method in ['rotation', 'concat']:
                rotation_delay = brainstate.nn.Delay(jnp.ones(shape), time=2., delay_method=delay_method,
                                                     interp_method='round')
                t0, n1 = 0.01, 0
                t1, n1 = 1.04, 10
                t2, n2 = 1.06, 11
                rotation_delay.init_state()

                @brainstate.transform.jit
                def retrieve(td, i):
                    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                        return rotation_delay.retrieve_at_time(td)

                print()
                for i in range(100):
                    t = i * brainstate.environ.get_dt()
                    with brainstate.environ.context(i=i, t=t):
                        rotation_delay.update(jnp.ones(shape) * i)
                        print(i,
                              retrieve(t - t0, i),
                              retrieve(t - t1, i),
                              retrieve(t - t2, i))
                        self.assertTrue(jnp.allclose(retrieve(t - t0, i), jnp.ones(shape) * i))
                        self.assertTrue(jnp.allclose(retrieve(t - t1, i), jnp.maximum(jnp.ones(shape) * i - n1, 0.)))
                        self.assertTrue(jnp.allclose(retrieve(t - t2, i), jnp.maximum(jnp.ones(shape) * i - n2, 0.)))

    def test_linear_interp(self):
        for shape in [(1,), (1, 1), (1, 1, 1)]:
            for delay_method in ['rotation', 'concat']:
                print(shape, delay_method)

                rotation_delay = brainstate.nn.Delay(jnp.ones(shape), time=2., delay_method=delay_method,
                                                     interp_method='linear_interp')
                t0, n0 = 0.01, 0.1
                t1, n1 = 1.04, 10.4
                t2, n2 = 1.06, 10.6
                rotation_delay.init_state()

                @brainstate.transform.jit
                def retrieve(td, i):
                    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                        return rotation_delay.retrieve_at_time(td)

                print()
                for i in range(100):
                    t = i * brainstate.environ.get_dt()
                    with brainstate.environ.context(i=i, t=t):
                        rotation_delay.update(jnp.ones(shape) * i)
                        print(i,
                              retrieve(t - t0, i),
                              retrieve(t - t1, i),
                              retrieve(t - t2, i))
                        self.assertTrue(jnp.allclose(retrieve(t - t0, i), jnp.maximum(jnp.ones(shape) * i - n0, 0.)))
                        self.assertTrue(jnp.allclose(retrieve(t - t1, i), jnp.maximum(jnp.ones(shape) * i - n1, 0.)))
                        self.assertTrue(jnp.allclose(retrieve(t - t2, i), jnp.maximum(jnp.ones(shape) * i - n2, 0.)))

    def test_rotation_and_concat_delay(self):
        rotation_delay = brainstate.nn.Delay(jnp.ones((1,)))
        concat_delay = brainstate.nn.Delay(jnp.ones([1]), delay_method='concat')
        t0 = 0.
        t1, n1 = 1., 10
        t2, n2 = 2., 20

        rotation_delay.register_entry('a', t0)
        rotation_delay.register_entry('b', t1)
        rotation_delay.register_entry('c', t2)
        concat_delay.register_entry('a', t0)
        concat_delay.register_entry('b', t1)
        concat_delay.register_entry('c', t2)

        rotation_delay.init_state()
        concat_delay.init_state()

        print()
        for i in range(100):
            brainstate.environ.set(i=i)
            new = jnp.ones((1,)) * i
            rotation_delay.update(new)
            concat_delay.update(new)
            self.assertTrue(jnp.allclose(rotation_delay.at('a'), concat_delay.at('a'), ))
            self.assertTrue(jnp.allclose(rotation_delay.at('b'), concat_delay.at('b'), ))
            self.assertTrue(jnp.allclose(rotation_delay.at('c'), concat_delay.at('c'), ))

    def test_delay_2d(self):
        with brainstate.environ.context(dt=0.1, i=0):
            rotation_delay = brainstate.nn.Delay(jnp.arange(2))
            index = (brainstate.random.uniform(0., 10., (2, 2)),
                     brainstate.random.randint(0, 2, (2, 2)))
            rotation_delay.register_entry('a', *index)
            rotation_delay.init_state()
            data = rotation_delay.at('a')
            print(index[0])
            print(index[1])
            print(data)
            assert data.shape == (2, 2)

    def test_delay_time2(self):
        with brainstate.environ.context(dt=0.1, i=0):
            rotation_delay = brainstate.nn.Delay(jnp.arange(2))
            index = (brainstate.random.uniform(0., 10., (2, 2)),
                     1)
            rotation_delay.register_entry('a', *index)
            rotation_delay.init_state()
            data = rotation_delay.at('a')
            print(index[0])
            print(index[1])
            print(data)
            assert data.shape == (2, 2)

    def test_delay_time3(self):
        with brainstate.environ.context(dt=0.1, i=0):
            rotation_delay = brainstate.nn.Delay(jnp.zeros((2, 2)))
            index = (brainstate.random.uniform(0., 10., (2, 2)),
                     1,
                     brainstate.random.randint(0, 2, (2, 2)))
            rotation_delay.register_entry('a', *index)
            rotation_delay.init_state()
            data = rotation_delay.at('a')
            print(index[0])
            print(index[1])
            print(data)
            assert data.shape == (2, 2)

    def test_delay_time4(self):
        with brainstate.environ.context(dt=0.1, i=0):
            rotation_delay = brainstate.nn.Delay(jnp.zeros((2, 2)))
            index = (brainstate.random.uniform(0., 10., (2, 2)),
                     1,
                     brainstate.random.randint(0, 2, (2, 2)))
            rotation_delay.register_entry('a', *index)
            rotation_delay.init_state()
            data = rotation_delay.at('a')
            print(index[0])
            print(index[1])
            print(data)
            assert data.shape == (2, 2)
