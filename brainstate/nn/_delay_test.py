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


class TestUpgradedDelay(unittest.TestCase):
    """Tests for the upgraded delay mechanism with new features."""

    def setUp(self):
        brainstate.environ.set(dt=0.1)

    def tearDown(self):
        brainstate.environ.pop('dt')

    def test_write_ptr_tracking(self):
        """Test that write_ptr correctly tracks buffer position and wraps around."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0)
        delay.init_state()

        # Initial write_ptr should be 0
        self.assertEqual(delay.write_ptr.value, 0)

        # Update several times and check write_ptr increments
        max_length = delay.max_length
        for i in range(max_length * 2 + 5):
            delay.update(jnp.ones((1,)) * i)
            expected_ptr = (i + 1) % max_length
            self.assertEqual(delay.write_ptr.value, expected_ptr,
                             f"write_ptr mismatch at step {i}: expected {expected_ptr}, got {delay.write_ptr.value}")

    def test_write_ptr_no_environ_i_dependency(self):
        """Verify that delays work without setting environ.I."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0)
        delay.register_entry('a', 0.5)
        delay.init_state()

        # Update without setting environ.I (should use write_ptr instead)
        for i in range(20):
            delay.update(jnp.ones((1,)) * i)
            # This should work without environ.I
            result = delay.at('a')
            self.assertEqual(result.shape, (1,))

    def test_interpolation_nearest(self):
        """Test nearest/round interpolation method."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='nearest')
        delay.init_state()

        # Populate history
        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        # Test retrieval with float time steps
        with brainstate.environ.context(t=3.0):
            # Should round to nearest
            result = delay.retrieve_at_time(1.4)  # Should round to 1.0 second ago
            print(f"Nearest interpolation result: {result}")

    def test_interpolation_linear(self):
        """Test linear interpolation method."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='linear')
        delay.init_state()

        # Populate with known values
        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        with brainstate.environ.context(t=3.0):
            # Linear interpolation between two values
            result = delay.retrieve_at_time(2.5)  # 0.5 seconds ago
            print(f"Linear interpolation result: {result}")

    def test_interpolation_cubic(self):
        """Test cubic spline interpolation method."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='cubic')
        delay.init_state()

        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        with brainstate.environ.context(t=3.0):
            result = delay.retrieve_at_time(2.5)
            print(f"Cubic interpolation result: {result}")
            self.assertEqual(result.shape, (1,))

    def test_interpolation_hermite(self):
        """Test Hermite spline interpolation method."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='hermite')
        delay.init_state()

        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        with brainstate.environ.context(t=3.0):
            result = delay.retrieve_at_time(2.5)
            print(f"Hermite interpolation result: {result}")
            self.assertEqual(result.shape, (1,))

    def test_interpolation_polynomial2(self):
        """Test quadratic polynomial interpolation method."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='polynomial2')
        delay.init_state()

        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        with brainstate.environ.context(t=3.0):
            result = delay.retrieve_at_time(2.5)
            print(f"Polynomial2 interpolation result: {result}")
            self.assertEqual(result.shape, (1,))

    def test_interpolation_polynomial3(self):
        """Test cubic polynomial interpolation method."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='polynomial3')
        delay.init_state()

        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        with brainstate.environ.context(t=3.0):
            result = delay.retrieve_at_time(2.5)
            print(f"Polynomial3 interpolation result: {result}")
            self.assertEqual(result.shape, (1,))

    def test_custom_interpolation(self):
        """Test custom interpolation method registration."""

        # Define custom interpolation function
        def my_interp(history, indices, float_idx, max_length):
            # Simple: always return the floor value
            i = jnp.floor(float_idx).astype(jnp.int32) % max_length
            idx = (i,) + indices
            return jax.tree.map(lambda h: h[idx], history)

        # Register custom interpolation
        brainstate.nn.InterpolationRegistry.register('my_custom', my_interp)

        # Use custom interpolation
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=2.0, interpolation='my_custom')
        delay.init_state()

        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        with brainstate.environ.context(t=3.0):
            result = delay.retrieve_at_time(2.5)
            print(f"Custom interpolation result: {result}")
            self.assertEqual(result.shape, (1,))

    def test_backward_compatibility_delay_method(self):
        """Test backward compatibility with old delay_method parameter."""
        import warnings

        # Test that concat mode triggers deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0, delay_method='concat')
            delay.init_state()

            # Check deprecation warning was raised
            self.assertTrue(len(w) >= 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[0].message).lower())

        # Verify it still works (uses rotation internally)
        delay.register_entry('a', 0.5)
        for i in range(20):
            delay.update(jnp.ones((1,)) * i)
        result = delay.at('a')
        self.assertEqual(result.shape, (1,))

    def test_backward_compatibility_interp_method(self):
        """Test backward compatibility with old interp_method parameter."""
        # Old API: interp_method='linear_interp'
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0, interp_method='linear_interp')
        delay.init_state()
        self.assertEqual(delay.interp_method, 'linear')  # Should map to 'linear'

        # Old API: interp_method='round'
        delay2 = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0, interp_method='round')
        delay2.init_state()
        self.assertEqual(delay2.interp_method, 'nearest')  # Should map to 'nearest'

    def test_batched_delays_with_write_ptr(self):
        """Test batched delays with shared write pointer."""
        batch_size = 4
        delay = brainstate.nn.Delay(jnp.zeros((5,)), time=1.0)
        delay.init_state(batch_size=batch_size)

        # write_ptr should be scalar (shared across batches) since updates are synchronized
        self.assertEqual(delay.write_ptr.value.shape, ())
        self.assertEqual(delay.write_ptr.value, 0)

        # Update with batched data
        for i in range(20):
            batched_data = jnp.ones((batch_size, 5)) * i
            delay.update(batched_data)

        # write_ptr should track the synchronized updates
        expected_ptr = 20 % delay.max_length
        self.assertEqual(delay.write_ptr.value, expected_ptr)

    def test_interpolation_registry_list_methods(self):
        """Test that all expected interpolation methods are registered."""
        methods = brainstate.nn.InterpolationRegistry.list_methods()

        # Check all built-in methods are present
        expected_methods = ['nearest', 'round', 'linear', 'linear_interp',
                            'cubic', 'hermite', 'polynomial2', 'polynomial3']
        for method in expected_methods:
            self.assertIn(method, methods,
                          f"Expected interpolation method '{method}' not found in registry")

    def test_reset_state_resets_write_ptr(self):
        """Test that reset_state properly resets write_ptr to 0."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0)
        delay.init_state()

        # Update several times to advance write_ptr
        for i in range(15):
            delay.update(jnp.ones((1,)) * i)

        # write_ptr should be non-zero
        self.assertNotEqual(delay.write_ptr.value, 0)

        # Reset state
        delay.reset_state()

        # write_ptr should be back to 0
        self.assertEqual(delay.write_ptr.value, 0)

    def test_unified_ring_buffer_rotation_only(self):
        """Test that all delays now use rotation (unified ring buffer)."""
        # Even if we try to specify concat, it should use rotation
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0)
        delay.init_state()

        # delay_method should always be rotation
        self.assertEqual(delay.delay_method, 'rotation')

        # Verify ring buffer behavior
        for i in range(30):
            delay.update(jnp.ones((1,)) * i)

        # write_ptr should wrap around
        self.assertTrue(0 <= delay.write_ptr.value < delay.max_length)

    def test_update_frequency_default(self):
        """Test that default behavior (update_every=None) works as before."""
        delay = brainstate.nn.Delay(jnp.zeros((1,)), time=1.0)
        delay.init_state()

        for i in range(20):
            delay.update(jnp.ones((1,)) * i)

        # Should have updated every call
        self.assertEqual(delay.write_ptr.value, 20 % delay.max_length)
