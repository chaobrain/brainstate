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


import unittest

import brainunit as u
import jax.numpy as jnp
import numpy as np

import brainstate


class TestExpEulerODE(unittest.TestCase):
    """Test cases for ODE integration using exp_euler_step."""

    def test_exponential_decay(self):
        """Test simple exponential decay: dx/dt = -x"""
        def drift(x, t):
            return -x

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            # Expected: x(t+dt) ≈ x(t) * exp(-dt) ≈ 0.99004983
            expected = np.exp(-0.01)
            np.testing.assert_allclose(x1, expected)

    def test_exponential_decay_with_time_constant(self):
        """Test exponential decay with time constant: dx/dt = -x/tau"""
        def drift(x, tau):
            return -x / tau

        with brainstate.environ.context(dt=1.0 * u.ms):
            x0 = u.math.asarray(1.0 * u.mV)
            tau = 10.0 * u.ms
            x1 = brainstate.nn.exp_euler_step(drift, x0, tau)

            # Expected: x(t+dt) ≈ x(t) * exp(-dt/tau)
            expected = np.exp(-0.1) * u.mV
            assert u.math.allclose(x1, expected)

    def test_linear_growth(self):
        """Test linear growth: dx/dt = a"""
        def drift(x):
            return jnp.asarray([2.0])

        with brainstate.environ.context(dt=0.1):
            x0 = jnp.asarray([1.0])
            x1 = brainstate.nn.exp_euler_step(drift, x0)

            # For constant derivative, result should be x0 + a*dt
            expected = 1.0 + 2.0 * 0.1
            np.testing.assert_allclose(x1, expected)

    def test_multidimensional_system(self):
        """Test multi-dimensional ODE: harmonic oscillator"""
        def drift(x, t):
            # dx/dt = [x1, -x0] (circular motion)
            return jnp.array([x[1], -x[0]])

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array([1.0, 0.0])
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            # Check that energy is approximately conserved
            energy0 = np.sum(x0 ** 2)
            energy1 = np.sum(x1 ** 2)
            np.testing.assert_allclose(energy1, energy0, rtol=0.1)

    def test_stiff_equation(self):
        """Test stiff equation where exponential Euler should be stable"""
        def drift(x, t):
            # Stiff equation: dx/dt = -100*x
            return -100.0 * x

        with brainstate.environ.context(dt=0.1):
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            # Should remain stable and decay
            expected = np.exp(-10.0)
            np.testing.assert_allclose(x1, expected, rtol=1e-2)
            self.assertGreater(x1, 0.0)  # Should not become negative


class TestExpEulerSDE(unittest.TestCase):
    """Test cases for SDE integration using exp_euler_step."""

    def test_simple_sde_with_constant_diffusion(self):
        """Test SDE with constant diffusion: dx = -x*dt + sigma*dW"""
        def drift(x, t):
            return -x

        def diffusion(x, t):
            return jnp.array(0.1)

        with brainstate.environ.context(dt=0.01):
            brainstate.random.seed(42)
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, diffusion, x0, None)

            # Result should have both drift and diffusion components
            # Cannot test exact value due to randomness, but check it's reasonable
            self.assertIsInstance(x1, (jnp.ndarray, float))

    def test_ornstein_uhlenbeck_process(self):
        """Test Ornstein-Uhlenbeck process: dx = -theta*x*dt + sigma*dW"""
        theta = 0.5
        sigma = 0.3

        def drift(x, t):
            return -theta * x

        def diffusion(x, t):
            return jnp.full_like(x, sigma)

        with brainstate.environ.context(dt=0.01):
            brainstate.random.seed(123)
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, diffusion, x0, None)

            # Mean should decrease (drift dominates initially)
            # Run multiple steps and check statistics
            x = x0
            results = []
            for _ in range(100):
                x = brainstate.nn.exp_euler_step(drift, diffusion, x, None)
                results.append(x)

            # Mean should converge toward 0
            final_mean = np.mean(results[-10:])
            self.assertLess(abs(final_mean), 0.5)

    def test_sde_multidimensional(self):
        """Test multi-dimensional SDE"""
        def drift(x, t):
            return -0.5 * x

        def diffusion(x, t):
            return jnp.array([0.1, 0.2])

        with brainstate.environ.context(dt=0.01):
            brainstate.random.seed(456)
            x0 = jnp.array([1.0, 1.0])
            x1 = brainstate.nn.exp_euler_step(drift, diffusion, x0, None)

            self.assertEqual(x1.shape, (2,))

    def test_state_dependent_diffusion(self):
        """Test SDE with state-dependent diffusion: dx = -x*dt + sqrt(x)*dW"""
        def drift(x, t):
            return -0.1 * x

        def diffusion(x, t):
            return jnp.sqrt(jnp.abs(x) + 1e-8)

        with brainstate.environ.context(dt=0.01):
            brainstate.random.seed(789)
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, diffusion, x0, None)

            self.assertIsInstance(x1, (jnp.ndarray, float))


class TestExpEulerUnits(unittest.TestCase):
    """Test cases for unit handling in exp_euler_step."""

    def test_unit_compatibility_drift(self):
        """Test that drift function units are validated correctly"""
        def drift(x, tau):
            return -x / tau

        with brainstate.environ.context(dt=1.0 * u.ms):
            x0 = 1.0 * u.mV
            tau = 10.0 * u.ms
            x1 = brainstate.nn.exp_euler_step(drift, x0, tau)

            # Result should have same units as input
            self.assertEqual(u.get_unit(x1), u.get_unit(x0))

    def test_unit_mismatch_raises_error(self):
        """Test that incompatible diffusion units raise an error"""
        def drift(x, t):
            return -x / (10.0 * u.ms)

        def diffusion(x, t):
            # Wrong units: should be mV/sqrt(ms) but returning mV
            return 0.1 * u.mV

        with brainstate.environ.context(dt=1.0 * u.ms):
            x0 = 1.0 * u.mV
            with self.assertRaises(ValueError):
                brainstate.nn.exp_euler_step(drift, diffusion, x0, None)

    def test_correct_diffusion_units(self):
        """Test SDE with correct diffusion units"""
        def drift(x, tau):
            return -x / tau

        def diffusion(x, t):
            # Correct units: mV/sqrt(ms)
            return 0.1 * u.mV / u.ms ** 0.5

        with brainstate.environ.context(dt=1.0 * u.ms):
            brainstate.random.seed(42)
            x0 = 1.0 * u.mV
            tau = 10.0 * u.ms
            x1 = brainstate.nn.exp_euler_step(drift, diffusion, x0, tau)

            self.assertEqual(u.get_unit(x1), u.get_unit(x0))

    def test_dimensionless_with_time_units(self):
        """Test dimensionless state with time units in dt"""
        def drift(x, t):
            return -2.0 * x / u.second

        with brainstate.environ.context(dt=0.1 * u.second):
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            expected = np.exp(-0.2)
            np.testing.assert_allclose(x1, expected, rtol=1e-5)


class TestExpEulerInputValidation(unittest.TestCase):
    """Test cases for input validation in exp_euler_step."""

    def test_non_callable_drift_raises_error(self):
        """Test that non-callable drift raises AssertionError"""
        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(1.0)
            with self.assertRaises(AssertionError):
                brainstate.nn.exp_euler_step("not a function", x0, None)

    def test_no_state_variable_raises_error(self):
        """Test that missing state variable raises AssertionError"""
        def drift(x, t):
            return -x

        with brainstate.environ.context(dt=0.01):
            with self.assertRaises(AssertionError):
                brainstate.nn.exp_euler_step(drift)

    def test_invalid_dtype_raises_error(self):
        """Test that invalid dtype raises ValueError"""
        def drift(x, t):
            return -x

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(1, dtype=jnp.int32)
            with self.assertRaises(ValueError):
                brainstate.nn.exp_euler_step(drift, x0, None)

    def test_float16_dtype_accepted(self):
        """Test that float16 dtype is accepted"""
        def drift(x, t):
            return -x

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(1.0, dtype=jnp.float16)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)
            self.assertEqual(x1.dtype, jnp.float16)

    def test_bfloat16_dtype_accepted(self):
        """Test that bfloat16 dtype is accepted"""
        def drift(x, t):
            return -x

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(1.0, dtype=jnp.bfloat16)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)
            self.assertEqual(x1.dtype, jnp.bfloat16)

    def test_diffusion_without_state_raises_error(self):
        """Test that diffusion function without state variable raises error"""
        def drift(x, t):
            return -x

        def diffusion(x, t):
            return 0.1

        with brainstate.environ.context(dt=0.01):
            with self.assertRaises(AssertionError):
                brainstate.nn.exp_euler_step(drift, diffusion)


class TestExpEulerEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_zero_initial_condition(self):
        """Test with zero initial condition"""
        def drift(x, t):
            return -x + 1.0

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(0.0)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            # Should move toward equilibrium at x=1
            self.assertGreater(x1, 0.0)

    def test_very_small_timestep(self):
        """Test with very small timestep"""
        def drift(x, t):
            return -x

        with brainstate.environ.context(dt=1e-8):
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            # Should barely change
            np.testing.assert_allclose(x1, x0)

    def test_large_timestep_stability(self):
        """Test stability with large timestep (advantage of exponential Euler)"""
        def drift(x, t):
            return -10.0 * x

        with brainstate.environ.context(dt=1.0):
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(drift, x0, None)

            # Should remain stable (not blow up or oscillate)
            expected = np.exp(-10.0)
            np.testing.assert_allclose(x1, expected, rtol=1e-1)
            self.assertGreater(x1, 0.0)

    def test_kwargs_passed_correctly(self):
        """Test that kwargs are passed to drift and diffusion functions"""
        def drift(x, scale=1., **kwargs):
            return -scale * x

        def diffusion(x, noise_level=0.1, **kwargs):
            return noise_level

        with brainstate.environ.context(dt=0.01):
            brainstate.random.seed(42)
            x0 = jnp.array(1.0)
            x1 = brainstate.nn.exp_euler_step(
                drift, diffusion, x0,
                scale=2.0, noise_level=0.2
            )

            self.assertIsInstance(x1, (jnp.ndarray, float))

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same random seed"""
        def drift(x):
            return -0.5 * x

        def diffusion(x):
            return 0.1

        with brainstate.environ.context(dt=0.01):
            x0 = jnp.array(1.0)

            brainstate.random.seed(42)
            x1_first = brainstate.nn.exp_euler_step(drift, diffusion, x0)

            brainstate.random.seed(42)
            x1_second = brainstate.nn.exp_euler_step(drift, diffusion, x0)

            np.testing.assert_array_equal(x1_first, x1_second)
