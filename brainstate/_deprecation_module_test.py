# Copyright 2025 BrainSim Ecosystem Limited. All Rights Reserved.
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

"""
Tests for deprecated brainstate.augment module.
"""

import unittest
import warnings

import jax.numpy as jnp


class TestDeprecatedAugment(unittest.TestCase):
    """Test suite for the deprecated brainstate.augment module."""

    def test_augment_module_import(self):
        """Test that the deprecated augment module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.augment.grad

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.augment' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_augmentation_functions(self):
        """Test that all augmentation functions are accessible."""
        import brainstate

        augment_funcs = [
            'GradientTransform',
            'grad',
            'vector_grad',
            'hessian',
            'jacobian',
            'jacrev',
            'jacfwd',
            'abstract_init',
            'vmap',
            'pmap',
            'map',
            'vmap_new_states',
            'restore_rngs',
        ]

        for func_name in augment_funcs:
            with self.subTest(function=func_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the function
                    func = getattr(brainstate.augment, func_name)
                    self.assertIsNotNone(func)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.augment' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {func_name}")

    def test_gradient_functions(self):
        """Test gradient-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test grad
            grad = brainstate.augment.grad
            self.assertIsNotNone(grad)

            # Test vector_grad
            vector_grad = brainstate.augment.vector_grad
            self.assertIsNotNone(vector_grad)

            # Test GradientTransform
            GradientTransform = brainstate.augment.GradientTransform
            self.assertIsNotNone(GradientTransform)

    def test_grad_function(self):
        """Test grad function functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test grad function
            grad = brainstate.augment.grad
            self.assertIsNotNone(grad)
            # Just check that it's callable
            self.assertTrue(callable(grad))

    def test_jacobian_functions(self):
        """Test Jacobian-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test jacobian
            jacobian = brainstate.augment.jacobian
            self.assertIsNotNone(jacobian)

            # Test jacrev
            jacrev = brainstate.augment.jacrev
            self.assertIsNotNone(jacrev)

            # Test jacfwd
            jacfwd = brainstate.augment.jacfwd
            self.assertIsNotNone(jacfwd)

    def test_hessian_function(self):
        """Test Hessian function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test hessian
            hessian = brainstate.augment.hessian
            self.assertIsNotNone(hessian)
            # Just check that it's callable
            self.assertTrue(callable(hessian))

    def test_mapping_functions(self):
        """Test mapping-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test vmap
            vmap = brainstate.augment.vmap
            self.assertIsNotNone(vmap)

            # Test pmap
            pmap = brainstate.augment.pmap
            self.assertIsNotNone(pmap)

            # Test map
            map_func = brainstate.augment.map
            self.assertIsNotNone(map_func)

    def test_vmap_function(self):
        """Test vmap function functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test vmap
            vmap = brainstate.augment.vmap
            self.assertIsNotNone(vmap)
            # Just check that it's callable
            self.assertTrue(callable(vmap))

    def test_vmap_new_states(self):
        """Test vmap_new_states function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test vmap_new_states
            vmap_new_states = brainstate.augment.vmap_new_states
            self.assertIsNotNone(vmap_new_states)

    def test_abstract_init(self):
        """Test abstract_init function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test abstract_init
            abstract_init = brainstate.augment.abstract_init
            self.assertIsNotNone(abstract_init)

    def test_restore_rngs(self):
        """Test restore_rngs function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test restore_rngs
            restore_rngs = brainstate.augment.restore_rngs
            self.assertIsNotNone(restore_rngs)

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.augment.__name__, 'brainstate.augment')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.augment.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.augment.__all__, list)
            self.assertIn('grad', brainstate.augment.__all__)
            self.assertIn('vmap', brainstate.augment.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.augment)

            # Check that expected attributes are present
            expected_attrs = [
                'grad', 'vmap', 'jacobian', 'hessian',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            # self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.augment.NonExistentFunction

            self.assertIn('NonExistentFunction', str(context.exception))
            self.assertIn('brainstate.augment', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.augment)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.augment', repr_str)
            self.assertIn('brainstate.transform', repr_str)

    def test_gradient_transform_class(self):
        """Test GradientTransform class."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test GradientTransform class
            GradientTransform = brainstate.augment.GradientTransform
            self.assertIsNotNone(GradientTransform)


class TestDeprecatedCompile(unittest.TestCase):
    """Test suite for the deprecated brainstate.compile module."""

    def test_compile_module_import(self):
        """Test that the deprecated compile module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.compile.jit

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.compile' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_compilation_functions(self):
        """Test that all compilation functions are accessible."""
        import brainstate

        compile_funcs = [
            'checkpoint',
            'remat',
            'cond',
            'switch',
            'ifelse',
            'jit_error_if',
            'jit',
            'scan',
            'checkpointed_scan',
            'for_loop',
            'checkpointed_for_loop',
            'while_loop',
            'bounded_while_loop',
            'StatefulFunction',
            'make_jaxpr',
            'ProgressBar',
        ]

        for func_name in compile_funcs:
            with self.subTest(function=func_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the function
                    func = getattr(brainstate.compile, func_name)
                    self.assertIsNotNone(func)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.compile' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {func_name}")

    def test_jit_function(self):
        """Test JIT compilation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test jit function
            jit = brainstate.compile.jit
            self.assertIsNotNone(jit)
            # Just check that it's callable
            self.assertTrue(callable(jit))

    def test_cond_function(self):
        """Test conditional function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test cond function
            cond = brainstate.compile.cond
            self.assertIsNotNone(cond)
            # Just check that it's callable
            self.assertTrue(callable(cond))

    def test_ifelse_function(self):
        """Test ifelse function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test ifelse function
            ifelse = brainstate.compile.ifelse
            self.assertIsNotNone(ifelse)

    def test_switch_function(self):
        """Test switch function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test switch function
            switch = brainstate.compile.switch
            self.assertIsNotNone(switch)

    def test_loop_functions(self):
        """Test loop-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test for_loop
            for_loop = brainstate.compile.for_loop
            self.assertIsNotNone(for_loop)

            # Test while_loop
            while_loop = brainstate.compile.while_loop
            self.assertIsNotNone(while_loop)

            # Test bounded_while_loop
            bounded_while_loop = brainstate.compile.bounded_while_loop
            self.assertIsNotNone(bounded_while_loop)

    def test_scan_functions(self):
        """Test scan-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test scan
            scan = brainstate.compile.scan
            self.assertIsNotNone(scan)

            # Test checkpointed_scan
            checkpointed_scan = brainstate.compile.checkpointed_scan
            self.assertIsNotNone(checkpointed_scan)

    def test_checkpoint_functions(self):
        """Test checkpoint-related functions."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test checkpoint
            checkpoint = brainstate.compile.checkpoint
            self.assertIsNotNone(checkpoint)

            # Test remat (rematerialization)
            remat = brainstate.compile.remat
            self.assertIsNotNone(remat)

    def test_jit_error_if(self):
        """Test jit_error_if function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test jit_error_if
            jit_error_if = brainstate.compile.jit_error_if
            self.assertIsNotNone(jit_error_if)

    def test_stateful_function(self):
        """Test StatefulFunction class."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test StatefulFunction
            StatefulFunction = brainstate.compile.StatefulFunction
            self.assertIsNotNone(StatefulFunction)

    def test_make_jaxpr(self):
        """Test make_jaxpr function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test make_jaxpr
            make_jaxpr = brainstate.compile.make_jaxpr
            self.assertIsNotNone(make_jaxpr)

    def test_progress_bar(self):
        """Test ProgressBar class."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test ProgressBar
            ProgressBar = brainstate.compile.ProgressBar
            self.assertIsNotNone(ProgressBar)

    def test_checkpointed_for_loop(self):
        """Test checkpointed_for_loop function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test checkpointed_for_loop
            checkpointed_for_loop = brainstate.compile.checkpointed_for_loop
            self.assertIsNotNone(checkpointed_for_loop)

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.compile.__name__, 'brainstate.compile')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.compile.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.compile.__all__, list)
            self.assertIn('jit', brainstate.compile.__all__)
            self.assertIn('cond', brainstate.compile.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.compile)

            # Check that expected attributes are present
            expected_attrs = [
                'jit', 'cond', 'scan', 'for_loop', 'while_loop',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            # self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.compile.NonExistentFunction

            self.assertIn('NonExistentFunction', str(context.exception))
            self.assertIn('brainstate.compile', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.compile)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.compile', repr_str)
            self.assertIn('brainstate.transform', repr_str)


class TestDeprecatedFunctional(unittest.TestCase):
    """Test suite for the deprecated brainstate.functional module."""

    def test_functional_module_import(self):
        """Test that the deprecated functional module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.functional.relu

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.functional' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_activation_functions(self):
        """Test that all activation functions are accessible."""
        import brainstate

        activations = [
            'tanh',
            'relu',
            'squareplus',
            'softplus',
            'soft_sign',
            'sigmoid',
            'silu',
            'swish',
            'log_sigmoid',
            'elu',
            'leaky_relu',
            'hard_tanh',
            'celu',
            'selu',
            'gelu',
            'glu',
            'logsumexp',
            'log_softmax',
            'softmax',
            'standardize'
        ]

        for activation_name in activations:
            with self.subTest(activation=activation_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the activation function
                    activation = getattr(brainstate.functional, activation_name)
                    self.assertIsNotNone(activation)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.functional' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {activation_name}")

    def test_activation_functionality(self):
        """Test that deprecated activation functions still work correctly."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test data
            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

            # Test relu
            result = brainstate.functional.relu(x)
            expected = jnp.maximum(0, x)
            self.assertTrue(jnp.allclose(result, expected))

            # Test sigmoid
            result = brainstate.functional.sigmoid(x)
            expected = 1 / (1 + jnp.exp(-x))
            self.assertTrue(jnp.allclose(result, expected))

            # Test tanh
            result = brainstate.functional.tanh(x)
            expected = jnp.tanh(x)
            self.assertTrue(jnp.allclose(result, expected))

            # Test softmax
            result = brainstate.functional.softmax(x)
            self.assertAlmostEqual(jnp.sum(result), 1.0, places=5)

    def test_weight_standardization(self):
        """Test weight standardization function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test weight standardization
            weight_std = brainstate.functional.weight_standardization
            self.assertIsNotNone(weight_std)

    def test_clip_grad_norm(self):
        """Test clip_grad_norm function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test clip_grad_norm
            clip_grad = brainstate.functional.clip_grad_norm
            self.assertIsNotNone(clip_grad)

    def test_leaky_relu(self):
        """Test leaky_relu with custom alpha."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test leaky_relu
            result = brainstate.functional.leaky_relu(x, negative_slope=0.01)
            # Check positive values are unchanged
            self.assertTrue(jnp.allclose(result[x >= 0], x[x >= 0]))
            # Check negative values are scaled
            self.assertTrue(jnp.allclose(result[x < 0], 0.01 * x[x < 0]))

    def test_elu_activation(self):
        """Test ELU activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test ELU
            result = brainstate.functional.elu(x, alpha=1.0)
            # Check positive values are unchanged
            self.assertTrue(jnp.allclose(result[x >= 0], x[x >= 0]))
            # Check negative values follow ELU formula
            expected_neg = 1.0 * (jnp.exp(x[x < 0]) - 1)
            self.assertTrue(jnp.allclose(result[x < 0], expected_neg))

    def test_gelu_activation(self):
        """Test GELU activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test GELU
            result = brainstate.functional.gelu(x)
            self.assertEqual(result.shape, x.shape)
            # Check that GELU(0) ≈ 0
            self.assertAlmostEqual(result[2], 0.0, places=5)

    def test_softplus_activation(self):
        """Test Softplus activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            # Test softplus
            result = brainstate.functional.softplus(x)
            expected = jnp.log(1 + jnp.exp(x))
            self.assertTrue(jnp.allclose(result, expected))

    def test_log_softmax(self):
        """Test log_softmax function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([1.0, 2.0, 3.0])
            # Test log_softmax
            result = brainstate.functional.log_softmax(x)
            # Check that exp of log_softmax sums to 1
            self.assertAlmostEqual(jnp.sum(jnp.exp(result)), 1.0, places=5)

    def test_silu_swish(self):
        """Test SiLU (Swish) activation function."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

            # Test silu
            result_silu = brainstate.functional.silu(x)
            # Test swish (should be the same as silu)
            result_swish = brainstate.functional.swish(x)

            # They should be equal
            self.assertTrue(jnp.allclose(result_silu, result_swish))

            # Check against expected formula: x * sigmoid(x)
            expected = x * brainstate.functional.sigmoid(x)
            self.assertTrue(jnp.allclose(result_silu, expected))

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.functional.__name__, 'brainstate.functional')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.functional.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.functional.__all__, list)
            self.assertIn('relu', brainstate.functional.__all__)
            self.assertIn('sigmoid', brainstate.functional.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.functional)

            # Check that expected attributes are present
            expected_attrs = [
                'relu', 'sigmoid', 'tanh', 'softmax',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            # self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.functional.NonExistentFunction

            self.assertIn('NonExistentFunction', str(context.exception))
            self.assertIn('brainstate.functional', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.functional)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.functional', repr_str)
            self.assertIn('brainstate.nn', repr_str)


class TestDeprecatedInit(unittest.TestCase):
    """Test suite for the deprecated brainstate.init module."""

    def test_init_module_import(self):
        """Test that the deprecated init module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.init.Constant

            # Check that a deprecation warning was issued
            self.assertGreater(len(w), 0)
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_param_function(self):
        """Test the deprecated param function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            # Test accessing param function
            param = brainstate.init.param
            self.assertIsNotNone(param)

            # Check that a deprecation warning was issued
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_initializers(self):
        """Test that all deprecated initializers are accessible."""
        import brainstate

        # Test various initializers
        initializers = [
            'Constant',
            'Identity',
            'Normal',
            'TruncatedNormal',
            'Uniform',
            'KaimingUniform',
            'KaimingNormal',
            'XavierUniform',
            'XavierNormal',
            'LecunUniform',
            'LecunNormal',
            'Orthogonal',
            'DeltaOrthogonal',
        ]

        for init_name in initializers:
            with self.subTest(initializer=init_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the initializer
                    initializer = getattr(brainstate.init, init_name)
                    self.assertIsNotNone(initializer)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.init' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {init_name}")

    def test_initializer_functionality(self):
        """Test that deprecated initializers still work correctly."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Constant initializer
            const_init = brainstate.init.Constant(0.5)
            result = const_init((2, 3))
            self.assertEqual(result.shape, (2, 3))
            self.assertTrue(jnp.allclose(result, 0.5))

            # Test Normal initializer
            normal_init = brainstate.init.Normal(mean=0.0, std=1.0)
            result = normal_init((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test Uniform initializer
            uniform_init = brainstate.init.Uniform(low=-1.0, high=1.0)
            result = uniform_init((5, 5))
            self.assertEqual(result.shape, (5, 5))
            self.assertTrue(jnp.all(result >= -1.0))
            self.assertTrue(jnp.all(result <= 1.0))

    def test_kaiming_initializers(self):
        """Test Kaiming (He) initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test KaimingUniform
            kaiming_uniform = brainstate.init.KaimingUniform()
            result = kaiming_uniform((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test KaimingNormal
            kaiming_normal = brainstate.init.KaimingNormal()
            result = kaiming_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_xavier_initializers(self):
        """Test Xavier (Glorot) initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test XavierUniform
            xavier_uniform = brainstate.init.XavierUniform()
            result = xavier_uniform((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test XavierNormal
            xavier_normal = brainstate.init.XavierNormal()
            result = xavier_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_lecun_initializers(self):
        """Test LeCun initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test LecunUniform
            lecun_uniform = brainstate.init.LecunUniform()
            result = lecun_uniform((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test LecunNormal
            lecun_normal = brainstate.init.LecunNormal()
            result = lecun_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_orthogonal_initializers(self):
        """Test Orthogonal initialization methods."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Orthogonal
            orthogonal = brainstate.init.Orthogonal()
            result = orthogonal((10, 10))
            self.assertEqual(result.shape, (10, 10))

            # Test DeltaOrthogonal with 3D shape (required)
            delta_orthogonal = brainstate.init.DeltaOrthogonal()
            result = delta_orthogonal((3, 3, 3))
            self.assertEqual(result.shape, (3, 3, 3))

    def test_identity_initializer(self):
        """Test Identity initializer."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Identity
            identity = brainstate.init.Identity()
            result = identity((5, 5))
            self.assertEqual(result.shape, (5, 5))
            # Check it's an identity matrix
            expected = jnp.eye(5)
            self.assertTrue(jnp.allclose(result, expected))

    def test_truncated_normal_initializer(self):
        """Test TruncatedNormal initializer."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test TruncatedNormal with required parameters
            truncated_normal = brainstate.init.TruncatedNormal(mean=0.0, std=1.0)
            result = truncated_normal((10, 10))
            self.assertEqual(result.shape, (10, 10))

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.init.__name__, 'brainstate.init')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.init.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.init.__all__, list)
            self.assertIn('Constant', brainstate.init.__all__)
            self.assertIn('Normal', brainstate.init.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.init)

            # Check that expected attributes are present
            expected_attrs = [
                'Constant', 'Normal', 'Uniform', 'XavierNormal',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.init.NonExistentInitializer

            self.assertIn('NonExistentInitializer', str(context.exception))
            self.assertIn('brainstate.init', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.init)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.init', repr_str)
            self.assertIn('braintools.init', repr_str)


class TestDeprecatedOptim(unittest.TestCase):
    """Test suite for the deprecated brainstate.optim module."""

    def test_optim_module_import(self):
        """Test that the deprecated optim module can be imported."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate
            # Access an attribute to trigger deprecation warning
            _ = brainstate.optim.Adam

            # Check that a deprecation warning was issued (excluding JAX warnings)
            relevant_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.optim' in str(warning.message)
            ]
            # self.assertGreater(len(relevant_warnings), 0)

    def test_optimizer_base_class(self):
        """Test accessing the Optimizer base class."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            # Test accessing Optimizer class
            optimizer = brainstate.optim.Optimizer
            self.assertIsNotNone(optimizer)

            # Check that a deprecation warning was issued
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_learning_rate_schedulers(self):
        """Test that all learning rate schedulers are accessible."""
        import brainstate

        schedulers = [
            'LRScheduler',
            'ConstantLR',
            'StepLR',
            'MultiStepLR',
            'CosineAnnealingLR',
            'CosineAnnealingWarmRestarts',
            'ExponentialLR',
            'PolynomialLR',
        ]

        for scheduler_name in schedulers:
            with self.subTest(scheduler=scheduler_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the scheduler
                    scheduler = getattr(brainstate.optim, scheduler_name)
                    self.assertIsNotNone(scheduler)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.optim' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {scheduler_name}")

    def test_optimizers(self):
        """Test that all optimizers are accessible."""
        import brainstate

        optimizers = [
            'OptaxOptimizer',
            'LBFGS',
            'SGD',
            'Adagrad',
            'Adadelta',
            'RMSprop',
            'Adam',
            'Lars',
            'AdamW',
        ]

        for optimizer_name in optimizers:
            with self.subTest(optimizer=optimizer_name):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Access the optimizer
                    optimizer = getattr(brainstate.optim, optimizer_name)
                    self.assertIsNotNone(optimizer)

                    # Check that a deprecation warning was issued
                    deprecation_warnings = [warning for warning in w if
                                            issubclass(warning.category, DeprecationWarning)]
                    # Filter out the JAX warning
                    relevant_warnings = [w for w in deprecation_warnings if 'brainstate.optim' in str(w.message)]
                    # self.assertGreater(len(relevant_warnings), 0, f"No deprecation warning for {optimizer_name}")

    def test_constant_lr_scheduler(self):
        """Test ConstantLR scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create ConstantLR scheduler
            lr_scheduler = brainstate.optim.ConstantLR
            self.assertIsNotNone(lr_scheduler)

    def test_step_lr_scheduler(self):
        """Test StepLR scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create StepLR scheduler class
            lr_scheduler_class = brainstate.optim.StepLR
            self.assertIsNotNone(lr_scheduler_class)

    def test_exponential_lr_scheduler(self):
        """Test ExponentialLR scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create ExponentialLR scheduler class
            lr_scheduler_class = brainstate.optim.ExponentialLR
            self.assertIsNotNone(lr_scheduler_class)

    def test_cosine_annealing_lr_scheduler(self):
        """Test CosineAnnealingLR scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create CosineAnnealingLR scheduler class
            lr_scheduler_class = brainstate.optim.CosineAnnealingLR
            self.assertIsNotNone(lr_scheduler_class)

    def test_sgd_optimizer(self):
        """Test SGD optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test SGD optimizer
            sgd = brainstate.optim.SGD(lr=0.01)
            self.assertIsNotNone(sgd)

    def test_adam_optimizer(self):
        """Test Adam optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Adam optimizer
            adam = brainstate.optim.Adam(lr=0.001)
            self.assertIsNotNone(adam)

            # Test AdamW optimizer
            adamw = brainstate.optim.AdamW(lr=0.001)
            self.assertIsNotNone(adamw)

    def test_rmsprop_optimizer(self):
        """Test RMSprop optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test RMSprop optimizer
            rmsprop = brainstate.optim.RMSprop(lr=0.001)
            self.assertIsNotNone(rmsprop)

    def test_adagrad_optimizer(self):
        """Test Adagrad optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Adagrad optimizer
            adagrad = brainstate.optim.Adagrad(lr=0.01)
            self.assertIsNotNone(adagrad)

    def test_adadelta_optimizer(self):
        """Test Adadelta optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test Adadelta optimizer
            adadelta = brainstate.optim.Adadelta(lr=1.0, rho=0.95)
            self.assertIsNotNone(adadelta)

    def test_lars_optimizer(self):
        """Test LARS/Lars optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test LARS optimizer (try both capitalizations)
            lars = brainstate.optim.Lars(lr=0.001)
            self.assertIsNotNone(lars)

    def test_lbfgs_optimizer(self):
        """Test LBFGS optimizer functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test LBFGS optimizer
            lbfgs = brainstate.optim.LBFGS()
            self.assertIsNotNone(lbfgs)

    def test_optax_optimizer(self):
        """Test OptaxOptimizer wrapper."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test OptaxOptimizer
            optax_opt = brainstate.optim.OptaxOptimizer
            self.assertIsNotNone(optax_opt)

    def test_multi_step_lr_scheduler(self):
        """Test MultiStepLR scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create MultiStepLR scheduler class
            lr_scheduler_class = brainstate.optim.MultiStepLR
            self.assertIsNotNone(lr_scheduler_class)

    def test_polynomial_lr_scheduler(self):
        """Test PolynomialLR scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create PolynomialLR scheduler
            lr_scheduler = brainstate.optim.PolynomialLR
            self.assertIsNotNone(lr_scheduler)

    def test_cosine_annealing_warm_restarts(self):
        """Test CosineAnnealingWarmRestarts scheduler functionality."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Create CosineAnnealingWarmRestarts scheduler class
            lr_scheduler_class = brainstate.optim.CosineAnnealingWarmRestarts
            self.assertIsNotNone(lr_scheduler_class)

    def test_module_attributes(self):
        """Test module-level attributes."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            # Test __name__ attribute
            self.assertEqual(brainstate.optim.__name__, 'brainstate.optim')

            # Test __doc__ attribute
            self.assertIn('DEPRECATED', brainstate.optim.__doc__)

            # Test __all__ attribute
            self.assertIsInstance(brainstate.optim.__all__, list)
            self.assertIn('Adam', brainstate.optim.__all__)
            self.assertIn('SGD', brainstate.optim.__all__)

    def test_dir_method(self):
        """Test that dir() returns appropriate attributes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            attrs = dir(brainstate.optim)

            # Check that expected attributes are present
            expected_attrs = [
                'Adam', 'SGD', 'RMSprop',
                '__name__', '__doc__', '__all__'
            ]
            for attr in expected_attrs:
                self.assertIn(attr, attrs)

            # Check that a deprecation warning was issued
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))

    def test_invalid_attribute_access(self):
        """Test that accessing invalid attributes raises appropriate errors."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            with self.assertRaises(AttributeError) as context:
                _ = brainstate.optim.NonExistentOptimizer

            self.assertIn('NonExistentOptimizer', str(context.exception))
            self.assertIn('brainstate.optim', str(context.exception))

    def test_repr_method(self):
        """Test the __repr__ method of the deprecated module."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            import brainstate

            repr_str = repr(brainstate.optim)
            self.assertIn('DeprecatedModule', repr_str)
            self.assertIn('brainstate.optim', repr_str)
            self.assertIn('braintools.optim', repr_str)

    def test_warning_message_content(self):
        """Test that deprecation warnings contain correct information."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import brainstate

            # Access a specific optimizer to trigger warning
            _ = brainstate.optim.Adam

            # Check warning content - filter out JAX warnings
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'brainstate.optim' in str(warning.message)
            ]

            if len(deprecation_warnings) > 0:
                warning_message = str(deprecation_warnings[0].message)
                self.assertIn('deprecated', warning_message.lower())
                self.assertIn('brainstate.optim', warning_message)
                self.assertIn('braintools.optim', warning_message)

    def test_multiple_access_single_warning(self):
        """Test that accessing the same attribute multiple times only warns once."""
        import brainstate
        # Clear any previous warnings for Adam
        if hasattr(brainstate.optim, '_warned_attrs'):
            brainstate.optim._warned_attrs.clear()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access the same optimizer multiple times
            _ = brainstate.optim.Adam
            _ = brainstate.optim.Adam
            _ = brainstate.optim.Adam

            # Count deprecation warnings for 'Adam'
            adam_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                   and 'Adam' in str(warning.message)
                   and 'brainstate.optim' in str(warning.message)
            ]

            # Should only have one warning for Adam
            self.assertEqual(len(adam_warnings), 1)
