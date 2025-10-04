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

"""
Comprehensive tests for brainstate.typing module.

This test suite validates all type annotations, protocols, and type utilities
provided by the typing module, ensuring they work correctly with JAX, NumPy,
and BrainUnit integration.
"""

import unittest
from typing import get_type_hints, Union, Any

import brainunit as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from brainstate.typing import (
    # Key and path types
    Key, PathParts, FilterLiteral, Filter,

    # Array types
    Array, ArrayLike, Shape, Size, Axes, DType, DTypeLike, SupportsDType,

    # PyTree types
    PyTree,

    # Random types
    SeedOrKey,

    # Utility types
    Missing,

    # Type variables
    K, _T, _Annotation,

    # Internal utilities for testing
    _item_to_str, _maybe_tuple_to_str, _Array
)


class TestKeyProtocol(unittest.TestCase):
    """Test the Key protocol and related path types."""

    def setUp(self):
        """Set up test fixtures."""
        self.string_key = "layer1"
        self.int_key = 42
        self.float_key = 3.14

    def test_key_protocol_string(self):
        """Test that strings implement the Key protocol."""
        self.assertIsInstance(self.string_key, Key)
        self.assertTrue(hasattr(self.string_key, '__hash__'))
        self.assertTrue(hasattr(self.string_key, '__lt__'))

    def test_key_protocol_int(self):
        """Test that integers implement the Key protocol."""
        self.assertIsInstance(self.int_key, Key)
        self.assertTrue(hasattr(self.int_key, '__hash__'))
        self.assertTrue(hasattr(self.int_key, '__lt__'))

    def test_key_ordering(self):
        """Test that keys can be ordered."""
        self.assertTrue("a" < "b")
        self.assertTrue(1 < 2)
        self.assertTrue(1.0 < 2.0)

    def test_custom_key_class(self):
        """Test custom class implementing Key protocol."""

        class CustomKey:
            def __init__(self, name: str):
                self.name = name

            def __hash__(self) -> int:
                return hash(self.name)

            def __eq__(self, other) -> bool:
                return isinstance(other, CustomKey) and self.name == other.name

            def __lt__(self, other) -> bool:
                return isinstance(other, CustomKey) and self.name < other.name

        key1 = CustomKey("first")
        key2 = CustomKey("second")

        self.assertIsInstance(key1, Key)
        self.assertTrue(key1 < key2)
        self.assertEqual(hash(key1), hash(CustomKey("first")))

    def test_path_parts(self):
        """Test PathParts type usage."""
        # Simple path
        path1: PathParts = ("model", "layers", 0, "weights")
        self.assertEqual(len(path1), 4)
        self.assertIsInstance(path1[0], str)
        self.assertIsInstance(path1[2], int)

        # Empty path
        path2: PathParts = ()
        self.assertEqual(len(path2), 0)

        # Mixed types path
        path3: PathParts = ("root", 1, "sub", 2.5)
        self.assertEqual(len(path3), 4)

    def test_predicate_functions(self):
        """Test Predicate function type."""

        def is_weight_matrix(path: PathParts, value: Any) -> bool:
            return len(path) > 0 and "weight" in str(path[-1]) and hasattr(value, 'ndim') and value.ndim == 2

        def is_bias_vector(path: PathParts, value: Any) -> bool:
            return len(path) > 0 and "bias" in str(path[-1]) and hasattr(value, 'ndim') and value.ndim == 1

        # Test with mock data
        weight_path: PathParts = ("layer", "weight")
        bias_path: PathParts = ("layer", "bias")

        weight_matrix = np.random.randn(10, 5)
        bias_vector = np.random.randn(5)

        self.assertTrue(is_weight_matrix(weight_path, weight_matrix))
        self.assertFalse(is_weight_matrix(bias_path, bias_vector))
        self.assertTrue(is_bias_vector(bias_path, bias_vector))
        self.assertFalse(is_bias_vector(weight_path, weight_matrix))

    def test_filter_types(self):
        """Test various filter type combinations."""
        # FilterLiteral types
        type_filter: FilterLiteral = float
        string_filter: FilterLiteral = "weight"
        predicate_filter: FilterLiteral = lambda path, x: hasattr(x, 'ndim')
        bool_filter: FilterLiteral = True
        ellipsis_filter: FilterLiteral = ...
        none_filter: FilterLiteral = None

        # Combined filters
        tuple_filter: Filter = (float, "weight")
        list_filter: Filter = [int, float, "bias"]
        nested_filter: Filter = [
            ("weight", lambda p, x: hasattr(x, 'ndim') and x.ndim == 2),
            ("bias", lambda p, x: hasattr(x, 'ndim') and x.ndim == 1),
        ]

        # Verify types are correctly assigned
        self.assertIsInstance(type_filter, type)
        self.assertIsInstance(string_filter, str)
        self.assertTrue(callable(predicate_filter))
        self.assertIsInstance(bool_filter, bool)
        self.assertEqual(ellipsis_filter, ...)
        self.assertIsNone(none_filter)


class TestArrayAnnotations(unittest.TestCase):
    """Test Array type annotations and related utilities."""

    def test_array_basic_annotation(self):
        """Test basic Array type annotation."""

        def process_array(x: Array) -> Array:
            return x * 2

        # Check that function can be called with various array types
        jax_array = jnp.array([1, 2, 3])
        numpy_array = np.array([1, 2, 3])

        result1 = process_array(jax_array)
        result2 = process_array(numpy_array)

        self.assertIsInstance(result1, jax.Array)
        self.assertIsInstance(result2, np.ndarray)

    def test_array_shape_annotation(self):
        """Test Array with shape annotations."""

        def matrix_multiply(a: Array["m, n"], b: Array["n, k"]) -> Array["m, k"]:
            return a @ b

        # Test with compatible shapes
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        a = jax.random.normal(key1, (3, 4))
        b = jax.random.normal(key2, (4, 5))
        result = matrix_multiply(a, b)

        self.assertEqual(result.shape, (3, 5))

    def test_array_class_getitem(self):
        """Test Array.__class_getitem__ functionality."""
        # Test shape annotation creation
        shaped_array = Array["batch, features"]
        self.assertIsNotNone(shaped_array)
        self.assertTrue(hasattr(shaped_array, '__origin__'))

        # Test complex shape annotation
        complex_array = Array["batch, seq_len, d_model"]
        self.assertIsNotNone(complex_array)

        # Test with ellipsis
        flexible_array = Array["batch, ..."]
        self.assertIsNotNone(flexible_array)

    def test_item_to_str_function(self):
        """Test _item_to_str utility function."""
        # String item
        self.assertEqual(_item_to_str("batch"), "'batch'")

        # Type item
        self.assertEqual(_item_to_str(float), "float")

        # Ellipsis item
        self.assertEqual(_item_to_str(...), "...")

        # Slice item
        slice_item = slice("start", "stop")
        expected = "'start': 'stop'"
        self.assertEqual(_item_to_str(slice_item), expected)

        # Slice with step should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            _item_to_str(slice("start", "stop", "step"))

    def test_maybe_tuple_to_str_function(self):
        """Test _maybe_tuple_to_str utility function."""
        # Single item
        self.assertEqual(_maybe_tuple_to_str("single"), "'single'")

        # Empty tuple
        self.assertEqual(_maybe_tuple_to_str(()), "()")

        # Non-empty tuple
        tuple_item = ("batch", "features")
        expected = "'batch', 'features'"
        self.assertEqual(_maybe_tuple_to_str(tuple_item), expected)

    def test_array_module_setting(self):
        """Test that Array has correct module for display."""
        self.assertEqual(Array.__module__, "builtins")


class TestShapeAndSizeTypes(unittest.TestCase):
    """Test shape, size, and axes type annotations."""

    def test_size_type_variants(self):
        """Test different Size type variants."""
        # Single integer
        size1: Size = 10
        self.assertIsInstance(size1, int)

        # Tuple of integers
        size2: Size = (3, 4, 5)
        self.assertIsInstance(size2, tuple)
        self.assertTrue(all(isinstance(x, int) for x in size2))

        # NumPy integers
        size3: Size = np.int32(8)
        self.assertIsInstance(size3, np.integer)

        # Sequence with mixed NumPy types
        size4: Size = [np.int64(2), 3, np.int32(4)]
        self.assertIsInstance(size4, list)

    def test_shape_type(self):
        """Test Shape type usage."""
        # 2D shape
        matrix_shape: Shape = (10, 20)
        self.assertEqual(len(matrix_shape), 2)
        self.assertTrue(all(isinstance(x, int) for x in matrix_shape))

        # 3D shape
        tensor_shape: Shape = (5, 10, 15)
        self.assertEqual(len(tensor_shape), 3)

        # 1D shape (still a sequence)
        vector_shape: Shape = (100,)
        self.assertEqual(len(vector_shape), 1)

    def test_axes_type(self):
        """Test Axes type variants."""
        # Single axis
        axis1: Axes = 0
        self.assertIsInstance(axis1, int)

        # Multiple axes
        axis2: Axes = (0, 2)
        self.assertIsInstance(axis2, tuple)
        self.assertTrue(all(isinstance(x, int) for x in axis2))

        # List of axes
        axis3: Axes = [1, 3, 4]
        self.assertIsInstance(axis3, list)

    def test_shape_operations(self):
        """Test operations using shape types."""

        def create_zeros(shape: Shape) -> jax.Array:
            return jnp.zeros(shape)

        def sum_along_axes(array: ArrayLike, axes: Axes) -> jax.Array:
            return jnp.sum(array, axis=axes)

        # Test shape creation
        arr = create_zeros((3, 4))
        self.assertEqual(arr.shape, (3, 4))

        # Test axes operations
        test_array = jnp.ones((2, 3, 4))
        result1 = sum_along_axes(test_array, 0)
        self.assertEqual(result1.shape, (3, 4))

        result2 = sum_along_axes(test_array, (0, 2))
        self.assertEqual(result2.shape, (3,))


class TestArrayLikeAndDType(unittest.TestCase):
    """Test ArrayLike and dtype-related types."""

    def test_arraylike_variants(self):
        """Test different ArrayLike type variants."""

        def process_data(data: ArrayLike) -> jax.Array:
            return jnp.asarray(data)

        # JAX array
        jax_array = jnp.array([1, 2, 3])
        result1 = process_data(jax_array)
        self.assertIsInstance(result1, jax.Array)

        # NumPy array
        numpy_array = np.array([1, 2, 3])
        result2 = process_data(numpy_array)
        self.assertIsInstance(result2, jax.Array)

        # Python scalars
        result3 = process_data(42)
        self.assertIsInstance(result3, jax.Array)
        self.assertEqual(result3.shape, ())

        result4 = process_data(3.14)
        self.assertIsInstance(result4, jax.Array)

        result5 = process_data(True)
        self.assertIsInstance(result5, jax.Array)

        result6 = process_data(1 + 2j)
        self.assertIsInstance(result6, jax.Array)

        # NumPy scalars
        result7 = process_data(np.float32(2.5))
        self.assertIsInstance(result7, jax.Array)

        result8 = process_data(np.bool_(False))
        self.assertIsInstance(result8, jax.Array)

        # BrainUnit quantities (if available)
        try:
            quantity = 1.5 * u.second
            # Convert to plain array for processing
            result9 = process_data(quantity.mantissa)
            self.assertIsInstance(result9, jax.Array)
        except (AttributeError, TypeError):
            # Skip if BrainUnit quantities not properly set up
            pass

    def test_dtype_variants(self):
        """Test DType and DTypeLike variants."""

        def cast_array(array: ArrayLike, dtype: DTypeLike) -> jax.Array:
            return jnp.asarray(array, dtype=dtype)

        test_data = [1, 2, 3]

        # String dtype
        result1 = cast_array(test_data, 'float32')
        self.assertEqual(result1.dtype, jnp.float32)

        # NumPy type
        result2 = cast_array(test_data, np.float32)
        self.assertEqual(result2.dtype, jnp.float32)

        # Python type
        result3 = cast_array(test_data, float)
        self.assertTrue(jnp.issubdtype(result3.dtype, jnp.floating))

        # NumPy dtype object
        result4 = cast_array(test_data, np.dtype('int32'))
        self.assertEqual(result4.dtype, jnp.int32)

    def test_supports_dtype_protocol(self):
        """Test SupportsDType protocol."""

        def get_dtype(obj: SupportsDType) -> DType:
            return obj.dtype

        # Test with arrays
        arr = jnp.array([1.0, 2.0])
        dtype = get_dtype(arr)
        self.assertIsInstance(dtype, np.dtype)

        # Test with NumPy arrays
        np_arr = np.array([1, 2], dtype=np.int64)
        dtype2 = get_dtype(np_arr)
        self.assertEqual(dtype2, np.int64)

    def test_dtype_alias(self):
        """Test DType alias."""

        def create_array(shape: Shape, dtype: DType) -> jax.Array:
            return jnp.zeros(shape, dtype=dtype)

        arr = create_array((3, 4), np.float32)
        self.assertEqual(arr.shape, (3, 4))
        self.assertEqual(arr.dtype, jnp.float32)


class TestPyTreeTypes(unittest.TestCase):
    """Test PyTree type annotations."""

    def test_pytree_basic_usage(self):
        """Test basic PyTree type usage."""

        def tree_function(tree: PyTree[float]) -> PyTree[float]:
            return jax.tree_util.tree_map(lambda x: x * 2, tree)

        # Test with different PyTree structures
        tree1 = {"a": 1.0, "b": 2.0}
        result1 = tree_function(tree1)
        self.assertAlmostEqual(result1["a"], 2.0)
        self.assertAlmostEqual(result1["b"], 4.0)

        tree2 = [1.0, 2.0, 3.0]
        result2 = tree_function(tree2)
        expected = [2.0, 4.0, 6.0]
        for i, (actual, expect) in enumerate(zip(result2, expected)):
            self.assertAlmostEqual(actual, expect)

    def test_pytree_with_structure(self):
        """Test PyTree with structure annotations."""

        def structured_function(tree: PyTree[float, "T"]) -> PyTree[float, "T"]:
            return jax.tree_util.tree_map(lambda x: x + 1, tree)

        # Test that function works with various structures
        tree = {"weights": 1.0, "bias": 2.0}
        result = structured_function(tree)
        self.assertAlmostEqual(result["weights"], 2.0)
        self.assertAlmostEqual(result["bias"], 3.0)

    def test_pytree_instantiation_error(self):
        """Test that PyTree cannot be instantiated."""
        with self.assertRaises(RuntimeError):
            PyTree()

    def test_pytree_subscripting(self):
        """Test PyTree subscripting behavior."""
        # Single type parameter
        pytree_type = PyTree[float]
        self.assertTrue(hasattr(pytree_type, 'leaftype'))
        self.assertEqual(pytree_type.leaftype, float)

        # Type and structure parameters
        pytree_structured = PyTree[int, "T"]
        self.assertTrue(hasattr(pytree_structured, 'leaftype'))
        self.assertTrue(hasattr(pytree_structured, 'structure'))
        self.assertEqual(pytree_structured.leaftype, int)
        self.assertEqual(pytree_structured.structure, "T")

    def test_pytree_structure_validation(self):
        """Test PyTree structure validation."""
        # Valid structure names
        valid_structures = ["T", "S T", "... T", "T ...", "foo bar"]
        for structure in valid_structures:
            PyTree[float, structure]

        # Invalid structures
        with self.assertRaises(ValueError):
            PyTree[float, ""]  # Empty string

        with self.assertRaises(ValueError):
            PyTree[float, "invalid-identifier"]  # Invalid identifier

        with self.assertRaises(ValueError):
            PyTree[float, "123abc"]  # Starts with number

    def test_pytree_tuple_length_validation(self):
        """Test PyTree tuple parameter validation."""
        # Valid 2-tuple
        PyTree[float, "T"]

        # Invalid tuple lengths
        with self.assertRaises(ValueError):
            PyTree[float, "T", "extra"]  # 3-tuple

        with self.assertRaises(ValueError):
            PyTree[float,]  # 1-tuple with trailing comma would be (float,)


class TestRandomTypes(unittest.TestCase):
    """Test random number generation types."""

    def test_seed_or_key_variants(self):
        """Test SeedOrKey type variants."""

        def generate_random(key: SeedOrKey, shape: Shape) -> jax.Array:
            if isinstance(key, int):
                key = jr.PRNGKey(key)
            return jr.normal(key, shape)

        # Integer seed
        result1 = generate_random(42, (3, 4))
        self.assertEqual(result1.shape, (3, 4))

        # JAX PRNG key
        jax_key = jr.PRNGKey(123)
        result2 = generate_random(jax_key, (5,))
        self.assertEqual(result2.shape, (5,))

        # NumPy array key
        np_key = np.array([1, 2], dtype=np.uint32)
        result3 = generate_random(np_key, (2, 2))
        self.assertEqual(result3.shape, (2, 2))

    def test_reproducibility_with_seeds(self):
        """Test that same seeds produce same results."""

        def generate_data(seed: SeedOrKey) -> jax.Array:
            if isinstance(seed, int):
                key = jr.PRNGKey(seed)
            else:
                key = seed
            return jr.normal(key, (5,))

        # Same integer seeds
        result1 = generate_data(42)
        result2 = generate_data(42)
        np.testing.assert_array_equal(result1, result2)

        # Same JAX keys
        key = jr.PRNGKey(999)
        result3 = generate_data(key)
        result4 = generate_data(key)
        np.testing.assert_array_equal(result3, result4)


class TestUtilityTypes(unittest.TestCase):
    """Test utility types and edge cases."""

    def test_missing_sentinel(self):
        """Test Missing sentinel class."""
        _MISSING = Missing()

        def function_with_optional_param(value: Union[int, None, Missing] = _MISSING):
            if value is _MISSING:
                return "no_value"
            elif value is None:
                return "explicit_none"
            else:
                return f"value_{value}"

        # Test different call patterns
        self.assertEqual(function_with_optional_param(), "no_value")
        self.assertEqual(function_with_optional_param(None), "explicit_none")
        self.assertEqual(function_with_optional_param(42), "value_42")

        # Test that different Missing instances are distinct objects
        missing1 = Missing()
        missing2 = Missing()
        self.assertIsNot(missing1, missing2)  # Different instances
        # Note: Missing doesn't define __eq__, so != comparison uses identity

    def test_type_variables(self):
        """Test type variables are properly defined."""
        # Test that type variables exist
        self.assertIsNotNone(K)
        self.assertIsNotNone(_T)
        self.assertIsNotNone(_Annotation)

        # Test that they are TypeVar instances
        from typing import TypeVar
        self.assertIsInstance(K, TypeVar)
        self.assertIsInstance(_T, TypeVar)
        self.assertIsInstance(_Annotation, TypeVar)

    def test_internal_array_type(self):
        """Test internal _Array type."""
        # Test that _Array exists and has proper module
        self.assertIsNotNone(_Array)
        self.assertEqual(_Array.__module__, "builtins")

        # Test that it can be parameterized
        parameterized = _Array[str]
        self.assertIsNotNone(parameterized)


class TestRealWorldUsagePattterns(unittest.TestCase):
    """Test real-world usage patterns and integration."""

    def test_neural_network_typing(self):
        """Test typing patterns common in neural networks."""

        def linear_layer(
            x: Array["batch, in_features"],
            weight: Array["out_features, in_features"],
            bias: Array["out_features"]
        ) -> Array["batch, out_features"]:
            return x @ weight.T + bias

        # Test with actual arrays
        batch_size, in_features, out_features = 32, 128, 64
        key = jr.PRNGKey(42)
        key1, key2, key3 = jr.split(key, 3)
        x = jr.normal(key1, (batch_size, in_features))
        weight = jr.normal(key2, (out_features, in_features))
        bias = jr.normal(key3, (out_features,))

        result = linear_layer(x, weight, bias)
        self.assertEqual(result.shape, (batch_size, out_features))

    def test_pytree_parameter_filtering(self):
        """Test PyTree filtering patterns."""

        def extract_weights(params: PyTree[ArrayLike]) -> PyTree[ArrayLike]:
            # Mock filtering - in real code this would use jax.tree_util
            return jax.tree_util.tree_map(lambda x: x, params)

        # Test with parameter structure
        params = {
            "layer1": {"weight": jnp.ones((10, 5)), "bias": jnp.zeros(10)},
            "layer2": {"weight": jnp.ones((5, 3)), "bias": jnp.zeros(5)}
        }

        result = extract_weights(params)
        self.assertIsInstance(result, dict)
        self.assertIn("layer1", result)
        self.assertIn("layer2", result)

    def test_mixed_type_operations(self):
        """Test operations mixing different typed inputs."""

        def process_mixed_data(
            arrays: ArrayLike,
            shape: Shape,
            dtype: DTypeLike,
            seed: SeedOrKey
        ) -> jax.Array:
            # Convert inputs
            data = jnp.asarray(arrays, dtype=dtype)
            key = jr.PRNGKey(seed) if isinstance(seed, int) else seed

            # Generate noise and add to data
            noise = jr.normal(key, shape) * 0.1
            return data.reshape(shape) + noise

        # Test with mixed inputs
        result = process_mixed_data(
            arrays=[1, 2, 3, 4],
            shape=(2, 2),
            dtype='float32',
            seed=42
        )

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, jnp.float32)

    def test_scientific_computing_pattern(self):
        """Test scientific computing usage patterns."""

        def numerical_integration(
            func: callable,
            bounds: ArrayLike,
            n_points: Size,
            dtype: DTypeLike = jnp.float32  # Use float32 for JAX compatibility
        ) -> jax.Array:
            # Mock numerical integration
            x = jnp.linspace(bounds[0], bounds[1], n_points, dtype=dtype)
            y = jax.vmap(func)(x)
            dx = (bounds[1] - bounds[0]) / n_points
            return jnp.sum(y) * dx

        # Test with simple bounds (skip units for simplicity)
        bounds_array = jnp.array([0.0, 1.0])

        result = numerical_integration(
            lambda t: t ** 2,
            bounds_array,
            1000,
            jnp.float32
        )

        self.assertIsInstance(result, jax.Array)
        self.assertEqual(result.dtype, jnp.float32)


class TestTypeHintCompatibility(unittest.TestCase):
    """Test compatibility with Python's typing system."""

    def test_get_type_hints(self):
        """Test that type hints can be retrieved from annotated functions."""

        def annotated_function(
            arr: ArrayLike,
            shape: Shape,
            dtype: DTypeLike
        ) -> jax.Array:
            return jnp.zeros(shape, dtype=dtype)

        hints = get_type_hints(annotated_function)

        # Check that hints are captured
        self.assertIn('arr', hints)
        self.assertIn('shape', hints)
        self.assertIn('dtype', hints)
        self.assertIn('return', hints)

    def test_isinstance_checks(self):
        """Test isinstance checks with protocol types."""
        # Test Key protocol
        self.assertIsInstance("string", Key)
        self.assertIsInstance(42, Key)
        self.assertIsInstance(3.14, Key)

        # Test SupportsDType protocol (check for dtype attribute)
        arr = jnp.array([1, 2, 3])
        self.assertTrue(hasattr(arr, 'dtype'))

        np_arr = np.array([1, 2, 3])
        self.assertTrue(hasattr(np_arr, 'dtype'))

        # Test that objects without dtype don't have the attribute
        self.assertFalse(hasattr("string", 'dtype'))

    def test_module_imports(self):
        """Test that all types can be imported correctly."""
        from brainstate.typing import (
            Key, PathParts, Predicate, Filter, Array, ArrayLike,
            Shape, Size, Axes, PyTree, SeedOrKey, DType, DTypeLike,
            Missing
        )

        # Verify all imports succeeded
        types_to_check = [
            Key, PathParts, Predicate, Filter, Array, ArrayLike,
            Shape, Size, Axes, PyTree, SeedOrKey, DType, DTypeLike,
            Missing
        ]

        for type_obj in types_to_check:
            self.assertIsNotNone(type_obj)

    def test_documentation_strings(self):
        """Test that types have proper documentation."""
        documented_types = [
            Key, Array, PyTree, Missing, SupportsDType
        ]

        for type_obj in documented_types:
            self.assertIsNotNone(type_obj.__doc__)
            self.assertGreater(len(type_obj.__doc__), 10)  # Has substantial documentation


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
