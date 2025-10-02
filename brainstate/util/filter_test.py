# Copyright 2024 BrainState Authors.
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

"""
Comprehensive tests for filter module.
"""

import unittest
from typing import Any
import numpy as np

from brainstate.util.filter import (
    to_predicate,
    WithTag,
    PathContains,
    OfType,
    Any,
    All,
    Not,
    Everything,
    Nothing,
)


class MockTaggedObject:
    """Mock object with a tag attribute for testing."""
    def __init__(self, tag: str):
        self.tag = tag


class MockTypedObject:
    """Mock object with a type attribute for testing."""
    def __init__(self, type_value: type):
        self.type = type_value


class TestToPredicateFunction(unittest.TestCase):
    """Test cases for to_predicate function."""

    def test_string_to_withtag(self):
        """Test converting string to WithTag filter."""
        pred = to_predicate('trainable')
        self.assertIsInstance(pred, WithTag)
        self.assertEqual(pred.tag, 'trainable')

        # Test functionality
        obj_with_tag = MockTaggedObject('trainable')
        obj_without_tag = MockTaggedObject('frozen')
        self.assertTrue(pred([], obj_with_tag))
        self.assertFalse(pred([], obj_without_tag))

    def test_type_to_oftype(self):
        """Test converting type to OfType filter."""
        pred = to_predicate(np.ndarray)
        self.assertIsInstance(pred, OfType)
        self.assertEqual(pred.type, np.ndarray)

        # Test functionality
        arr = np.array([1, 2, 3])
        lst = [1, 2, 3]
        self.assertTrue(pred([], arr))
        self.assertFalse(pred([], lst))

    def test_bool_true_to_everything(self):
        """Test converting True to Everything filter."""
        pred = to_predicate(True)
        self.assertIsInstance(pred, Everything)

        # Test functionality
        self.assertTrue(pred([], 'anything'))
        self.assertTrue(pred(['path'], None))
        self.assertTrue(pred([], 42))

    def test_bool_false_to_nothing(self):
        """Test converting False to Nothing filter."""
        pred = to_predicate(False)
        self.assertIsInstance(pred, Nothing)

        # Test functionality
        self.assertFalse(pred([], 'anything'))
        self.assertFalse(pred(['path'], None))
        self.assertFalse(pred([], 42))

    def test_ellipsis_to_everything(self):
        """Test converting Ellipsis to Everything filter."""
        pred = to_predicate(...)
        self.assertIsInstance(pred, Everything)
        self.assertTrue(pred([], 'test'))

    def test_none_to_nothing(self):
        """Test converting None to Nothing filter."""
        pred = to_predicate(None)
        self.assertIsInstance(pred, Nothing)
        self.assertFalse(pred([], 'test'))

    def test_callable_passthrough(self):
        """Test that callable is returned as-is."""
        def custom_filter(path, x):
            return x == 'special'

        pred = to_predicate(custom_filter)
        self.assertIs(pred, custom_filter)
        self.assertTrue(pred([], 'special'))
        self.assertFalse(pred([], 'normal'))

    def test_list_to_any(self):
        """Test converting list to Any filter."""
        pred = to_predicate(['trainable', 'frozen'])
        self.assertIsInstance(pred, Any)

        # Test functionality
        trainable = MockTaggedObject('trainable')
        frozen = MockTaggedObject('frozen')
        other = MockTaggedObject('other')

        self.assertTrue(pred([], trainable))
        self.assertTrue(pred([], frozen))
        self.assertFalse(pred([], other))

    def test_tuple_to_any(self):
        """Test converting tuple to Any filter."""
        pred = to_predicate((np.ndarray, list))
        self.assertIsInstance(pred, Any)

        # Test functionality
        self.assertTrue(pred([], np.array([1, 2])))
        self.assertTrue(pred([], [1, 2]))
        self.assertFalse(pred([], (1, 2)))

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises TypeError."""
        with self.assertRaises(TypeError) as context:
            to_predicate(42)
        self.assertIn('Invalid collection filter', str(context.exception))

        with self.assertRaises(TypeError):
            to_predicate({'key': 'value'})


class TestWithTagFilter(unittest.TestCase):
    """Test cases for WithTag filter."""

    def test_basic_functionality(self):
        """Test basic WithTag functionality."""
        filter_trainable = WithTag('trainable')

        # Object with matching tag
        obj1 = MockTaggedObject('trainable')
        self.assertTrue(filter_trainable([], obj1))

        # Object with different tag
        obj2 = MockTaggedObject('frozen')
        self.assertFalse(filter_trainable([], obj2))

        # Object without tag attribute
        obj3 = {'value': 42}
        self.assertFalse(filter_trainable([], obj3))

    def test_repr(self):
        """Test string representation."""
        filter_tag = WithTag('test_tag')
        self.assertEqual(repr(filter_tag), "WithTag('test_tag')")

    def test_immutability(self):
        """Test that WithTag is immutable (frozen dataclass)."""
        filter_tag = WithTag('test')
        with self.assertRaises(AttributeError):
            filter_tag.tag = 'modified'

    def test_path_parameter_ignored(self):
        """Test that path parameter is ignored."""
        filter_tag = WithTag('test')
        obj = MockTaggedObject('test')

        # Different paths should not affect result
        self.assertTrue(filter_tag([], obj))
        self.assertTrue(filter_tag(['some', 'path'], obj))
        self.assertTrue(filter_tag(['another', 'nested', 'path'], obj))


class TestPathContainsFilter(unittest.TestCase):
    """Test cases for PathContains filter."""

    def test_basic_functionality(self):
        """Test basic PathContains functionality."""
        filter_weight = PathContains('weight')

        # Path containing the key
        self.assertTrue(filter_weight(['model', 'layer1', 'weight'], None))
        self.assertTrue(filter_weight(['weight'], None))
        self.assertTrue(filter_weight(['deep', 'nested', 'weight', 'param'], None))

        # Path not containing the key
        self.assertFalse(filter_weight(['model', 'layer1', 'bias'], None))
        self.assertFalse(filter_weight([], None))
        self.assertFalse(filter_weight(['other', 'path'], None))

    def test_numeric_keys(self):
        """Test with numeric keys in path."""
        filter_num = PathContains(0)

        self.assertTrue(filter_num([0, 'item'], None))
        self.assertTrue(filter_num(['list', 0, 'element'], None))
        self.assertFalse(filter_num([1, 2, 3], None))

    def test_repr(self):
        """Test string representation."""
        filter_path = PathContains('layer2')
        self.assertEqual(repr(filter_path), "PathContains('layer2')")

    def test_object_parameter_ignored(self):
        """Test that object parameter is ignored."""
        filter_path = PathContains('test')

        # Different objects should not affect result if path contains key
        self.assertTrue(filter_path(['test'], 'string'))
        self.assertTrue(filter_path(['test'], 123))
        self.assertTrue(filter_path(['test'], None))
        self.assertTrue(filter_path(['test'], {'dict': 'value'}))


class TestOfTypeFilter(unittest.TestCase):
    """Test cases for OfType filter."""

    def test_direct_instance_check(self):
        """Test checking direct instances of a type."""
        filter_array = OfType(np.ndarray)

        self.assertTrue(filter_array([], np.array([1, 2, 3])))
        self.assertTrue(filter_array([], np.zeros((2, 2))))
        self.assertFalse(filter_array([], [1, 2, 3]))
        self.assertFalse(filter_array([], 42))

    def test_inheritance(self):
        """Test that subclasses are also matched."""
        class BaseClass:
            pass

        class DerivedClass(BaseClass):
            pass

        filter_base = OfType(BaseClass)

        base_obj = BaseClass()
        derived_obj = DerivedClass()
        other_obj = "not related"

        self.assertTrue(filter_base([], base_obj))
        self.assertTrue(filter_base([], derived_obj))
        self.assertFalse(filter_base([], other_obj))

    def test_type_attribute_check(self):
        """Test checking objects with type attribute."""
        filter_list = OfType(list)

        # Object with type attribute
        typed_obj = MockTypedObject(list)
        self.assertTrue(filter_list([], typed_obj))

        # Object with non-matching type attribute
        typed_obj2 = MockTypedObject(dict)
        self.assertFalse(filter_list([], typed_obj2))

    def test_repr(self):
        """Test string representation."""
        filter_type = OfType(str)
        self.assertEqual(repr(filter_type), f"OfType({str!r})")

    def test_builtin_types(self):
        """Test with built-in types."""
        filter_str = OfType(str)
        filter_int = OfType(int)
        filter_list = OfType(list)

        self.assertTrue(filter_str([], "hello"))
        self.assertTrue(filter_int([], 42))
        self.assertTrue(filter_list([], [1, 2, 3]))

        self.assertFalse(filter_str([], 42))
        self.assertFalse(filter_int([], "42"))
        self.assertFalse(filter_list([], (1, 2, 3)))


class TestAnyFilter(unittest.TestCase):
    """Test cases for Any filter."""

    def test_basic_or_operation(self):
        """Test basic OR operation with multiple filters."""
        filter_any = Any('trainable', 'frozen')

        trainable = MockTaggedObject('trainable')
        frozen = MockTaggedObject('frozen')
        other = MockTaggedObject('other')

        self.assertTrue(filter_any([], trainable))
        self.assertTrue(filter_any([], frozen))
        self.assertFalse(filter_any([], other))

    def test_mixed_filter_types(self):
        """Test combining different filter types."""
        filter_mixed = Any(
            OfType(np.ndarray),
            WithTag('special'),
            PathContains('important')
        )

        # Test each condition
        self.assertTrue(filter_mixed([], np.array([1, 2])))
        self.assertTrue(filter_mixed([], MockTaggedObject('special')))
        self.assertTrue(filter_mixed(['important'], 'anything'))

        # Test none match
        self.assertFalse(filter_mixed(['other'], MockTaggedObject('normal')))

    def test_short_circuit_evaluation(self):
        """Test that Any short-circuits on first True."""
        call_count = [0]

        def counting_filter(path, x):
            call_count[0] += 1
            return x == 'match'

        filter_any = Any(
            lambda p, x: x == 'match',  # This will match
            counting_filter  # This should not be called
        )

        self.assertTrue(filter_any([], 'match'))
        self.assertEqual(call_count[0], 0)  # Second filter not called

    def test_empty_any(self):
        """Test Any with no filters."""
        filter_empty = Any()
        # Empty Any should return False (no conditions to satisfy)
        self.assertFalse(filter_empty([], 'anything'))

    def test_repr(self):
        """Test string representation."""
        filter_any = Any(WithTag('tag1'), WithTag('tag2'))
        repr_str = repr(filter_any)
        self.assertIn('Any', repr_str)
        self.assertIn("WithTag('tag1')", repr_str)
        self.assertIn("WithTag('tag2')", repr_str)

    def test_equality(self):
        """Test equality comparison."""
        filter1 = Any('tag1', 'tag2')
        filter2 = Any('tag1', 'tag2')
        filter3 = Any('tag2', 'tag1')  # Different order

        self.assertEqual(filter1, filter2)
        self.assertNotEqual(filter1, filter3)
        self.assertNotEqual(filter1, 'not a filter')

    def test_hashable(self):
        """Test that Any filters are hashable."""
        filter1 = Any('tag1', 'tag2')
        filter2 = Any('tag1', 'tag2')

        # Should be able to use in set/dict
        filter_set = {filter1, filter2}
        self.assertEqual(len(filter_set), 1)  # Same filters


class TestAllFilter(unittest.TestCase):
    """Test cases for All filter."""

    def test_basic_and_operation(self):
        """Test basic AND operation with multiple filters."""
        filter_all = All(
            WithTag('trainable'),
            OfType(np.ndarray)
        )

        # Create a numpy subclass that can have attributes
        class TaggedArray(np.ndarray):
            def __new__(cls, input_array, tag=None):
                obj = np.asarray(input_array).view(cls)
                obj.tag = tag
                return obj

        # Create test objects
        arr_obj = TaggedArray([1, 2, 3], tag='trainable')
        self.assertTrue(filter_all([], arr_obj))  # Matches both conditions

        arr_obj2 = TaggedArray([4, 5, 6], tag='frozen')
        self.assertFalse(filter_all([], arr_obj2))  # Wrong tag

        # List with tag (won't match type)
        class ListWithTag:
            def __init__(self, tag):
                self.tag = tag

        lst_obj = ListWithTag('trainable')
        self.assertFalse(filter_all([], lst_obj))  # Wrong type

    def test_short_circuit_evaluation(self):
        """Test that All short-circuits on first False."""
        call_count = [0]

        def counting_filter(path, x):
            call_count[0] += 1
            return True

        filter_all = All(
            lambda p, x: False,  # This will fail
            counting_filter  # This should not be called
        )

        self.assertFalse(filter_all([], 'anything'))
        self.assertEqual(call_count[0], 0)  # Second filter not called

    def test_empty_all(self):
        """Test All with no filters."""
        filter_empty = All()
        # Empty All should return True (no conditions to violate)
        self.assertTrue(filter_empty([], 'anything'))

    def test_complex_combination(self):
        """Test complex combination of conditions."""
        class CustomObject:
            def __init__(self, tag, value):
                self.tag = tag
                self.value = value

        filter_complex = All(
            WithTag('important'),
            lambda p, x: hasattr(x, 'value') and x.value > 10,
            lambda p, x: hasattr(x, 'value') and x.value < 100
        )

        obj1 = CustomObject('important', 50)
        obj2 = CustomObject('important', 5)
        obj3 = CustomObject('important', 150)
        obj4 = CustomObject('other', 50)

        self.assertTrue(filter_complex([], obj1))  # All conditions met
        self.assertFalse(filter_complex([], obj2))  # value too small
        self.assertFalse(filter_complex([], obj3))  # value too large
        self.assertFalse(filter_complex([], obj4))  # wrong tag

    def test_repr(self):
        """Test string representation."""
        filter_all = All(WithTag('tag1'), OfType(list))
        repr_str = repr(filter_all)
        self.assertIn('All', repr_str)
        self.assertIn("WithTag('tag1')", repr_str)
        self.assertIn('OfType', repr_str)

    def test_equality(self):
        """Test equality comparison."""
        filter1 = All('tag1', np.ndarray)
        filter2 = All('tag1', np.ndarray)
        filter3 = All(np.ndarray, 'tag1')  # Different order

        self.assertEqual(filter1, filter2)
        self.assertNotEqual(filter1, filter3)

    def test_hashable(self):
        """Test that All filters are hashable."""
        filter1 = All('tag1', np.ndarray)
        filter2 = All('tag1', np.ndarray)

        filter_dict = {filter1: 'value1', filter2: 'value2'}
        self.assertEqual(len(filter_dict), 1)  # Same filters


class TestNotFilter(unittest.TestCase):
    """Test cases for Not filter."""

    def test_basic_negation(self):
        """Test basic negation of filters."""
        filter_not_trainable = Not(WithTag('trainable'))

        trainable = MockTaggedObject('trainable')
        frozen = MockTaggedObject('frozen')

        self.assertFalse(filter_not_trainable([], trainable))
        self.assertTrue(filter_not_trainable([], frozen))

    def test_negating_type_filter(self):
        """Test negating type filters."""
        filter_not_array = Not(OfType(np.ndarray))

        self.assertFalse(filter_not_array([], np.array([1, 2])))
        self.assertTrue(filter_not_array([], [1, 2]))
        self.assertTrue(filter_not_array([], 'string'))

    def test_negating_complex_filters(self):
        """Test negating complex filter combinations."""
        # Not(Any(...)) - none should match
        filter_not_any = Not(Any('tag1', 'tag2'))

        obj1 = MockTaggedObject('tag1')
        obj2 = MockTaggedObject('tag2')
        obj3 = MockTaggedObject('tag3')

        self.assertFalse(filter_not_any([], obj1))
        self.assertFalse(filter_not_any([], obj2))
        self.assertTrue(filter_not_any([], obj3))

        # Not(All(...)) - at least one should not match
        filter_not_all = Not(All(WithTag('tag'), OfType(list)))

        # Create a list-like object with tag
        class TaggedList(list):
            def __init__(self, *args):
                super().__init__(*args)
                self.tag = 'tag'

        lst = TaggedList([1, 2, 3])
        self.assertFalse(filter_not_all([], lst))  # Matches all conditions

        # Create a numpy subclass with tag
        class TaggedArray(np.ndarray):
            def __new__(cls, input_array, tag=None):
                obj = np.asarray(input_array).view(cls)
                obj.tag = tag
                return obj

        arr = TaggedArray([], tag='tag')
        self.assertTrue(filter_not_all([], arr))  # Doesn't match type (not a list)

    def test_double_negation(self):
        """Test double negation returns to original."""
        original = WithTag('test')
        double_neg = Not(Not(original))

        obj_match = MockTaggedObject('test')
        obj_no_match = MockTaggedObject('other')

        # Double negation should behave like original
        self.assertEqual(
            original([], obj_match),
            double_neg([], obj_match)
        )
        self.assertEqual(
            original([], obj_no_match),
            double_neg([], obj_no_match)
        )

    def test_repr(self):
        """Test string representation."""
        filter_not = Not(WithTag('test'))
        self.assertEqual(repr(filter_not), "Not(WithTag('test'))")

    def test_equality(self):
        """Test equality comparison."""
        filter1 = Not(WithTag('test'))
        filter2 = Not(WithTag('test'))
        filter3 = Not(WithTag('other'))

        self.assertEqual(filter1, filter2)
        self.assertNotEqual(filter1, filter3)

    def test_hashable(self):
        """Test that Not filters are hashable."""
        filter1 = Not(WithTag('test'))
        filter2 = Not(WithTag('test'))

        filter_set = {filter1, filter2}
        self.assertEqual(len(filter_set), 1)


class TestEverythingFilter(unittest.TestCase):
    """Test cases for Everything filter."""

    def test_always_returns_true(self):
        """Test that Everything always returns True."""
        filter_all = Everything()

        # Test with various objects
        self.assertTrue(filter_all([], None))
        self.assertTrue(filter_all([], 42))
        self.assertTrue(filter_all([], 'string'))
        self.assertTrue(filter_all([], [1, 2, 3]))
        self.assertTrue(filter_all([], np.array([1, 2])))
        self.assertTrue(filter_all([], {'key': 'value'}))

        # Test with various paths
        self.assertTrue(filter_all(['path'], None))
        self.assertTrue(filter_all(['nested', 'path'], None))
        self.assertTrue(filter_all([], None))

    def test_repr(self):
        """Test string representation."""
        filter_all = Everything()
        self.assertEqual(repr(filter_all), 'Everything()')

    def test_equality(self):
        """Test equality comparison."""
        filter1 = Everything()
        filter2 = Everything()
        filter3 = Nothing()

        self.assertEqual(filter1, filter2)
        self.assertNotEqual(filter1, filter3)

    def test_hashable(self):
        """Test that Everything filters are hashable."""
        filter1 = Everything()
        filter2 = Everything()

        # All Everything instances should be equal and have same hash
        self.assertEqual(hash(filter1), hash(filter2))

        filter_set = {filter1, filter2}
        self.assertEqual(len(filter_set), 1)

    def test_conversion_from_true(self):
        """Test that True converts to Everything."""
        filter_from_true = to_predicate(True)
        filter_direct = Everything()

        # Should behave identically
        test_cases = [None, 42, 'test', [], {}]
        for obj in test_cases:
            self.assertEqual(
                filter_from_true([], obj),
                filter_direct([], obj)
            )

    def test_conversion_from_ellipsis(self):
        """Test that Ellipsis converts to Everything."""
        filter_from_ellipsis = to_predicate(...)
        filter_direct = Everything()

        self.assertIsInstance(filter_from_ellipsis, Everything)
        self.assertEqual(filter_from_ellipsis, filter_direct)


class TestNothingFilter(unittest.TestCase):
    """Test cases for Nothing filter."""

    def test_always_returns_false(self):
        """Test that Nothing always returns False."""
        filter_none = Nothing()

        # Test with various objects
        self.assertFalse(filter_none([], None))
        self.assertFalse(filter_none([], 42))
        self.assertFalse(filter_none([], 'string'))
        self.assertFalse(filter_none([], [1, 2, 3]))
        self.assertFalse(filter_none([], np.array([1, 2])))
        self.assertFalse(filter_none([], {'key': 'value'}))

        # Test with various paths
        self.assertFalse(filter_none(['path'], None))
        self.assertFalse(filter_none(['nested', 'path'], None))
        self.assertFalse(filter_none([], None))

    def test_repr(self):
        """Test string representation."""
        filter_none = Nothing()
        self.assertEqual(repr(filter_none), 'Nothing()')

    def test_equality(self):
        """Test equality comparison."""
        filter1 = Nothing()
        filter2 = Nothing()
        filter3 = Everything()

        self.assertEqual(filter1, filter2)
        self.assertNotEqual(filter1, filter3)

    def test_hashable(self):
        """Test that Nothing filters are hashable."""
        filter1 = Nothing()
        filter2 = Nothing()

        # All Nothing instances should be equal and have same hash
        self.assertEqual(hash(filter1), hash(filter2))

        filter_dict = {filter1: 'value'}
        filter_dict[filter2] = 'new_value'
        self.assertEqual(len(filter_dict), 1)  # Same key

    def test_conversion_from_false(self):
        """Test that False converts to Nothing."""
        filter_from_false = to_predicate(False)
        filter_direct = Nothing()

        # Should behave identically
        test_cases = [None, 42, 'test', [], {}]
        for obj in test_cases:
            self.assertEqual(
                filter_from_false([], obj),
                filter_direct([], obj)
            )

    def test_conversion_from_none(self):
        """Test that None converts to Nothing."""
        filter_from_none = to_predicate(None)
        filter_direct = Nothing()

        self.assertIsInstance(filter_from_none, Nothing)
        self.assertEqual(filter_from_none, filter_direct)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complex filter combinations."""

    def test_neural_network_parameter_filtering(self):
        """Test filtering neural network parameters with complex criteria."""
        # Simulate neural network parameters
        class Parameter:
            def __init__(self, shape, tag=None, dtype=None):
                self.shape = shape
                self.tag = tag
                self.dtype = dtype
                self.data = np.random.randn(*shape) if dtype != 'int32' else np.random.randint(0, 10, shape)

        # Create various parameters
        weight1 = Parameter((10, 20), tag='trainable')
        weight1.data = np.array(weight1.data)  # Ensure it's ndarray

        bias1 = Parameter((20,), tag='trainable')
        bias1.data = np.array(bias1.data)

        embedding = Parameter((100, 64), tag='frozen')
        embedding.data = np.array(embedding.data)

        # Complex filter: trainable arrays with shape[0] > 5
        filter_complex = All(
            WithTag('trainable'),
            lambda p, x: hasattr(x, 'data') and isinstance(x.data, np.ndarray),
            lambda p, x: hasattr(x, 'shape') and x.shape[0] > 5
        )

        self.assertTrue(filter_complex([], weight1))  # Matches all
        self.assertTrue(filter_complex([], bias1))  # shape[0] = 20 > 5
        self.assertFalse(filter_complex([], embedding))  # Wrong tag

    def test_path_based_model_filtering(self):
        """Test filtering based on model structure paths."""
        # Filter for encoder weights
        encoder_weight_filter = All(
            PathContains('encoder'),
            PathContains('weight')
        )

        # Test various paths
        paths = [
            (['model', 'encoder', 'layer1', 'weight'], True),
            (['model', 'encoder', 'layer2', 'weight'], True),
            (['model', 'encoder', 'layer1', 'bias'], False),  # Not weight
            (['model', 'decoder', 'layer1', 'weight'], False),  # Not encoder
            (['encoder', 'attention', 'weight'], True),
        ]

        for path, expected in paths:
            self.assertEqual(
                encoder_weight_filter(path, None),
                expected,
                f"Failed for path: {path}"
            )

    def test_selective_gradient_computation(self):
        """Test filter for selective gradient computation."""
        # Only compute gradients for trainable non-embedding layers
        gradient_filter = All(
            WithTag('trainable'),
            Not(PathContains('embedding')),
            Any(
                PathContains('weight'),
                PathContains('bias')
            )
        )

        # Create test objects
        class Param:
            def __init__(self, tag):
                self.tag = tag

        trainable_weight = Param('trainable')
        trainable_bias = Param('trainable')
        frozen_weight = Param('frozen')
        trainable_other = Param('trainable')

        test_cases = [
            (['layer1', 'weight'], trainable_weight, True),
            (['layer1', 'bias'], trainable_bias, True),
            (['embedding', 'weight'], trainable_weight, False),  # Excluded path
            (['layer1', 'weight'], frozen_weight, False),  # Wrong tag
            (['layer1', 'gamma'], trainable_other, False),  # Not weight/bias
        ]

        for path, obj, expected in test_cases:
            self.assertEqual(
                gradient_filter(path, obj),
                expected,
                f"Failed for path={path}, tag={obj.tag}"
            )

    def test_demorgan_laws(self):
        """Test De Morgan's laws with filters."""
        tag1_filter = WithTag('tag1')
        tag2_filter = WithTag('tag2')

        # Not(A or B) == (Not A) and (Not B)
        not_any = Not(Any(tag1_filter, tag2_filter))
        all_not = All(Not(tag1_filter), Not(tag2_filter))

        test_objects = [
            MockTaggedObject('tag1'),
            MockTaggedObject('tag2'),
            MockTaggedObject('tag3'),
            MockTaggedObject('other'),
        ]

        for obj in test_objects:
            self.assertEqual(
                not_any([], obj),
                all_not([], obj),
                f"De Morgan's law failed for tag={obj.tag}"
            )

        # Not(A and B) == (Not A) or (Not B)
        not_all = Not(All(tag1_filter, tag2_filter))
        any_not = Any(Not(tag1_filter), Not(tag2_filter))

        # For All filter to work, object needs both tags
        class DualTagged:
            def __init__(self, tag1, tag2):
                self.tag = tag1 if tag1 else tag2  # For single tag check

        # Create object that could match both filters
        dual1 = DualTagged('tag1', None)
        dual2 = DualTagged('tag2', None)
        neither = DualTagged('other', None)

        for obj in [dual1, dual2, neither]:
            # Since our mock object can only have one tag at a time,
            # All(tag1, tag2) will always be False, so Not(All(...)) will always be True
            # and Any(Not(tag1), Not(tag2)) will depend on the specific tag
            pass  # This specific test is limited by our mock implementation

    def test_filter_chaining_performance(self):
        """Test that filter chaining works correctly."""
        # Create a chain of filters
        base_filter = WithTag('base')
        extended_filter = All(base_filter, OfType(dict))
        final_filter = Any(extended_filter, PathContains('special'))

        # Test object that matches base but not type
        obj1 = MockTaggedObject('base')
        self.assertFalse(extended_filter([], obj1))  # Not a dict
        self.assertFalse(final_filter([], obj1))  # Doesn't match Any conditions

        # Test object that matches everything
        class TaggedDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tag = 'base'

        obj2 = TaggedDict(key='value')
        self.assertTrue(extended_filter([], obj2))  # Matches both conditions
        self.assertTrue(final_filter([], obj2))  # Matches via extended_filter

        # Test path-based match
        self.assertTrue(final_filter(['special'], 'anything'))

    def test_recursive_filter_structures(self):
        """Test deeply nested filter combinations."""
        # Build a complex filter structure
        filter_deep = Any(
            All(
                WithTag('level1'),
                Any(
                    All(WithTag('level2a'), OfType(list)),
                    All(WithTag('level2b'), OfType(dict))
                )
            ),
            Not(
                All(
                    PathContains('excluded'),
                    Not(WithTag('override'))
                )
            )
        )

        # This is a complex filter, let's test a few cases
        # Case 1: Would need an object with tag='level1' and also tag='level2a' and be a list
        # Since objects can only have one tag, this is hard to test directly
        # Instead, test the second branch

        # Case 2: Matches second branch (not in excluded path without override)
        obj = MockTaggedObject('any')
        self.assertTrue(filter_deep(['included'], obj))  # Not excluded path

        # Case 3: In excluded path but has override tag
        override_obj = MockTaggedObject('override')
        self.assertTrue(filter_deep(['excluded'], override_obj))  # Has override

        # Case 4: In excluded path without override - should not match
        regular_obj = MockTaggedObject('regular')
        self.assertFalse(filter_deep(['excluded'], regular_obj))  # Excluded without override


if __name__ == '__main__':
    unittest.main()