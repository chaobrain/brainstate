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
Comprehensive tests for pretty_repr module.
"""

import dataclasses
import unittest
from typing import Iterator, Union

from brainstate.util._pretty_repr import (
    PrettyType,
    PrettyAttr,
    PrettyRepr,
    pretty_repr_elem,
    pretty_repr_object,
    MappingReprMixin,
    PrettyMapping,
    PrettyReprContext,
    yield_unique_pretty_repr_items,
    _default_repr_object,
    _default_repr_attr,
)


class TestPrettyType(unittest.TestCase):
    """Test cases for PrettyType dataclass."""

    def test_default_values(self):
        """Test PrettyType with default values."""
        pt = PrettyType(type='MyClass')
        self.assertEqual(pt.type, 'MyClass')
        self.assertEqual(pt.start, '(')
        self.assertEqual(pt.end, ')')
        self.assertEqual(pt.value_sep, '=')
        self.assertEqual(pt.elem_indent, '  ')
        self.assertEqual(pt.empty_repr, '')

    def test_custom_values(self):
        """Test PrettyType with custom values."""
        pt = PrettyType(
            type=dict,
            start='{',
            end='}',
            value_sep=': ',
            elem_indent='    ',
            empty_repr='<empty>'
        )
        self.assertEqual(pt.type, dict)
        self.assertEqual(pt.start, '{')
        self.assertEqual(pt.end, '}')
        self.assertEqual(pt.value_sep, ': ')
        self.assertEqual(pt.elem_indent, '    ')
        self.assertEqual(pt.empty_repr, '<empty>')

    def test_type_can_be_string_or_class(self):
        """Test that type can be either string or class."""
        pt1 = PrettyType(type='StringType')
        self.assertIsInstance(pt1.type, str)

        pt2 = PrettyType(type=list)
        self.assertEqual(pt2.type, list)


class TestPrettyAttr(unittest.TestCase):
    """Test cases for PrettyAttr dataclass."""

    def test_default_values(self):
        """Test PrettyAttr with default values."""
        pa = PrettyAttr(key='name', value='test')
        self.assertEqual(pa.key, 'name')
        self.assertEqual(pa.value, 'test')
        self.assertEqual(pa.start, '')
        self.assertEqual(pa.end, '')

    def test_custom_values(self):
        """Test PrettyAttr with custom values."""
        pa = PrettyAttr(key='count', value=42, start='[', end=']')
        self.assertEqual(pa.key, 'count')
        self.assertEqual(pa.value, 42)
        self.assertEqual(pa.start, '[')
        self.assertEqual(pa.end, ']')

    def test_value_types(self):
        """Test PrettyAttr with various value types."""
        pa1 = PrettyAttr('str_value', 'string')
        self.assertEqual(pa1.value, 'string')

        pa2 = PrettyAttr('int_value', 123)
        self.assertEqual(pa2.value, 123)

        pa3 = PrettyAttr('list_value', [1, 2, 3])
        self.assertEqual(pa3.value, [1, 2, 3])

        pa4 = PrettyAttr('dict_value', {'a': 1})
        self.assertEqual(pa4.value, {'a': 1})


class SimplePrettyRepr(PrettyRepr):
    """Simple implementation of PrettyRepr for testing."""

    def __init__(self, value, name='SimpleObject'):
        self.value = value
        self.name = name

    def __pretty_repr__(self) -> Iterator[Union[PrettyType, PrettyAttr]]:
        yield PrettyType(type=self.name)
        yield PrettyAttr('value', self.value)


class TestPrettyRepr(unittest.TestCase):
    """Test cases for PrettyRepr abstract class."""

    def test_simple_repr(self):
        """Test simple pretty representation."""
        obj = SimplePrettyRepr(42)
        result = repr(obj)
        self.assertIn('SimpleObject', result)
        self.assertIn('value=42', result)

    def test_custom_type_config(self):
        """Test PrettyRepr with custom type configuration."""

        class CustomRepr(PrettyRepr):
            def __init__(self, data):
                self.data = data

            def __pretty_repr__(self):
                yield PrettyType(type='CustomObject', start='<', end='>', value_sep=' -> ')
                yield PrettyAttr('data', self.data)

        obj = CustomRepr({'key': 'value'})
        result = repr(obj)
        self.assertIn('CustomObject<', result)
        self.assertIn('data -> ', result)
        self.assertIn('>', result)

    def test_multiple_attributes(self):
        """Test PrettyRepr with multiple attributes."""

        class MultiAttrRepr(PrettyRepr):
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c

            def __pretty_repr__(self):
                yield PrettyType(type=self.__class__)
                yield PrettyAttr('a', self.a)
                yield PrettyAttr('b', self.b)
                yield PrettyAttr('c', self.c)

        obj = MultiAttrRepr(1, 'two', [3])
        result = repr(obj)
        self.assertIn('MultiAttrRepr', result)
        self.assertIn('a=1', result)
        self.assertIn("b=two", result)  # String value is not re-quoted
        self.assertIn('c=[3]', result)

    def test_empty_object(self):
        """Test PrettyRepr with no attributes."""

        class EmptyRepr(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType(type='EmptyObject', empty_repr='<no data>')

        obj = EmptyRepr()
        result = repr(obj)
        self.assertIn('EmptyObject', result)
        self.assertIn('<no data>', result)


class TestPrettyReprElem(unittest.TestCase):
    """Test cases for pretty_repr_elem function."""

    def test_basic_elem(self):
        """Test basic element formatting."""
        pt = PrettyType(type='Test')
        elem = PrettyAttr('key', 'value')
        result = pretty_repr_elem(pt, elem)
        # Value is already a string, so it's not quoted again
        self.assertEqual(result, "  key=value")

    def test_elem_with_custom_indent(self):
        """Test element with custom indentation."""
        pt = PrettyType(type='Test', elem_indent='    ')
        elem = PrettyAttr('key', 123)
        result = pretty_repr_elem(pt, elem)
        self.assertEqual(result, "    key=123")

    def test_elem_with_custom_separator(self):
        """Test element with custom value separator."""
        pt = PrettyType(type='Test', value_sep=': ')
        elem = PrettyAttr('key', 'value')
        result = pretty_repr_elem(pt, elem)
        self.assertEqual(result, "  key: value")

    def test_elem_with_start_end(self):
        """Test element with start and end markers."""
        pt = PrettyType(type='Test')
        elem = PrettyAttr('key', 'value', start='[', end=']')
        result = pretty_repr_elem(pt, elem)
        self.assertEqual(result, "  [key=value]")

    def test_elem_with_multiline_value(self):
        """Test element with multiline value."""
        pt = PrettyType(type='Test')
        elem = PrettyAttr('key', 'line1\nline2\nline3')
        result = pretty_repr_elem(pt, elem)
        expected = "  key=line1\n  line2\n  line3"
        self.assertEqual(result, expected)

    def test_elem_invalid_type(self):
        """Test that non-PrettyAttr raises TypeError."""
        pt = PrettyType(type='Test')
        with self.assertRaises(TypeError) as cm:
            pretty_repr_elem(pt, "not a PrettyAttr")
        self.assertIn("Item must be Elem", str(cm.exception))


class TestPrettyReprObject(unittest.TestCase):
    """Test cases for pretty_repr_object function."""

    def test_valid_object(self):
        """Test with valid PrettyRepr object."""
        obj = SimplePrettyRepr(42, 'TestObject')
        result = pretty_repr_object(obj)
        self.assertIn('TestObject', result)
        self.assertIn('value=42', result)

    def test_invalid_object(self):
        """Test that non-PrettyRepr object raises TypeError."""
        with self.assertRaises(TypeError) as cm:
            pretty_repr_object("not a PrettyRepr")
        self.assertIn("is not representable", str(cm.exception))

    def test_invalid_first_item(self):
        """Test that invalid first item raises TypeError."""

        class InvalidRepr(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyAttr('key', 'value')  # Should yield PrettyType first

        obj = InvalidRepr()
        with self.assertRaises(TypeError) as cm:
            pretty_repr_object(obj)
        self.assertIn("First item must be PrettyType", str(cm.exception))

    def test_empty_representation(self):
        """Test object with no attributes."""

        class EmptyRepr(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType(type='Empty', empty_repr='∅')

        obj = EmptyRepr()
        result = pretty_repr_object(obj)
        self.assertEqual(result, 'Empty(∅)')

    def test_complex_nested_formatting(self):
        """Test complex nested formatting."""

        class ComplexRepr(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType(
                    type='Complex',
                    start='{\n',
                    end='\n}',
                    elem_indent='    ',
                    value_sep=' => '
                )
                yield PrettyAttr('first', 'value1')
                yield PrettyAttr('second', {'nested': 'dict'})

        obj = ComplexRepr()
        result = pretty_repr_object(obj)
        self.assertIn('Complex', result)
        self.assertIn('first => ', result)
        self.assertIn('second => ', result)


class TestMappingReprMixin(unittest.TestCase):
    """Test cases for MappingReprMixin."""

    def test_basic_mapping(self):
        """Test basic mapping representation."""

        class MyMapping(dict, MappingReprMixin):
            pass

        m = MyMapping({'a': 1, 'b': 2})
        # Get the pretty repr items - MappingReprMixin only provides __pretty_repr__
        # but needs to be mixed with a dict-like class
        items = list(m.__pretty_repr__())

        # Check first item is PrettyType
        self.assertIsInstance(items[0], PrettyType)
        self.assertEqual(items[0].value_sep, ': ')
        self.assertEqual(items[0].start, '{')
        self.assertEqual(items[0].end, '}')

        # Check that we have the expected number of items
        self.assertEqual(len(items), 3)  # PrettyType + 2 attrs

        # Check attributes
        attr_items = [item for item in items[1:] if isinstance(item, PrettyAttr)]
        self.assertEqual(len(attr_items), 2)

        # Keys should be repr'd (with quotes for strings)
        keys = [item.key for item in attr_items]
        self.assertIn("'a'", keys)
        self.assertIn("'b'", keys)

    def test_empty_mapping(self):
        """Test empty mapping representation."""

        class MyMapping(dict, MappingReprMixin):
            pass

        m = MyMapping()
        items = list(m.__pretty_repr__())
        self.assertEqual(len(items), 1)  # Only PrettyType, no attributes


class TestPrettyMapping(unittest.TestCase):
    """Test cases for PrettyMapping class."""

    def test_basic_pretty_mapping(self):
        """Test basic PrettyMapping."""
        pm = PrettyMapping({'x': 10, 'y': 20})
        result = repr(pm)
        self.assertIn("'x': 10", result)
        self.assertIn("'y': 20", result)

    def test_pretty_mapping_with_type_name(self):
        """Test PrettyMapping with custom type name."""
        pm = PrettyMapping({'a': 1}, type_name='MyDict')
        result = repr(pm)
        self.assertIn('MyDict', result)
        self.assertIn("'a': 1", result)

    def test_empty_pretty_mapping(self):
        """Test empty PrettyMapping."""
        pm = PrettyMapping({})
        result = repr(pm)
        self.assertIn('{', result)
        self.assertIn('}', result)

    def test_nested_mapping(self):
        """Test PrettyMapping with nested values."""
        pm = PrettyMapping({
            'simple': 42,
            'nested': {'inner': 'value'},
            'list': [1, 2, 3]
        })
        result = repr(pm)
        self.assertIn("'simple': 42", result)
        self.assertIn("'nested':", result)
        self.assertIn("'list': [1, 2, 3]", result)


class TestPrettyReprContext(unittest.TestCase):
    """Test cases for PrettyReprContext."""

    def test_initial_state(self):
        """Test initial state of context."""
        ctx = PrettyReprContext()
        self.assertIsNone(ctx.seen_modules_repr)

    def test_thread_local_behavior(self):
        """Test that context is thread-local."""
        ctx1 = PrettyReprContext()
        ctx2 = PrettyReprContext()

        ctx1.seen_modules_repr = {'test': 1}
        self.assertIsNone(ctx2.seen_modules_repr)


class TestYieldUniquePrettyReprItems(unittest.TestCase):
    """Test cases for yield_unique_pretty_repr_items function."""

    def test_basic_usage(self):
        """Test basic usage with simple object."""

        @dataclasses.dataclass
        class SimpleObject:
            value: int

        obj = SimpleObject(42)
        items = list(yield_unique_pretty_repr_items(obj))

        # Should yield PrettyType first
        self.assertIsInstance(items[0], PrettyType)
        self.assertEqual(items[0].type, SimpleObject)

        # Should yield attribute
        attr_items = [item for item in items[1:] if isinstance(item, PrettyAttr)]
        self.assertTrue(any(item.key == 'value' for item in attr_items))

    def test_custom_repr_functions(self):
        """Test with custom repr functions."""

        def custom_repr_object(node):
            yield PrettyType(type='CustomType', start='<', end='>')

        def custom_repr_attr(node):
            yield PrettyAttr('custom_attr', 'custom_value')

        obj = object()
        items = list(yield_unique_pretty_repr_items(
            obj,
            repr_object=custom_repr_object,
            repr_attr=custom_repr_attr
        ))

        self.assertIsInstance(items[0], PrettyType)
        self.assertEqual(items[0].type, 'CustomType')

        attr_items = [item for item in items[1:] if isinstance(item, PrettyAttr)]
        self.assertTrue(any(item.key == 'custom_attr' for item in attr_items))

    def test_circular_reference_handling(self):
        """Test handling of circular references."""

        class Node:
            def __init__(self, value):
                self.value = value
                self.next = None

        # Create circular reference
        node1 = Node(1)
        node2 = Node(2)
        node1.next = node2
        node2.next = node1

        # Test that within same context, circular reference is detected
        from brainstate.util._pretty_repr import CONTEXT

        # Clean start
        CONTEXT.seen_modules_repr = None

        # Set up context to track seen objects
        CONTEXT.seen_modules_repr = {}

        # First pass - node1 will be added to seen
        items1 = list(yield_unique_pretty_repr_items(node1))

        # Second pass - should detect node1 is already seen
        items2 = list(yield_unique_pretty_repr_items(node1))

        # Check that second pass detected circular reference
        type_items = [item for item in items2 if isinstance(item, PrettyType)]
        self.assertTrue(len(type_items) > 0)
        self.assertTrue(any(item.empty_repr == '...' for item in type_items))

        # Clean up
        CONTEXT.seen_modules_repr = None

    def test_context_cleanup(self):
        """Test that context is properly cleaned up."""
        from brainstate.util._pretty_repr import CONTEXT

        # Clean up any previous state
        CONTEXT.seen_modules_repr = None

        # Use a class instance that has __dict__
        class TestObj:
            def __init__(self):
                self.test = 'value'

        obj = TestObj()
        list(yield_unique_pretty_repr_items(obj))

        # Context should be cleaned up after
        self.assertIsNone(CONTEXT.seen_modules_repr)

    def test_nested_calls(self):
        """Test nested calls don't recreate context."""
        from brainstate.util._pretty_repr import CONTEXT

        # Clean up any previous state
        CONTEXT.seen_modules_repr = None

        class Outer:
            def __init__(self):
                self.inner = Inner()

        class Inner:
            def __init__(self):
                self.value = 42

        def repr_outer_attr(node):
            # This will trigger nested yield_unique_pretty_repr_items
            for item in yield_unique_pretty_repr_items(node.inner):
                pass
            yield PrettyAttr('inner', node.inner)

        outer = Outer()

        # This should not error and should handle nested calls properly
        items = list(yield_unique_pretty_repr_items(
            outer,
            repr_attr=repr_outer_attr
        ))

        # Clean up should happen
        self.assertIsNone(CONTEXT.seen_modules_repr)


class TestDefaultReprFunctions(unittest.TestCase):
    """Test cases for default repr functions."""

    def test_default_repr_object(self):
        """Test _default_repr_object function."""

        class MyClass:
            pass

        obj = MyClass()
        items = list(_default_repr_object(obj))

        self.assertEqual(len(items), 1)
        self.assertIsInstance(items[0], PrettyType)
        self.assertEqual(items[0].type, MyClass)

    def test_default_repr_attr(self):
        """Test _default_repr_attr function."""

        class MyClass:
            def __init__(self):
                self.public_attr = 'public'
                self._private_attr = 'private'
                self.__dunder_attr = 'dunder'
                self.number = 42
                self.list_attr = [1, 2, 3]

        obj = MyClass()
        items = list(_default_repr_attr(obj))

        # Should include public attributes
        attr_keys = {item.key for item in items}
        self.assertIn('public_attr', attr_keys)
        self.assertIn('number', attr_keys)
        self.assertIn('list_attr', attr_keys)

        # Should exclude private attributes
        self.assertNotIn('_private_attr', attr_keys)
        self.assertNotIn('__dunder_attr', attr_keys)

        # Check values are repr'd
        public_item = next(item for item in items if item.key == 'public_attr')
        self.assertEqual(public_item.value, "'public'")

        number_item = next(item for item in items if item.key == 'number')
        self.assertEqual(number_item.value, '42')

    def test_default_repr_attr_no_vars(self):
        """Test _default_repr_attr with object that has no __dict__."""

        class NoVars:
            __slots__ = ('x', 'y')

            def __init__(self):
                self.x = 1
                self.y = 2

        obj = NoVars()
        # vars() will raise TypeError for objects without __dict__
        with self.assertRaises(TypeError):
            list(_default_repr_attr(obj))


class TestIntegration(unittest.TestCase):
    """Integration tests for the pretty_repr module."""

    def test_complex_nested_structure(self):
        """Test complex nested structure representation."""

        class Container(PrettyRepr):
            def __init__(self, name, children=None):
                self.name = name
                self.children = children or []

            def __pretty_repr__(self):
                yield PrettyType(type=self.__class__.__name__, start='[', end=']')
                yield PrettyAttr('name', self.name)
                if self.children:
                    yield PrettyAttr('children', self.children)

        # Create nested structure
        leaf1 = Container('leaf1')
        leaf2 = Container('leaf2')
        branch = Container('branch', [leaf1, leaf2])
        root = Container('root', [branch])

        result = repr(root)
        self.assertIn('Container', result)
        self.assertIn('name=', result)
        self.assertIn('root', result)
        self.assertIn('children=', result)

    def test_mixed_types_representation(self):
        """Test representation with mixed types."""

        class MixedTypes(PrettyRepr):
            def __init__(self):
                self.string = "hello"
                self.number = 42
                self.float_num = 3.14
                self.bool_val = True
                self.none_val = None
                self.list_val = [1, 2, 3]
                self.dict_val = {'key': 'value'}
                self.tuple_val = (1, 2)
                self.set_val = {1, 2, 3}

            def __pretty_repr__(self):
                yield PrettyType(type='MixedTypes')
                for key, value in vars(self).items():
                    yield PrettyAttr(key, value)

        obj = MixedTypes()
        result = repr(obj)

        # Check all types are represented correctly
        # Note: string values passed to PrettyAttr are not re-quoted
        self.assertIn("string=hello", result)
        self.assertIn("number=42", result)
        self.assertIn("float_num=3.14", result)
        self.assertIn("bool_val=True", result)
        self.assertIn("none_val=None", result)
        self.assertIn("list_val=[1, 2, 3]", result)
        self.assertIn("dict_val={'key': 'value'}", result)

    def test_custom_formatting_styles(self):
        """Test various custom formatting styles."""

        class XMLStyle(PrettyRepr):
            def __init__(self, tag, content):
                self.tag = tag
                self.content = content

            def __pretty_repr__(self):
                yield PrettyType(
                    type='',
                    start=f'<{self.tag}>',
                    end=f'</{self.tag}>',
                    value_sep='',
                    elem_indent='',
                    empty_repr=''
                )
                yield PrettyAttr('', self.content)

        obj = XMLStyle('div', 'Hello World')
        result = repr(obj)
        self.assertIn('<div>', result)
        self.assertIn('</div>', result)
        self.assertIn('Hello World', result)

    def test_unicode_handling(self):
        """Test handling of unicode characters."""

        class UnicodeObj(PrettyRepr):
            def __init__(self):
                self.emoji = "🎉"
                self.chinese = "你好"
                self.special = "café"

            def __pretty_repr__(self):
                yield PrettyType(type='UnicodeObj')
                for key, value in vars(self).items():
                    yield PrettyAttr(key, value)

        obj = UnicodeObj()
        result = repr(obj)

        # Unicode should be preserved (string values are not re-quoted)
        self.assertIn("emoji=🎉", result)
        self.assertIn("chinese=你好", result)
        self.assertIn("special=café", result)


if __name__ == '__main__':
    unittest.main()
