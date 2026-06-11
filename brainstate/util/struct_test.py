"""
Comprehensive tests for the struct module.
"""

import pickle
import unittest
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util
import pytest

# Import the modules to test
from brainstate.util import (
    field,
    is_dataclass,
    dataclass,
    PyTreeNode,
    FrozenDict,
    freeze,
    unfreeze,
    copy,
    pop,
    pretty_repr,
)


class TestField:
    """Test the field function."""

    def test_field_with_pytree_node_true(self):
        """Test field with pytree_node=True."""
        f = field(pytree_node=True)
        assert f.metadata['pytree_node'] is True

    def test_field_with_pytree_node_false(self):
        """Test field with pytree_node=False."""
        f = field(pytree_node=False)
        assert f.metadata['pytree_node'] is False

    def test_field_with_default(self):
        """Test field with default value."""
        f = field(default=42)
        assert f.default == 42
        assert f.metadata['pytree_node'] is True

    def test_field_with_metadata(self):
        """Test field preserves additional metadata."""
        f = field(pytree_node=False, metadata={'custom': 'data'})
        assert f.metadata['pytree_node'] is False
        assert f.metadata['custom'] == 'data'


class TestDataclass:
    """Test the dataclass decorator."""

    def test_basic_dataclass(self):
        """Test basic dataclass creation."""

        @dataclass
        class Point:
            x: float
            y: float

        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0

    def test_dataclass_is_frozen(self):
        """Test that dataclasses are frozen by default."""

        @dataclass
        class Point:
            x: float
            y: float

        p = Point(1.0, 2.0)
        with pytest.raises(Exception):  # Should be immutable
            p.x = 3.0

    def test_dataclass_replace_method(self):
        """Test the replace method."""

        @dataclass
        class Point:
            x: float
            y: float

        p1 = Point(1.0, 2.0)
        p2 = p1.replace(x=3.0)
        assert p1.x == 1.0
        assert p2.x == 3.0
        assert p2.y == 2.0

    def test_dataclass_with_defaults(self):
        """Test dataclass with default values."""

        @dataclass
        class Config:
            learning_rate: float = 0.001
            batch_size: int = 32
            name: str = field(default="default", pytree_node=False)

        c1 = Config()
        assert c1.learning_rate == 0.001
        assert c1.batch_size == 32
        assert c1.name == "default"

        c2 = Config(learning_rate=0.01)
        assert c2.learning_rate == 0.01

    def test_dataclass_pytree_behavior(self):
        """Test that dataclass works as JAX pytree."""

        @dataclass
        class Model:
            weights: jax.Array
            bias: jax.Array
            name: str = field(pytree_node=False, default="model")

        weights = jnp.ones((3, 3))
        bias = jnp.zeros(3)
        model = Model(weights=weights, bias=bias)

        # Test tree_map
        model2 = jax.tree_util.tree_map(lambda x: x * 2, model)
        assert jnp.allclose(model2.weights, weights * 2)
        assert jnp.allclose(model2.bias, bias * 2)
        assert model2.name == "model"  # Should not be affected

        # Test tree_leaves
        leaves = jax.tree_util.tree_leaves(model)
        assert len(leaves) == 2  # Only weights and bias

    def test_dataclass_with_jax_transformations(self):
        """Test dataclass with JAX transformations."""

        @dataclass
        class Linear:
            weight: jax.Array
            bias: jax.Array

        layer = Linear(
            weight=jnp.ones((4, 3)),
            bias=jnp.zeros(4)
        )

        # Test with jit
        @jax.jit
        def apply(layer, x):
            return jnp.dot(x, layer.weight.T) + layer.bias

        x = jnp.ones(3)
        y = apply(layer, x)
        assert y.shape == (4,)

        # Test with grad
        def loss_fn(layer):
            return jnp.sum(layer.weight ** 2) + jnp.sum(layer.bias ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(layer)
        assert grads.weight.shape == layer.weight.shape
        assert grads.bias.shape == layer.bias.shape

    def test_dataclass_no_double_decoration(self):
        """Test that dataclass decorator is idempotent."""

        @dataclass
        @dataclass  # Should not cause issues
        class Point:
            x: float
            y: float

        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert hasattr(Point, '_brainstate_dataclass')

    def test_dataclass_supports_bare_parentheses(self):
        """Allow @dataclass() usage like the stdlib decorator."""
        @dataclass()
        class Point:
            x: float

        p = Point(1.0)

        assert p.x == 1.0

    def test_field_does_not_mutate_caller_metadata(self):
        """Copy provided metadata before adding the pytree marker."""
        metadata = {'custom': 'data'}

        field(pytree_node=False, metadata=metadata)

        assert metadata == {'custom': 'data'}

    def test_field_rejects_conflicting_pytree_metadata(self):
        """Reject metadata that disagrees with the explicit pytree_node flag."""
        with pytest.raises(ValueError, match='pytree_node'):
            field(pytree_node=False, metadata={'pytree_node': True})

    def test_is_dataclass_rejects_spoofed_marker(self):
        """Do not treat arbitrary marker attributes as BrainState dataclasses."""
        class Spoofed:
            _brainstate_dataclass = True

        assert not is_dataclass(Spoofed)


class TestPyTreeNode:
    """Test the PyTreeNode base class."""

    def test_pytreenode_subclass(self):
        """Test creating a PyTreeNode subclass."""

        class Layer(PyTreeNode):
            weights: jax.Array
            bias: jax.Array
            activation: str = field(pytree_node=False, default="relu")

        layer = Layer(
            weights=jnp.ones((4, 4)),
            bias=jnp.zeros(4)
        )
        assert layer.activation == "relu"
        assert jnp.allclose(layer.weights, jnp.ones((4, 4)))

    def test_pytreenode_is_frozen(self):
        """Test that PyTreeNode subclasses are frozen."""

        class Layer(PyTreeNode):
            weights: jax.Array

        layer = Layer(weights=jnp.ones(3))
        with pytest.raises(Exception):
            layer.weights = jnp.zeros(3)

    def test_pytreenode_replace(self):
        """Test replace method on PyTreeNode."""

        class Layer(PyTreeNode):
            weights: jax.Array
            bias: jax.Array

        layer1 = Layer(weights=jnp.ones(3), bias=jnp.zeros(3))
        layer2 = layer1.replace(weights=jnp.ones(3) * 2)
        assert jnp.allclose(layer2.weights, jnp.ones(3) * 2)
        assert jnp.allclose(layer2.bias, jnp.zeros(3))

    def test_pytreenode_with_jax(self):
        """Test PyTreeNode with JAX transformations."""

        class MLP(PyTreeNode):
            layer1: Any
            layer2: Any

        class Linear(PyTreeNode):
            weight: jax.Array
            bias: jax.Array

        mlp = MLP(
            layer1=Linear(weight=jnp.ones((4, 3)), bias=jnp.zeros(4)),
            layer2=Linear(weight=jnp.ones((2, 4)), bias=jnp.zeros(2))
        )

        # Test tree_map
        mlp2 = jax.tree_util.tree_map(lambda x: x * 2, mlp)
        assert jnp.allclose(mlp2.layer1.weight, mlp.layer1.weight * 2)

        # Test with grad
        def loss_fn(model):
            return jnp.sum(model.layer1.weight ** 2) + jnp.sum(model.layer2.weight ** 2)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(mlp)
        assert grads.layer1.weight.shape == mlp.layer1.weight.shape


class TestFrozenDict:
    """Test the FrozenDict class."""

    def test_frozendict_creation(self):
        """Test creating FrozenDict."""
        # From dict
        fd1 = FrozenDict({'a': 1, 'b': 2})
        assert fd1['a'] == 1
        assert fd1['b'] == 2

        # From kwargs
        fd2 = FrozenDict(a=1, b=2)
        assert fd2['a'] == 1

        # From items
        fd3 = FrozenDict([('a', 1), ('b', 2)])
        assert fd3['a'] == 1

    def test_frozendict_immutability(self):
        """Test that FrozenDict is immutable."""
        fd = FrozenDict({'a': 1})

        with pytest.raises(TypeError):
            fd['a'] = 2

        with pytest.raises(TypeError):
            fd['c'] = 3

        with pytest.raises(TypeError):
            del fd['a']

    def test_frozendict_basic_operations(self):
        """Test basic dictionary operations."""
        fd = FrozenDict({'a': 1, 'b': 2, 'c': 3})

        # Contains
        assert 'a' in fd
        assert 'd' not in fd

        # Length
        assert len(fd) == 3

        # Iteration
        keys = list(fd)
        assert set(keys) == {'a', 'b', 'c'}

        # Get
        assert fd.get('a') == 1
        assert fd.get('d') is None
        assert fd.get('d', 10) == 10

    def test_frozendict_views(self):
        """Test dictionary views."""
        fd = FrozenDict({'a': 1, 'b': 2})

        # Keys view
        keys = fd.keys()
        assert set(keys) == {'a', 'b'}
        assert 'FrozenDict.keys' in repr(keys)

        # Values view
        values = fd.values()
        assert set(values) == {1, 2}
        assert 'FrozenDict.values' in repr(values)

        # Items view
        items = list(fd.items())
        assert len(items) == 2
        assert ('a', 1) in items

    def test_frozendict_copy(self):
        """Test copy method."""
        fd1 = FrozenDict({'a': 1, 'b': 2})

        # Copy without changes
        fd2 = fd1.copy()
        assert fd2 == fd1
        assert fd2 is not fd1

        # Copy with updates
        fd3 = fd1.copy({'c': 3, 'a': 10})
        assert fd3['a'] == 10
        assert fd3['b'] == 2
        assert fd3['c'] == 3
        assert fd1['a'] == 1  # Original unchanged

    def test_frozendict_pop(self):
        """Test pop method."""
        fd1 = FrozenDict({'a': 1, 'b': 2, 'c': 3})

        fd2, value = fd1.pop('b')
        assert value == 2
        assert 'b' not in fd2
        assert len(fd2) == 2
        assert 'b' in fd1  # Original unchanged

        # Pop non-existent key
        with pytest.raises(KeyError):
            fd1.pop('d')

    def test_frozendict_nested(self):
        """Test nested FrozenDict."""
        fd = FrozenDict({
            'a': 1,
            'b': {'c': 2, 'd': {'e': 3}}
        })

        # Access nested values
        assert fd['b']['c'] == 2
        assert fd['b']['d']['e'] == 3

        # Nested values are also FrozenDict
        assert isinstance(fd['b'], FrozenDict)
        assert isinstance(fd['b']['d'], FrozenDict)

    def test_frozendict_hash(self):
        """Test FrozenDict hashing."""
        fd1 = FrozenDict({'a': 1, 'b': 2})
        fd2 = FrozenDict({'a': 1, 'b': 2})
        fd3 = FrozenDict({'a': 1, 'b': 3})

        # Equal dicts have same hash
        assert hash(fd1) == hash(fd2)

        # Can be used in sets
        s = {fd1, fd2, fd3}
        assert len(s) == 2

    def test_frozendict_equality(self):
        """Test FrozenDict equality."""
        fd1 = FrozenDict({'a': 1, 'b': 2})
        fd2 = FrozenDict({'a': 1, 'b': 2})
        fd3 = FrozenDict({'a': 1, 'b': 3})
        d = {'a': 1, 'b': 2}

        assert fd1 == fd2
        assert fd1 != fd3
        assert fd1 == d
        assert fd1 != "not a dict"

    def test_frozendict_pickle(self):
        """Test FrozenDict pickling."""
        fd = FrozenDict({'a': 1, 'b': {'c': 2}})

        # Pickle and unpickle
        pickled = pickle.dumps(fd)
        fd2 = pickle.loads(pickled)

        assert fd == fd2
        assert fd['b']['c'] == fd2['b']['c']

    def test_frozendict_pretty_repr(self):
        """Test pretty representation."""
        fd = FrozenDict({'a': 1, 'b': {'c': 2}})
        repr_str = fd.pretty_repr()

        assert 'FrozenDict' in repr_str
        assert "'a': 1" in repr_str
        assert "'c': 2" in repr_str

    def test_frozendict_as_pytree(self):
        """Test FrozenDict as JAX pytree."""
        fd = FrozenDict({'a': jnp.ones(3), 'b': jnp.zeros(2)})

        # Tree map
        fd2 = jax.tree_util.tree_map(lambda x: x * 2, fd)
        assert jnp.allclose(fd2['a'], jnp.ones(3) * 2)
        assert jnp.allclose(fd2['b'], jnp.zeros(2))

        # Tree leaves
        leaves = jax.tree_util.tree_leaves(fd)
        assert len(leaves) == 2

        # Tree flatten and unflatten
        values, treedef = jax.tree_util.tree_flatten(fd)
        fd3 = jax.tree_util.tree_unflatten(treedef, values)
        assert fd == fd3


class TestUtilityFunctions:
    """Test utility functions."""

    def test_freeze(self):
        """Test freeze function."""
        # Regular dict
        d = {'a': 1, 'b': {'c': 2}}
        fd = freeze(d)
        assert isinstance(fd, FrozenDict)
        assert fd['a'] == 1
        assert isinstance(fd['b'], FrozenDict)

        # Already frozen
        fd2 = freeze(fd)
        assert fd2 is fd

    def test_unfreeze(self):
        """Test unfreeze function."""
        # FrozenDict
        fd = FrozenDict({'a': 1, 'b': {'c': 2}})
        d = unfreeze(fd)
        assert isinstance(d, dict)
        assert not isinstance(d, FrozenDict)
        assert d['a'] == 1
        assert isinstance(d['b'], dict)

        # Regular dict
        d2 = {'a': 1}
        d3 = unfreeze(d2)
        assert d3 == d2
        assert d3 is not d2  # Should be a copy

        # Non-dict
        assert unfreeze(42) == 42

    def test_copy_function(self):
        """Test copy function."""
        # FrozenDict
        fd1 = FrozenDict({'a': 1})
        fd2 = copy(fd1, {'b': 2})
        assert isinstance(fd2, FrozenDict)
        assert fd2['a'] == 1
        assert fd2['b'] == 2

        # Regular dict
        d1 = {'a': 1}
        d2 = copy(d1, {'b': 2})
        assert isinstance(d2, dict)
        assert not isinstance(d2, FrozenDict)
        assert d2['a'] == 1
        assert d2['b'] == 2
        assert d1 == {'a': 1}  # Original unchanged

        # Invalid type
        with pytest.raises(TypeError):
            copy([1, 2, 3])

    def test_pop_function(self):
        """Test pop function."""
        # FrozenDict
        fd1 = FrozenDict({'a': 1, 'b': 2})
        fd2, value = pop(fd1, 'a')
        assert isinstance(fd2, FrozenDict)
        assert value == 1
        assert 'a' not in fd2
        assert 'a' in fd1

        # Regular dict
        d1 = {'a': 1, 'b': 2}
        d2, value = pop(d1, 'a')
        assert isinstance(d2, dict)
        assert value == 1
        assert 'a' not in d2
        assert 'a' in d1

        # Invalid type
        with pytest.raises(TypeError):
            pop([1, 2, 3], 0)

    def test_pretty_repr_function(self):
        """Test pretty_repr function."""
        # FrozenDict
        fd = FrozenDict({'a': 1, 'b': {'c': 2}})
        s = pretty_repr(fd)
        assert 'FrozenDict' in s

        # Regular dict
        d = {'a': 1, 'b': {'c': 2}}
        s = pretty_repr(d)
        assert 'a' in s
        assert 'c' in s

        # Other type
        s = pretty_repr([1, 2, 3])
        assert s == "[1, 2, 3]"


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_nested_structures(self):
        """Test complex nested structures."""

        @dataclass
        class Config:
            hyperparams: FrozenDict
            metadata: dict = field(pytree_node=False)

        class Model(PyTreeNode):
            config: Config
            weights: jax.Array

        config = Config(
            hyperparams=FrozenDict({'lr': 0.001, 'batch_size': 32}),
            metadata={'version': '1.0'}
        )
        model = Model(
            config=config,
            weights=jnp.ones((4, 4))
        )

        # Test tree operations
        model2 = jax.tree_util.tree_map(lambda x: x * 2 if isinstance(x, jax.Array) else x, model)
        assert jnp.allclose(model2.weights, model.weights * 2)
        assert model2.config.hyperparams['lr'] == 0.001
        assert model2.config.metadata['version'] == '1.0'

    def test_jax_transformations_integration(self):
        """Test integration with various JAX transformations."""

        @dataclass
        class State:
            params: FrozenDict
            step: int = field(pytree_node=False, default=0)

        state = State(
            params=FrozenDict({
                'w': jnp.ones((3, 3)),
                'b': jnp.zeros(3)
            })
        )

        # JIT compilation
        @jax.jit
        def update(state, grad):
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - 0.01 * g,
                state.params,
                grad
            )
            return state.replace(
                params=new_params,
                step=state.step + 1
            )

        grad = FrozenDict({'w': jnp.ones((3, 3)), 'b': jnp.ones(3)})
        new_state = update(state, grad)
        assert new_state.step == 1
        assert jnp.allclose(new_state.params['w'], state.params['w'] - 0.01)

        # VMAP
        @jax.vmap
        def batch_process(params, x):
            return jnp.dot(x, params['w']) + params['b']

        batch_params = jax.tree_util.tree_map(
            lambda x: jnp.stack([x, x * 2]),
            state.params
        )
        batch_x = jnp.ones((2, 3))
        result = batch_process(batch_params, batch_x)
        assert result.shape == (2, 3)


class TestStructSemantics(unittest.TestCase):
    """Frozen dataclass behavior and pytree_node field separation."""

    def test_pytreenode_is_frozen(self):
        """Assign to a PyTreeNode field after construction and expect a raise."""
        class Layer(PyTreeNode):
            weights: jax.Array

        layer = Layer(weights=jnp.ones(3))
        with self.assertRaises(Exception):
            layer.weights = jnp.zeros(3)

    def test_replace_does_not_mutate_original(self):
        """Return a new instance from replace, leaving the original unchanged."""
        class Layer(PyTreeNode):
            weights: jax.Array

        a = Layer(weights=jnp.ones(3))
        b = a.replace(weights=jnp.ones(3) * 2)
        self.assertTrue(bool(jnp.allclose(a.weights, 1.0)))
        self.assertTrue(bool(jnp.allclose(b.weights, 2.0)))

    def test_pytreenode_is_jax_pytree(self):
        """Flatten and unflatten a PyTreeNode through jax.tree."""
        class Layer(PyTreeNode):
            weights: jax.Array

        a = Layer(weights=jnp.arange(3.0))
        leaves, treedef = jax.tree.flatten(a)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        self.assertTrue(bool(jnp.allclose(rebuilt.weights, a.weights)))

    def test_static_field_not_a_leaf(self):
        """Exclude a pytree_node=False field from jax.tree leaves and survive flatten/unflatten."""
        class Layer(PyTreeNode):
            weights: jax.Array
            name: str = field(pytree_node=False, default="layer")

        a = Layer(weights=jnp.ones(3), name="static-name")
        leaves = jax.tree_util.tree_leaves(a)
        # Only the array weights should be a leaf; the static name must not appear.
        self.assertEqual(len(leaves), 1)
        self.assertNotIn("static-name", leaves)
        # The static field survives a flatten/unflatten roundtrip unchanged.
        flat, treedef = jax.tree_util.tree_flatten(a)
        rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
        self.assertEqual(rebuilt.name, "static-name")
        self.assertTrue(bool(jnp.allclose(rebuilt.weights, a.weights)))

    def test_pytreenode_init_stub_raises(self):
        """Raise NotImplementedError when instantiating PyTreeNode directly."""
        with self.assertRaises(NotImplementedError):
            PyTreeNode()

    def test_pytreenode_replace_stub_raises(self):
        """Raise NotImplementedError when calling the unbound PyTreeNode.replace stub."""
        with self.assertRaises(NotImplementedError):
            PyTreeNode.replace(object())

    def test_register_pytree_fallback_path(self):
        """Drive the register_pytree_with_keys fallback closures when register_dataclass is absent."""
        import brainstate.util.struct as struct_mod

        had_attr = hasattr(jax.tree_util, 'register_dataclass')
        saved = jax.tree_util.register_dataclass if had_attr else None
        if had_attr:
            del jax.tree_util.register_dataclass
        try:
            @struct_mod.dataclass
            class Fallback:
                weights: jax.Array
                name: str = field(pytree_node=False, default="fb")

            obj = Fallback(weights=jnp.ones(2))

            # tree_flatten / tree_unflatten exercise flatten_fn and unflatten_fn.
            leaves, treedef = jax.tree_util.tree_flatten(obj)
            self.assertEqual(len(leaves), 1)
            rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
            self.assertEqual(rebuilt.name, "fb")
            self.assertTrue(bool(jnp.allclose(rebuilt.weights, obj.weights)))

            # tree_flatten_with_path exercises flatten_with_keys_fn.
            with_keys, _ = jax.tree_util.tree_flatten_with_path(obj)
            self.assertEqual(len(with_keys), 1)
        finally:
            if had_attr:
                jax.tree_util.register_dataclass = saved


class TestFrozenDictCoverage(unittest.TestCase):
    """Cover FrozenDict helpers, hashing, repr, and pretty-print edge cases."""

    def test_frozendict_immutable_and_roundtrip(self):
        """Reject mutation and support a freeze/unfreeze roundtrip."""
        fd = freeze({'a': 1, 'b': 2})
        self.assertIsInstance(fd, FrozenDict)
        with self.assertRaises(Exception):
            fd['a'] = 99
        self.assertEqual(unfreeze(fd), {'a': 1, 'b': 2})

    def test_deep_freeze_unwraps_nested_frozendict(self):
        """Unwrap a nested FrozenDict value to its underlying data during construction."""
        inner = FrozenDict({'x': 1})
        # A FrozenDict held as a value drives _deep_freeze's FrozenDict branch.
        outer = FrozenDict({'a': inner})
        self.assertEqual(outer['a']['x'], 1)
        # The nested value is re-frozen as plain data internally.
        self.assertNotIsInstance(outer._data['a'], FrozenDict)
        self.assertEqual(outer._data['a'], {'x': 1})

    def test_repr_uses_pretty_repr(self):
        """Produce a FrozenDict-prefixed string from __repr__."""
        fd = FrozenDict({'a': 1})
        r = repr(fd)
        self.assertIn('FrozenDict', r)
        self.assertIn("'a': 1", r)

    def test_hash_with_nested_dict_value(self):
        """Hash a FrozenDict whose value is a plain dict, wrapping it during hashing."""
        # Construct with _data holding a raw dict so __hash__ hits the dict branch.
        fd = FrozenDict({'a': {'b': 1}})
        h1 = hash(fd)
        # Cached hash returns the same value on a second call.
        self.assertEqual(h1, hash(fd))
        fd2 = FrozenDict({'a': {'b': 1}})
        self.assertEqual(hash(fd), hash(fd2))

    def test_hash_with_mixed_key_types(self):
        """Hash mappings with valid but mutually incomparable key types."""
        fd1 = FrozenDict({1: 'int', '1': 'str'})
        fd2 = FrozenDict({'1': 'str', 1: 'int'})

        self.assertEqual(hash(fd1), hash(fd2))
        self.assertEqual(fd1, fd2)

    def test_jax_flatten_with_mixed_key_types(self):
        """Flatten mappings with valid but mutually incomparable key types."""
        fd = FrozenDict({1: jnp.array(1), '1': jnp.array(2)})

        leaves = jax.tree_util.tree_leaves(fd)

        self.assertEqual(len(leaves), 2)

    def test_pretty_repr_empty_frozendict(self):
        """Render an empty FrozenDict as FrozenDict({})."""
        fd = FrozenDict({})
        self.assertEqual(fd.pretty_repr(), 'FrozenDict({})')

    def test_pretty_repr_empty_nested_dict(self):
        """Render a nested empty dict as {} inside FrozenDict.pretty_repr."""
        fd = FrozenDict({'a': {}})
        r = fd.pretty_repr()
        self.assertIn('FrozenDict', r)
        self.assertIn("'a': {}", r)

    def test_module_pretty_repr_empty_nested_dict(self):
        """Render a nested empty dict as {} via the module-level pretty_repr."""
        r = pretty_repr({'a': {}})
        self.assertIn("'a': {}", r)

    def test_module_pretty_repr_empty_dict(self):
        """Render an empty top-level dict as {} via the module-level pretty_repr."""
        self.assertEqual(pretty_repr({}), '{}')

    def test_copy_pop_items_equality(self):
        """Cover copy, pop, items, and equality on FrozenDict."""
        fd = FrozenDict({'a': 1, 'b': 2})
        # copy with replacement
        fd2 = fd.copy({'a': 10, 'c': 3})
        self.assertEqual(fd2['a'], 10)
        self.assertEqual(fd2['c'], 3)
        self.assertEqual(fd['a'], 1)
        # pop returns new dict and value
        fd3, val = fd.pop('a')
        self.assertEqual(val, 1)
        self.assertNotIn('a', fd3)
        # items view
        self.assertEqual(dict(fd.items()), {'a': 1, 'b': 2})
        # equality against dict and FrozenDict
        self.assertEqual(fd, {'a': 1, 'b': 2})
        self.assertEqual(fd, FrozenDict({'a': 1, 'b': 2}))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
