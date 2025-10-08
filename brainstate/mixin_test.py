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
Comprehensive tests for brainstate.mixin module.

This test suite covers all functionality in the mixin module including:
- Base mixin classes
- Parameter description and deferred instantiation
- Type utilities (JointTypes, OneOfTypes)
- Mode system (Mode, JointMode, Training, Batching)
- Helper utilities (hashable, not_implemented, etc.)
"""

import unittest

import jax.numpy as jnp

import brainstate


class TestHashableFunction(unittest.TestCase):
    """Test the hashable utility function."""

    def test_hashable_primitives(self):
        """Test hashable with primitive types."""
        self.assertTrue(brainstate.mixin.hashable(42))
        self.assertTrue(brainstate.mixin.hashable(3.14))
        self.assertTrue(brainstate.mixin.hashable("string"))
        self.assertTrue(brainstate.mixin.hashable(True))
        self.assertTrue(brainstate.mixin.hashable(None))

    def test_hashable_tuples(self):
        """Test hashable with tuples."""
        self.assertTrue(brainstate.mixin.hashable((1, 2, 3)))
        self.assertTrue(brainstate.mixin.hashable(("a", "b")))
        self.assertTrue(brainstate.mixin.hashable(()))

    def test_non_hashable_types(self):
        """Test non-hashable types."""
        self.assertFalse(brainstate.mixin.hashable([1, 2, 3]))
        self.assertFalse(brainstate.mixin.hashable({"key": "value"}))
        self.assertFalse(brainstate.mixin.hashable({1, 2, 3}))
        self.assertFalse(brainstate.mixin.hashable(jnp.array([1, 2, 3])))


class TestMixin(unittest.TestCase):
    """Test the base Mixin class."""

    def test_mixin_exists(self):
        """Test that Mixin class exists."""
        self.assertTrue(brainstate.mixin.Mixin)

    def test_mixin_inheritance(self):
        """Test creating a custom mixin."""

        class LoggingMixin(brainstate.mixin.Mixin):
            def log(self, message):
                return f"[LOG] {message}"

        class Component(LoggingMixin):
            pass

        comp = Component()
        self.assertEqual(comp.log("test"), "[LOG] test")

    def test_mixin_multiple_inheritance(self):
        """Test multiple mixin inheritance."""

        class MixinA(brainstate.mixin.Mixin):
            def method_a(self):
                return "A"

        class MixinB(brainstate.mixin.Mixin):
            def method_b(self):
                return "B"

        class Component(MixinA, MixinB):
            pass

        comp = Component()
        self.assertEqual(comp.method_a(), "A")
        self.assertEqual(comp.method_b(), "B")


class TestParamDesc(unittest.TestCase):
    """Test ParamDesc mixin and ParamDescriber."""

    def test_param_desc_basic(self):
        """Test basic ParamDesc functionality."""

        class Network(brainstate.mixin.ParamDesc):
            def __init__(self, size, learning_rate=0.01):
                self.size = size
                self.learning_rate = learning_rate

        # Test desc method exists
        self.assertTrue(hasattr(Network, 'desc'))

        # Create a descriptor
        desc = Network.desc(size=100)
        self.assertIsInstance(desc, brainstate.mixin.ParamDescriber)

    def test_param_describer_instantiation(self):
        """Test ParamDescriber can create instances."""

        class Network(brainstate.mixin.ParamDesc):
            def __init__(self, size, learning_rate=0.01):
                self.size = size
                self.learning_rate = learning_rate

        desc = Network.desc(size=100, learning_rate=0.001)

        # Create instances
        net1 = desc()
        self.assertEqual(net1.size, 100)
        self.assertEqual(net1.learning_rate, 0.001)

        # Create with overrides
        net2 = desc(learning_rate=0.005)
        self.assertEqual(net2.size, 100)
        self.assertEqual(net2.learning_rate, 0.005)

    def test_param_describer_init_method(self):
        """Test ParamDescriber.init() method."""

        class Model(brainstate.mixin.ParamDesc):
            def __init__(self, value):
                self.value = value

        desc = Model.desc(value=42)
        instance = desc.init()
        self.assertEqual(instance.value, 42)

    def test_param_describer_identifier(self):
        """Test ParamDescriber identifier property."""

        class Model(brainstate.mixin.ParamDesc):
            def __init__(self, x, y=10):
                self.x = x
                self.y = y

        desc = Model.desc(x=5, y=20)
        identifier = desc.identifier

        # Identifier should be a tuple
        self.assertIsInstance(identifier, tuple)
        self.assertEqual(len(identifier), 3)
        self.assertEqual(identifier[0], Model)

        # Identifier should be read-only
        with self.assertRaises(AttributeError):
            desc.identifier = "new"

    def test_param_describer_class_getitem(self):
        """Test ParamDescriber[Class] notation."""

        class Model:
            def __init__(self, value):
                self.value = value

        desc = brainstate.mixin.ParamDescriber[Model]
        self.assertIsInstance(desc, brainstate.mixin.ParamDescriber)
        self.assertEqual(desc.cls, Model)

    def test_no_subclass_meta(self):
        """Test that ParamDescriber cannot be subclassed."""

        with self.assertRaises(TypeError):
            class CustomDescriber(brainstate.mixin.ParamDescriber):
                pass


class TestHashableDict(unittest.TestCase):
    """Test HashableDict class."""

    def test_hashable_dict_basic(self):
        """Test basic HashableDict functionality."""
        d = brainstate.mixin.HashableDict({"a": 1, "b": 2})
        h = hash(d)
        self.assertIsInstance(h, int)

    def test_hashable_dict_with_arrays(self):
        """Test HashableDict with non-hashable values."""
        d = brainstate.mixin.HashableDict({
            "array": jnp.array([1, 2, 3]),
            "value": 42
        })
        h = hash(d)
        self.assertIsInstance(h, int)

    def test_hashable_dict_consistency(self):
        """Test that equal dicts have equal hashes."""
        d1 = brainstate.mixin.HashableDict({"a": 1, "b": 2})
        d2 = brainstate.mixin.HashableDict({"b": 2, "a": 1})
        self.assertEqual(hash(d1), hash(d2))

    def test_hashable_dict_usable_as_key(self):
        """Test that HashableDict can be used as dict key."""
        d = brainstate.mixin.HashableDict({"x": 10})
        cache = {d: "result"}
        self.assertEqual(cache[d], "result")


class TestJointTypes(unittest.TestCase):
    """Test JointTypes functionality."""

    def test_joint_types_basic(self):
        """Test basic JointTypes creation."""

        class A:
            pass

        class B:
            pass

        JointAB = brainstate.mixin.JointTypes(A, B)
        self.assertIsNotNone(JointAB)

    def test_joint_types_isinstance(self):
        """Test isinstance with JointTypes."""

        class Serializable:
            def save(self):
                pass

        class Visualizable:
            def plot(self):
                pass

        Combined = brainstate.mixin.JointTypes(Serializable, Visualizable)

        class Model(Serializable, Visualizable):
            def save(self):
                return "saved"

            def plot(self):
                return "plotted"

        model = Model()
        self.assertTrue(isinstance(model, Combined))

    def test_joint_types_issubclass(self):
        """Test issubclass with JointTypes."""

        class A:
            pass

        class B:
            pass

        JointAB = brainstate.mixin.JointTypes(A, B)

        class C(A, B):
            pass

        self.assertTrue(issubclass(C, JointAB))

    def test_joint_types_single_type(self):
        """Test JointTypes with single type returns that type."""

        class A:
            pass

        result = brainstate.mixin.JointTypes(A)
        self.assertEqual(result, A)

    def test_joint_types_no_types(self):
        """Test JointTypes with no types raises error."""
        with self.assertRaises(TypeError):
            brainstate.mixin.JointTypes()

    def test_joint_types_removes_duplicates(self):
        """Test that JointTypes removes duplicate types."""

        class A:
            pass

        # Should handle duplicates gracefully
        JointA = brainstate.mixin.JointTypes(A, A, A)
        self.assertEqual(JointA, A)


class TestOneOfTypes(unittest.TestCase):
    """Test OneOfTypes functionality."""

    def test_one_of_types_basic(self):
        """Test basic OneOfTypes creation."""
        IntOrFloat = brainstate.mixin.OneOfTypes(int, float)
        self.assertIsNotNone(IntOrFloat)

    def test_one_of_types_isinstance(self):
        """Test isinstance with OneOfTypes."""
        NumType = brainstate.mixin.OneOfTypes(int, float)

        self.assertTrue(isinstance(42, NumType))
        self.assertTrue(isinstance(3.14, NumType))
        self.assertFalse(isinstance("hello", NumType))

    def test_one_of_types_single_type(self):
        """Test OneOfTypes with single type returns that type."""
        result = brainstate.mixin.OneOfTypes(int)
        self.assertEqual(result, int)

    def test_one_of_types_no_types(self):
        """Test OneOfTypes with no types raises error."""
        with self.assertRaises(TypeError):
            brainstate.mixin.OneOfTypes()

    def test_one_of_types_with_none(self):
        """Test OneOfTypes with None for optional types."""
        MaybeInt = brainstate.mixin.OneOfTypes(int, type(None))

        self.assertTrue(isinstance(42, MaybeInt))
        self.assertTrue(isinstance(None, MaybeInt))
        self.assertFalse(isinstance("hello", MaybeInt))



class TestNotImplemented(unittest.TestCase):
    """Test not_implemented decorator."""

    def test_not_implemented_decorator(self):
        """Test not_implemented decorator marks functions."""

        @brainstate.mixin.not_implemented
        def my_function():
            pass

        self.assertTrue(hasattr(my_function, 'not_implemented'))
        self.assertTrue(my_function.not_implemented)

    def test_not_implemented_raises(self):
        """Test not_implemented decorator raises error when called."""

        @brainstate.mixin.not_implemented
        def my_function():
            pass

        with self.assertRaises(NotImplementedError) as cm:
            my_function()

        self.assertIn("my_function", str(cm.exception))


class TestMode(unittest.TestCase):
    """Test Mode base class."""

    def test_mode_creation(self):
        """Test basic Mode creation."""
        mode = brainstate.mixin.Mode()
        self.assertIsNotNone(mode)

    def test_mode_repr(self):
        """Test Mode string representation."""
        mode = brainstate.mixin.Mode()
        self.assertEqual(repr(mode), "Mode")

    def test_mode_equality(self):
        """Test Mode equality comparison."""
        mode1 = brainstate.mixin.Mode()
        mode2 = brainstate.mixin.Mode()
        self.assertEqual(mode1, mode2)

    def test_mode_is_a(self):
        """Test Mode.is_a() method."""
        mode = brainstate.mixin.Mode()
        self.assertTrue(mode.is_a(brainstate.mixin.Mode))
        self.assertFalse(mode.is_a(brainstate.mixin.Training))

    def test_mode_has(self):
        """Test Mode.has() method."""
        mode = brainstate.mixin.Mode()
        self.assertTrue(mode.has(brainstate.mixin.Mode))
        self.assertFalse(mode.has(brainstate.mixin.Training))

    def test_custom_mode(self):
        """Test creating custom mode."""

        class CustomMode(brainstate.mixin.Mode):
            def __init__(self, value):
                self.value = value

        mode = CustomMode(42)
        self.assertEqual(mode.value, 42)
        self.assertTrue(mode.has(brainstate.mixin.Mode))


class TestTraining(unittest.TestCase):
    """Test Training mode."""

    def test_training_creation(self):
        """Test Training mode creation."""
        training = brainstate.mixin.Training()
        self.assertIsNotNone(training)

    def test_training_is_mode(self):
        """Test Training is a Mode."""
        training = brainstate.mixin.Training()
        self.assertTrue(training.has(brainstate.mixin.Mode))

    def test_training_is_a(self):
        """Test Training.is_a() method."""
        training = brainstate.mixin.Training()
        self.assertTrue(training.is_a(brainstate.mixin.Training))
        self.assertFalse(training.is_a(brainstate.mixin.Batching))

    def test_training_has(self):
        """Test Training.has() method."""
        training = brainstate.mixin.Training()
        self.assertTrue(training.has(brainstate.mixin.Training))
        self.assertFalse(training.has(brainstate.mixin.Batching))

    def test_training_joint_types(self):
        """Test Training with JointTypes."""
        training = brainstate.mixin.Training()
        self.assertTrue(training.is_a(brainstate.mixin.JointTypes(brainstate.mixin.Training)))
        self.assertTrue(training.has(brainstate.mixin.JointTypes(brainstate.mixin.Training)))


class TestBatching(unittest.TestCase):
    """Test Batching mode."""

    def test_batching_creation(self):
        """Test Batching mode creation."""
        batching = brainstate.mixin.Batching()
        self.assertIsNotNone(batching)

    def test_batching_default_params(self):
        """Test Batching default parameters."""
        batching = brainstate.mixin.Batching()
        self.assertEqual(batching.batch_size, 1)
        self.assertEqual(batching.batch_axis, 0)

    def test_batching_custom_params(self):
        """Test Batching with custom parameters."""
        batching = brainstate.mixin.Batching(batch_size=32, batch_axis=1)
        self.assertEqual(batching.batch_size, 32)
        self.assertEqual(batching.batch_axis, 1)

    def test_batching_repr(self):
        """Test Batching string representation."""
        batching = brainstate.mixin.Batching(batch_size=64, batch_axis=0)
        self.assertIn("64", repr(batching))
        self.assertIn("0", repr(batching))

    def test_batching_is_mode(self):
        """Test Batching is a Mode."""
        batching = brainstate.mixin.Batching()
        self.assertTrue(batching.has(brainstate.mixin.Mode))

    def test_batching_is_a(self):
        """Test Batching.is_a() method."""
        batching = brainstate.mixin.Batching()
        self.assertTrue(batching.is_a(brainstate.mixin.Batching))
        self.assertFalse(batching.is_a(brainstate.mixin.Training))

    def test_batching_has(self):
        """Test Batching.has() method."""
        batching = brainstate.mixin.Batching()
        self.assertTrue(batching.has(brainstate.mixin.Batching))
        self.assertFalse(batching.has(brainstate.mixin.Training))


class TestJointMode(unittest.TestCase):
    """Test JointMode functionality."""

    def test_joint_mode_creation(self):
        """Test JointMode creation."""
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching()
        joint = brainstate.mixin.JointMode(training, batching)
        self.assertIsNotNone(joint)

    def test_joint_mode_repr(self):
        """Test JointMode string representation."""
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching(batch_size=32)
        joint = brainstate.mixin.JointMode(training, batching)

        repr_str = repr(joint)
        self.assertIn("JointMode", repr_str)
        self.assertIn("Training", repr_str)
        self.assertIn("Batching", repr_str)

    def test_joint_mode_has(self):
        """Test JointMode.has() method."""
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching()
        joint = brainstate.mixin.JointMode(training, batching)

        self.assertTrue(joint.has(brainstate.mixin.Training))
        self.assertTrue(joint.has(brainstate.mixin.Batching))
        self.assertTrue(joint.has(brainstate.mixin.Mode))

    def test_joint_mode_is_a(self):
        """Test JointMode.is_a() method."""
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching()
        joint = brainstate.mixin.JointMode(training, batching)

        # JointMode.is_a() works by checking if the JointTypes of the mode types
        # matches the expected type. This is a complex comparison.
        # For practical use, test that it correctly identifies single types
        self.assertFalse(joint.is_a(brainstate.mixin.Training))  # Not just Training
        self.assertFalse(joint.is_a(brainstate.mixin.Batching))  # Not just Batching

        # But a single mode joint should match
        single_joint = brainstate.mixin.JointMode(training)
        self.assertTrue(single_joint.is_a(brainstate.mixin.Training))

    def test_joint_mode_single_mode(self):
        """Test JointMode with single mode."""
        batching = brainstate.mixin.Batching()
        joint = brainstate.mixin.JointMode(batching)

        self.assertTrue(joint.has(brainstate.mixin.Batching))
        self.assertTrue(joint.is_a(brainstate.mixin.Batching))

    def test_joint_mode_attribute_access(self):
        """Test JointMode attribute delegation."""
        batching = brainstate.mixin.Batching(batch_size=32, batch_axis=1)
        training = brainstate.mixin.Training()
        joint = brainstate.mixin.JointMode(batching, training)

        # Should access batching attributes
        self.assertEqual(joint.batch_size, 32)
        self.assertEqual(joint.batch_axis, 1)

    def test_joint_mode_invalid_type(self):
        """Test JointMode with non-Mode raises error."""
        with self.assertRaises(TypeError):
            brainstate.mixin.JointMode("not a mode")

    def test_joint_mode_modes_attribute(self):
        """Test accessing modes attribute."""
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching()
        joint = brainstate.mixin.JointMode(training, batching)

        self.assertEqual(len(joint.modes), 2)
        self.assertIn(training, joint.modes)
        self.assertIn(batching, joint.modes)

    def test_joint_mode_types_attribute(self):
        """Test accessing types attribute."""
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching()
        joint = brainstate.mixin.JointMode(training, batching)

        self.assertEqual(len(joint.types), 2)
        self.assertIn(brainstate.mixin.Training, joint.types)
        self.assertIn(brainstate.mixin.Batching, joint.types)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features."""

    def test_param_desc_with_modes(self):
        """Test ParamDesc with Mode system."""

        class Model(brainstate.mixin.ParamDesc):
            def __init__(self, size, mode=None):
                self.size = size
                self.mode = mode if mode is not None else brainstate.mixin.Mode()

        # Create descriptor with training mode
        train_model_desc = Model.desc(size=100, mode=brainstate.mixin.Training())
        model = train_model_desc()

        self.assertEqual(model.size, 100)
        self.assertTrue(model.mode.has(brainstate.mixin.Training))

    def test_joint_types_with_multiple_mixins(self):
        """Test JointTypes with multiple mixin classes."""

        class Serializable(brainstate.mixin.Mixin):
            def save(self):
                return "saved"

        class Trainable(brainstate.mixin.Mixin):
            def train(self):
                return "trained"

        class Evaluable(brainstate.mixin.Mixin):
            def evaluate(self):
                return "evaluated"

        FullModel = brainstate.mixin.JointTypes(Serializable, Trainable, Evaluable)

        class MyModel(Serializable, Trainable, Evaluable):
            pass

        model = MyModel()
        self.assertTrue(isinstance(model, FullModel))
        self.assertEqual(model.save(), "saved")
        self.assertEqual(model.train(), "trained")
        self.assertEqual(model.evaluate(), "evaluated")

    def test_complex_mode_scenario(self):
        """Test complex scenario with multiple modes."""

        class NeuralNetwork:
            def __init__(self):
                self.mode = None

            def set_mode(self, mode):
                self.mode = mode

            def forward(self, x):
                if self.mode is None:
                    return x

                if self.mode.has(brainstate.mixin.Training):
                    # Add noise during training
                    x = x + 0.1

                if self.mode.has(brainstate.mixin.Batching):
                    # Process in batches
                    batch_size = self.mode.batch_size
                    # Just return with batch info for testing
                    return x, batch_size

                return x

        net = NeuralNetwork()

        # Test evaluation mode
        result = net.forward(1.0)
        self.assertEqual(result, 1.0)

        # Test training mode
        net.set_mode(brainstate.mixin.Training())
        result = net.forward(1.0)
        self.assertAlmostEqual(result, 1.1)

        # Test joint mode
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching(batch_size=32)
        net.set_mode(brainstate.mixin.JointMode(training, batching))

        result, batch_size = net.forward(1.0)
        self.assertAlmostEqual(result, 1.1)
        self.assertEqual(batch_size, 32)


class TestJointTypesComprehensive(unittest.TestCase):
    """Comprehensive tests for JointTypes special methods and functionality."""

    def setUp(self):
        """Set up test classes."""
        class A:
            pass

        class B:
            pass

        class C:
            pass

        self.A = A
        self.B = B
        self.C = C

    def test_repr(self):
        """Test __repr__ method."""
        JT = brainstate.mixin.JointTypes[self.A, self.B]
        repr_str = repr(JT)
        self.assertIn('JointTypes', repr_str)
        self.assertIn('A', repr_str)
        self.assertIn('B', repr_str)

    def test_eq_same_order(self):
        """Test equality with same type order."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.A, self.B]
        self.assertEqual(JT1, JT2)

    def test_eq_different_order(self):
        """Test equality with different type order."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.B, self.A]
        self.assertEqual(JT1, JT2)

    def test_eq_different_types(self):
        """Test inequality with different types."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.A, self.C]
        self.assertNotEqual(JT1, JT2)

    def test_eq_with_non_jointtypes(self):
        """Test equality with non-JointTypes object."""
        JT = brainstate.mixin.JointTypes[self.A, self.B]
        self.assertNotEqual(JT, "not a type")
        self.assertNotEqual(JT, 42)
        self.assertNotEqual(JT, self.A)

    def test_hash_consistency(self):
        """Test hash consistency."""
        JT = brainstate.mixin.JointTypes[self.A, self.B]
        hash1 = hash(JT)
        hash2 = hash(JT)
        self.assertEqual(hash1, hash2)

    def test_hash_order_independent(self):
        """Test hash is order-independent."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.B, self.A]
        self.assertEqual(hash(JT1), hash(JT2))

    def test_hash_different_for_different_types(self):
        """Test different types have different hashes."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.A, self.C]
        # Note: hash collision is possible but unlikely for different types
        self.assertNotEqual(hash(JT1), hash(JT2))

    def test_hashable_in_set(self):
        """Test JointTypes can be used in sets."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.B, self.A]
        JT3 = brainstate.mixin.JointTypes[self.A, self.C]

        type_set = {JT1, JT2, JT3}
        # JT1 and JT2 are equal, so set should have 2 elements
        self.assertEqual(len(type_set), 2)
        self.assertIn(JT1, type_set)
        self.assertIn(JT2, type_set)
        self.assertIn(JT3, type_set)

    def test_as_dict_key(self):
        """Test JointTypes can be used as dict keys."""
        JT1 = brainstate.mixin.JointTypes[self.A, self.B]
        JT2 = brainstate.mixin.JointTypes[self.B, self.A]

        type_dict = {JT1: "AB type"}
        self.assertIn(JT1, type_dict)
        # JT2 should work as key since it's equal to JT1
        self.assertIn(JT2, type_dict)
        self.assertEqual(type_dict[JT2], "AB type")

    def test_pickle_roundtrip(self):
        """Test pickling and unpickling with built-in types."""
        import pickle
        # Use built-in types since local classes can't be pickled
        JT = brainstate.mixin.JointTypes[int, str]
        pickled = pickle.dumps(JT)
        unpickled = pickle.loads(pickled)
        self.assertEqual(JT, unpickled)
        self.assertEqual(hash(JT), hash(unpickled))

    def test_pickle_preserves_isinstance(self):
        """Test isinstance works after pickle with built-in types."""
        import pickle

        class IntStr(int):
            """A class that inherits from int."""
            pass

        # Use built-in types for pickling
        JT = brainstate.mixin.JointTypes[int, object]
        pickled = pickle.dumps(JT)
        unpickled = pickle.loads(pickled)

        obj = 42
        self.assertTrue(isinstance(obj, JT))
        self.assertTrue(isinstance(obj, unpickled))

    def test_multiple_types(self):
        """Test JointTypes with more than 2 types."""
        JT = brainstate.mixin.JointTypes[self.A, self.B, self.C]

        class ABC(self.A, self.B, self.C):
            pass

        self.assertTrue(issubclass(ABC, JT))

        class AB(self.A, self.B):
            pass

        self.assertFalse(issubclass(AB, JT))

    def test_subscript_vs_call_syntax(self):
        """Test subscript and call syntax produce equal results."""
        JT_subscript = brainstate.mixin.JointTypes[self.A, self.B]
        JT_call = brainstate.mixin.JointTypes(self.A, self.B)
        self.assertEqual(JT_subscript, JT_call)
        self.assertEqual(hash(JT_subscript), hash(JT_call))

    def test_args_attribute(self):
        """Test __args__ attribute contains correct types."""
        JT = brainstate.mixin.JointTypes[self.A, self.B]
        self.assertIn(self.A, JT.__args__)
        self.assertIn(self.B, JT.__args__)
        self.assertEqual(len(JT.__args__), 2)


class TestOneOfTypesComprehensive(unittest.TestCase):
    """Comprehensive tests for OneOfTypes special methods and functionality."""

    def setUp(self):
        """Set up test classes."""
        class A:
            pass

        class B:
            pass

        class C:
            pass

        self.A = A
        self.B = B
        self.C = C

    def test_repr(self):
        """Test __repr__ method."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B]
        repr_str = repr(OT)
        self.assertIn('OneOfTypes', repr_str)
        self.assertIn('A', repr_str)
        self.assertIn('B', repr_str)

    def test_eq_same_order(self):
        """Test equality with same type order."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.A, self.B]
        self.assertEqual(OT1, OT2)

    def test_eq_different_order(self):
        """Test equality with different type order."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.B, self.A]
        self.assertEqual(OT1, OT2)

    def test_eq_different_types(self):
        """Test inequality with different types."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.A, self.C]
        self.assertNotEqual(OT1, OT2)

    def test_eq_with_non_oneoftypes(self):
        """Test equality with non-OneOfTypes object."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B]
        self.assertNotEqual(OT, "not a type")
        self.assertNotEqual(OT, 42)
        self.assertNotEqual(OT, self.A)

    def test_hash_consistency(self):
        """Test hash consistency."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B]
        hash1 = hash(OT)
        hash2 = hash(OT)
        self.assertEqual(hash1, hash2)

    def test_hash_order_independent(self):
        """Test hash is order-independent."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.B, self.A]
        self.assertEqual(hash(OT1), hash(OT2))

    def test_hash_different_for_different_types(self):
        """Test different types have different hashes."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.A, self.C]
        self.assertNotEqual(hash(OT1), hash(OT2))

    def test_hashable_in_set(self):
        """Test OneOfTypes can be used in sets."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.B, self.A]
        OT3 = brainstate.mixin.OneOfTypes[self.A, self.C]

        type_set = {OT1, OT2, OT3}
        # OT1 and OT2 are equal, so set should have 2 elements
        self.assertEqual(len(type_set), 2)
        self.assertIn(OT1, type_set)
        self.assertIn(OT2, type_set)
        self.assertIn(OT3, type_set)

    def test_as_dict_key(self):
        """Test OneOfTypes can be used as dict keys."""
        OT1 = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT2 = brainstate.mixin.OneOfTypes[self.B, self.A]

        type_dict = {OT1: "A or B type"}
        self.assertIn(OT1, type_dict)
        self.assertIn(OT2, type_dict)
        self.assertEqual(type_dict[OT2], "A or B type")

    def test_pickle_roundtrip(self):
        """Test pickling and unpickling with built-in types."""
        import pickle
        # Use built-in types since local classes can't be pickled
        OT = brainstate.mixin.OneOfTypes[int, str]
        pickled = pickle.dumps(OT)
        unpickled = pickle.loads(pickled)
        self.assertEqual(OT, unpickled)
        self.assertEqual(hash(OT), hash(unpickled))

    def test_pickle_preserves_isinstance(self):
        """Test isinstance works after pickle with built-in types."""
        import pickle
        # Use built-in types for pickling
        OT = brainstate.mixin.OneOfTypes[int, str]
        pickled = pickle.dumps(OT)
        unpickled = pickle.loads(pickled)

        obj_a = 42
        obj_b = "hello"

        self.assertTrue(isinstance(obj_a, OT))
        self.assertTrue(isinstance(obj_a, unpickled))
        self.assertTrue(isinstance(obj_b, OT))
        self.assertTrue(isinstance(obj_b, unpickled))

    def test_isinstance_with_any_type(self):
        """Test isinstance returns True if object is instance of any type."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B]

        obj_a = self.A()
        obj_b = self.B()
        obj_c = self.C()

        self.assertTrue(isinstance(obj_a, OT))
        self.assertTrue(isinstance(obj_b, OT))
        self.assertFalse(isinstance(obj_c, OT))

    def test_issubclass_with_any_type(self):
        """Test issubclass returns True if class is subclass of any type."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B]

        class SubA(self.A):
            pass

        class SubB(self.B):
            pass

        self.assertTrue(issubclass(SubA, OT))
        self.assertTrue(issubclass(SubB, OT))
        self.assertTrue(issubclass(self.A, OT))
        self.assertTrue(issubclass(self.B, OT))
        self.assertFalse(issubclass(self.C, OT))

    def test_multiple_types(self):
        """Test OneOfTypes with more than 2 types."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B, self.C]

        obj_a = self.A()
        obj_b = self.B()
        obj_c = self.C()

        self.assertTrue(isinstance(obj_a, OT))
        self.assertTrue(isinstance(obj_b, OT))
        self.assertTrue(isinstance(obj_c, OT))

    def test_subscript_vs_call_syntax(self):
        """Test subscript and call syntax produce equal results."""
        OT_subscript = brainstate.mixin.OneOfTypes[self.A, self.B]
        OT_call = brainstate.mixin.OneOfTypes(self.A, self.B)
        self.assertEqual(OT_subscript, OT_call)
        self.assertEqual(hash(OT_subscript), hash(OT_call))

    def test_args_attribute(self):
        """Test __args__ attribute contains correct types."""
        OT = brainstate.mixin.OneOfTypes[self.A, self.B]
        self.assertIn(self.A, OT.__args__)
        self.assertIn(self.B, OT.__args__)
        self.assertEqual(len(OT.__args__), 2)

    def test_with_builtin_types(self):
        """Test OneOfTypes with built-in types."""
        OT = brainstate.mixin.OneOfTypes[int, float, str]

        self.assertTrue(isinstance(42, OT))
        self.assertTrue(isinstance(3.14, OT))
        self.assertTrue(isinstance("hello", OT))
        self.assertFalse(isinstance([], OT))


class TestJointTy:
    def test1(self):
        class Potassium:
            pass

        class Calcium:
            pass

        # Test JointTypes
        result1 = brainstate.mixin.JointTypes(Potassium, Calcium)
        result2 = brainstate.mixin.JointTypes[Potassium, Calcium]
        print(f'Function call: {result1}')
        print(f'Subscript: {result2}')
        print(f'Same? {result1 == result2}')

        # Test OneOfTypes
        result3 = brainstate.mixin.OneOfTypes(Potassium, Calcium)
        result4 = brainstate.mixin.OneOfTypes[Potassium, Calcium]
        print(f'\nOneOfTypes Function call: {result3}')
        print(f'OneOfTypes Subscript: {result4}')
        print(f'Same? {result3 == result4}')


if __name__ == '__main__':
    unittest.main()
