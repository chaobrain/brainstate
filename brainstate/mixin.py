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

# -*- coding: utf-8 -*-

"""
Mixin classes and utility types for brainstate.

This module provides various mixin classes and custom type definitions that
enhance the functionality of brainstate components. It includes parameter
description mixins, alignment interfaces, and custom type definitions for
expressing complex type requirements.
"""

from typing import Sequence, Optional, TypeVar, Union
import jax

# Type variable for generic typing
T = TypeVar('T')
ArrayLike = jax.typing.ArrayLike

__all__ = [
    'Mixin',
    'ParamDesc',
    'ParamDescriber',
    'AlignPost',
    'BindCondData',

    # types
    'JointTypes',
    'OneOfTypes',

    # behavior modes
    'Mode',
    'JointMode',
    'Batching',
    'Training',
]


def hashable(x):
    """
    Check if an object is hashable.

    Parameters
    ----------
    x : Any
        The object to check for hashability.

    Returns
    -------
    bool
        True if the object is hashable, False otherwise.

    Examples
    --------
    .. code-block:: python

        import brainstate

        # Hashable objects
        assert brainstate.mixin.hashable(42) == True
        assert brainstate.mixin.hashable("string") == True
        assert brainstate.mixin.hashable((1, 2, 3)) == True

        # Non-hashable objects
        assert brainstate.mixin.hashable([1, 2, 3]) == False
        assert brainstate.mixin.hashable({"key": "value"}) == False
    """
    try:
        hash(x)
        return True
    except TypeError:
        return False


class Mixin(object):
    """
    Base Mixin object for behavioral extensions.

    The key characteristic of a :py:class:`~.Mixin` is that it provides only
    behavioral functions without requiring initialization. Mixins are used to
    add specific functionality to classes through multiple inheritance without
    the complexity of a full base class.

    Notes
    -----
    Mixins should not define ``__init__`` methods. They should only provide
    methods that add specific behaviors to the classes that inherit from them.

    Examples
    --------
    Creating a custom mixin:

    .. code-block:: python

        import brainstate

        class LoggingMixin(brainstate.mixin.Mixin):
            def log(self, message):
                print(f"[{self.__class__.__name__}] {message}")

        class MyComponent(brainstate.nn.Module, LoggingMixin):
            def __init__(self):
                super().__init__()

            def process(self):
                self.log("Processing data...")
                return "Done"

        component = MyComponent()
        component.process()  # Prints: [MyComponent] Processing data...
    """
    pass


class ParamDesc(Mixin):
    """
    Mixin for describing initialization parameters.

    This mixin enables a class to have a ``desc`` classmethod, which produces
    an instance of :py:class:`~.ParamDescriber`. This is useful for creating
    parameter templates that can be reused to instantiate multiple objects
    with the same configuration.

    Attributes
    ----------
    non_hashable_params : sequence of str, optional
        Names of parameters that are not hashable and should be handled specially.

    Notes
    -----
    This mixin can be applied to any Python class, not just brainstate-specific classes.

    Examples
    --------
    Basic usage of ParamDesc:

    .. code-block:: python

        import brainstate

        class NeuronModel(brainstate.mixin.ParamDesc):
            def __init__(self, size, tau=10.0, threshold=1.0):
                self.size = size
                self.tau = tau
                self.threshold = threshold

        # Create a parameter descriptor
        neuron_desc = NeuronModel.desc(size=100, tau=20.0)

        # Use the descriptor to create instances
        neuron1 = neuron_desc(threshold=0.8)  # Creates with threshold=0.8
        neuron2 = neuron_desc(threshold=1.2)  # Creates with threshold=1.2

        # Both neurons share size=100, tau=20.0 but have different thresholds

    Creating reusable templates:

    .. code-block:: python

        # Define a template for excitatory neurons
        exc_neuron_template = NeuronModel.desc(size=1000, tau=10.0, threshold=1.0)

        # Define a template for inhibitory neurons
        inh_neuron_template = NeuronModel.desc(size=250, tau=5.0, threshold=0.5)

        # Create multiple instances from templates
        exc_population = [exc_neuron_template() for _ in range(5)]
        inh_population = [inh_neuron_template() for _ in range(2)]
    """

    # Optional list of parameter names that are not hashable
    # These will be converted to strings for hashing purposes
    non_hashable_params: Optional[Sequence[str]] = None

    @classmethod
    def desc(cls, *args, **kwargs) -> 'ParamDescriber':
        """
        Create a parameter describer for this class.

        Parameters
        ----------
        *args
            Positional arguments to be used in future instantiations.
        **kwargs
            Keyword arguments to be used in future instantiations.

        Returns
        -------
        ParamDescriber
            A descriptor that can be used to create instances with these parameters.
        """
        return ParamDescriber(cls, *args, **kwargs)


class HashableDict(dict):
    """
    A dictionary that can be hashed by converting non-hashable values to strings.

    This is used internally to make parameter dictionaries hashable so they can
    be used as part of cache keys or other contexts requiring hashability.

    Parameters
    ----------
    the_dict : dict
        The dictionary to make hashable.

    Notes
    -----
    Non-hashable values in the dictionary are automatically converted to their
    string representation.

    Examples
    --------
    .. code-block:: python

        import brainstate
        import jax.numpy as jnp

        # Regular dict with non-hashable values cannot be hashed
        regular_dict = {"array": jnp.array([1, 2, 3]), "value": 42}
        # hash(regular_dict)  # This would raise TypeError

        # HashableDict can be hashed
        hashable = brainstate.mixin.HashableDict(regular_dict)
        key = hash(hashable)  # This works!

        # Can be used in sets or as dict keys
        cache = {hashable: "result"}
    """

    def __init__(self, the_dict: dict):
        # Process the dictionary to ensure all values are hashable
        out = dict()
        for k, v in the_dict.items():
            if not hashable(v):
                # Convert non-hashable values to their string representation
                v = str(v)
            out[k] = v
        super().__init__(out)

    def __hash__(self):
        """
        Compute hash from sorted items for consistent hashing regardless of insertion order.
        """
        return hash(tuple(sorted(self.items())))


class NoSubclassMeta(type):
    """
    Metaclass that prevents a class from being subclassed.

    This is used to ensure that certain classes (like ParamDescriber) are used
    as-is and not extended through inheritance, which could lead to unexpected
    behavior.

    Raises
    ------
    TypeError
        If an attempt is made to subclass a class using this metaclass.
    """

    def __new__(cls, name, bases, classdict):
        # Check if any base class uses NoSubclassMeta
        for b in bases:
            if isinstance(b, NoSubclassMeta):
                raise TypeError("type '{0}' is not an acceptable base type".format(b.__name__))
        return type.__new__(cls, name, bases, dict(classdict))


class ParamDescriber(metaclass=NoSubclassMeta):
    """
    Parameter descriptor for deferred object instantiation.

    This class stores a class reference along with arguments and keyword arguments,
    allowing for deferred instantiation. It's useful for creating templates that
    can be reused to create multiple instances with similar configurations.

    Parameters
    ----------
    cls : type
        The class to be instantiated.
    *desc_tuple
        Positional arguments to be stored and used during instantiation.
    **desc_dict
        Keyword arguments to be stored and used during instantiation.

    Attributes
    ----------
    cls : type
        The class that will be instantiated.
    args : tuple
        Stored positional arguments.
    kwargs : dict
        Stored keyword arguments.
    identifier : tuple
        A hashable identifier for this descriptor.

    Notes
    -----
    ParamDescriber cannot be subclassed due to the NoSubclassMeta metaclass.
    This ensures consistent behavior across the codebase.

    Examples
    --------
    Manual creation of a descriptor:

    .. code-block:: python

        import brainstate

        class Network:
            def __init__(self, n_neurons, learning_rate=0.01):
                self.n_neurons = n_neurons
                self.learning_rate = learning_rate

        # Create a descriptor
        network_desc = brainstate.mixin.ParamDescriber(
            Network, n_neurons=1000, learning_rate=0.001
        )

        # Use the descriptor to create instances with additional args
        net1 = network_desc()
        net2 = network_desc()  # Same configuration

    Using with ParamDesc mixin:

    .. code-block:: python

        class Network(brainstate.mixin.ParamDesc):
            def __init__(self, n_neurons, learning_rate=0.01):
                self.n_neurons = n_neurons
                self.learning_rate = learning_rate

        # More concise syntax using the desc() classmethod
        network_desc = Network.desc(n_neurons=1000)
        net = network_desc(learning_rate=0.005)  # Override learning_rate
    """

    def __init__(self, cls: T, *desc_tuple, **desc_dict):
        # Store the class to be instantiated
        self.cls: type = cls

        # Store the arguments for later instantiation
        self.args = desc_tuple
        self.kwargs = desc_dict

        # Create a hashable identifier for caching/comparison purposes
        # This combines the class, args tuple, and hashable kwargs dict
        self._identifier = (cls, tuple(desc_tuple), HashableDict(desc_dict))

    def __call__(self, *args, **kwargs) -> T:
        """
        Instantiate the class with stored and additional arguments.

        Parameters
        ----------
        *args
            Additional positional arguments to append.
        **kwargs
            Additional keyword arguments to merge (will override stored kwargs).

        Returns
        -------
        T
            An instance of the described class.
        """
        # Merge stored arguments with new arguments
        # Stored args come first, then new args
        # Merge kwargs with new kwargs overriding stored ones
        merged_kwargs = {**self.kwargs, **kwargs}
        return self.cls(*self.args, *args, **merged_kwargs)

    def init(self, *args, **kwargs):
        """
        Alias for __call__, explicitly named for clarity.

        Parameters
        ----------
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        T
            An instance of the described class.
        """
        return self.__call__(*args, **kwargs)

    def __instancecheck__(self, instance):
        """
        Check if an instance is compatible with this descriptor.

        Parameters
        ----------
        instance : Any
            The instance to check.

        Returns
        -------
        bool
            True if the instance is a ParamDescriber for a compatible class.
        """
        # Must be a ParamDescriber
        if not isinstance(instance, ParamDescriber):
            return False
        # The described class must be a subclass of our class
        if not issubclass(instance.cls, self.cls):
            return False
        return True

    @classmethod
    def __class_getitem__(cls, item: type):
        """
        Support for subscript notation: ParamDescriber[MyClass].

        Parameters
        ----------
        item : type
            The class to create a descriptor for.

        Returns
        -------
        ParamDescriber
            A descriptor for the given class.
        """
        return ParamDescriber(item)

    @property
    def identifier(self):
        """
        Get the unique identifier for this descriptor.

        Returns
        -------
        tuple
            A hashable identifier consisting of (class, args, kwargs).
        """
        return self._identifier

    @identifier.setter
    def identifier(self, value: ArrayLike):
        """
        Prevent modification of the identifier.

        Raises
        ------
        AttributeError
            Always, as the identifier is read-only.
        """
        raise AttributeError('Cannot set the identifier.')


class AlignPost(Mixin):
    """
    Mixin for aligning post-synaptic inputs.

    This mixin provides an interface for components that need to receive and
    process post-synaptic inputs, such as synaptic connections or neural
    populations. The ``align_post_input_add`` method should be implemented
    to handle the accumulation of external currents or inputs.

    Notes
    -----
    Classes that inherit from this mixin must implement the
    ``align_post_input_add`` method.

    Examples
    --------
    Implementing a synapse with post-synaptic alignment:

    .. code-block:: python

        import brainstate
        import jax.numpy as jnp

        class Synapse(brainstate.mixin.AlignPost):
            def __init__(self, weight):
                self.weight = weight
                self.post_current = brainstate.State(0.0)

            def align_post_input_add(self, current):
                # Accumulate the weighted current into post-synaptic target
                self.post_current.value += current * self.weight

        # Usage
        synapse = Synapse(weight=0.5)
        synapse.align_post_input_add(10.0)
        print(synapse.post_current.value)  # Output: 5.0

    Using with neural populations:

    .. code-block:: python

        class NeuronGroup(brainstate.mixin.AlignPost):
            def __init__(self, size):
                self.size = size
                self.input_current = brainstate.State(jnp.zeros(size))

            def align_post_input_add(self, current):
                # Add external current to neurons
                self.input_current.value = self.input_current.value + current

        neurons = NeuronGroup(100)
        external_input = jnp.ones(100) * 0.5
        neurons.align_post_input_add(external_input)
    """

    def align_post_input_add(self, *args, **kwargs):
        """
        Add external inputs to the post-synaptic component.

        Parameters
        ----------
        *args
            Positional arguments for the input.
        **kwargs
            Keyword arguments for the input.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class BindCondData(Mixin):
    """
    Mixin for binding temporary conductance data.

    This mixin provides an interface for temporarily storing conductance data,
    which is useful in synaptic models where conductance values need to be
    passed between computation steps without being part of the permanent state.

    Attributes
    ----------
    _conductance : Any, optional
        Temporarily bound conductance data.

    Examples
    --------
    Using conductance binding in a synapse:

    .. code-block:: python

        import brainstate
        import jax.numpy as jnp

        class ConductanceBasedSynapse(brainstate.mixin.BindCondData):
            def __init__(self):
                self._conductance = None

            def compute(self, pre_spike):
                if pre_spike:
                    # Bind conductance data temporarily
                    self.bind_cond(0.5)

                # Use conductance if available
                if self._conductance is not None:
                    current = self._conductance * (0.0 - (-70.0))
                    # Clear after use
                    self.unbind_cond()
                    return current
                return 0.0

        synapse = ConductanceBasedSynapse()
        current = synapse.compute(pre_spike=True)

    Managing conductance in a network:

    .. code-block:: python

        class SynapticConnection(brainstate.mixin.BindCondData):
            def __init__(self, g_max):
                self.g_max = g_max
                self._conductance = None

            def prepare_conductance(self, activation):
                # Bind conductance based on activation
                g = self.g_max * activation
                self.bind_cond(g)

            def apply_conductance(self, voltage):
                if self._conductance is not None:
                    current = self._conductance * voltage
                    self.unbind_cond()
                    return current
                return 0.0
    """
    # Attribute to store temporary conductance data
    _conductance: Optional

    def bind_cond(self, conductance):
        """
        Bind conductance data temporarily.

        Parameters
        ----------
        conductance : Any
            The conductance data to bind.
        """
        self._conductance = conductance

    def unbind_cond(self):
        """
        Unbind (clear) the conductance data.
        """
        self._conductance = None


def not_implemented(func):
    """
    Decorator to mark a function as not implemented.

    This decorator wraps a function to raise NotImplementedError when called,
    and adds a ``not_implemented`` attribute for checking.

    Parameters
    ----------
    func : callable
        The function to mark as not implemented.

    Returns
    -------
    callable
        A wrapper function that raises NotImplementedError.

    Examples
    --------
    .. code-block:: python

        import brainstate

        class BaseModel:
            @brainstate.mixin.not_implemented
            def process(self, x):
                pass

        model = BaseModel()
        # model.process(10)  # Raises: NotImplementedError: process is not implemented.

        # Check if a method is not implemented
        assert hasattr(BaseModel.process, 'not_implemented')
    """

    def wrapper(*args, **kwargs):
        raise NotImplementedError(f'{func.__name__} is not implemented.')

    # Mark the wrapper so we can detect not-implemented methods
    wrapper.not_implemented = True
    return wrapper


class _MetaUnionType(type):
    """
    Metaclass for creating union types that require all base types.

    This metaclass is used internally to create types that require instances
    to be instances of all specified base types (similar to intersection types).
    """

    def __new__(cls, name, bases, dct):
        # Normalize bases to a tuple
        if isinstance(bases, type):
            bases = (bases,)
        elif isinstance(bases, (list, tuple)):
            bases = tuple(bases)
            # Ensure all bases are types
            for base in bases:
                assert isinstance(base, type), f'Must be type. But got {base}'
        else:
            raise TypeError(f'Must be type. But got {bases}')
        return super().__new__(cls, name, bases, dct)

    def __instancecheck__(self, other):
        """
        Check if an instance is an instance of all base types.
        """
        cls_of_other = other.__class__
        # Must be instance of ALL base classes
        return all([issubclass(cls_of_other, cls) for cls in self.__bases__])

    def __subclasscheck__(self, subclass):
        """
        Check if a class is a subclass of all base types.
        """
        # Must be subclass of ALL base classes
        return all([issubclass(subclass, cls) for cls in self.__bases__])


def JointTypes(*types):
    """
    Create a type that requires all specified types (intersection type).

    This function creates a type hint that indicates a value must satisfy all
    of the specified types simultaneously. It's useful for expressing complex
    type requirements where a single object must implement multiple interfaces.

    Parameters
    ----------
    *types : type
        The types that must all be satisfied.

    Returns
    -------
    type
        A type that checks for all specified types.

    Notes
    -----
    - If only one type is provided, that type is returned directly.
    - Redundant types are automatically removed.
    - The order of types doesn't matter for equality checks.

    Examples
    --------
    Basic usage with interfaces:

    .. code-block:: python

        import brainstate
        from typing import Protocol

        class Trainable(Protocol):
            def train(self): ...

        class Evaluable(Protocol):
            def evaluate(self): ...

        # A model that is both trainable and evaluable
        TrainableEvaluableModel = brainstate.mixin.JointTypes(Trainable, Evaluable)

        class NeuralNetwork(Trainable, Evaluable):
            def train(self):
                return "Training..."

            def evaluate(self):
                return "Evaluating..."

        model = NeuralNetwork()
        # model satisfies JointTypes(Trainable, Evaluable)

    Using with mixin classes:

    .. code-block:: python

        class Serializable:
            def save(self): pass

        class Visualizable:
            def plot(self): pass

        # Require both serialization and visualization
        FullFeaturedModel = brainstate.mixin.JointTypes(Serializable, Visualizable)

        class MyModel(Serializable, Visualizable):
            def save(self):
                return "Saved"

            def plot(self):
                return "Plotted"
    """
    if len(types) == 0:
        raise TypeError("Cannot create a JointTypes of no types.")

    # Remove duplicates while preserving some order
    seen = set()
    unique_types = []
    for t in types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    # If only one type, return it directly
    if len(unique_types) == 1:
        return unique_types[0]

    # Create a new type dynamically using _MetaUnionType metaclass
    # The metaclass ensures isinstance/issubclass checks work correctly
    JointType = _MetaUnionType(
        'JointType',
        tuple(unique_types),
        {'__doc__': 'A type that requires all specified types.'}
    )
    return JointType


def OneOfTypes(*types):
    """
    Create a type that requires one of the specified types (union type).

    This is similar to typing.Union but provides a more intuitive name and
    consistent behavior with JointTypes. It indicates that a value must satisfy
    at least one of the specified types.

    Parameters
    ----------
    *types : type
        The types, one of which must be satisfied.

    Returns
    -------
    Union type
        A union type of the specified types.

    Notes
    -----
    - If only one type is provided, that type is returned directly.
    - Redundant types are automatically removed.
    - The order of types doesn't matter for equality checks.
    - This is equivalent to typing.Union[...].

    Examples
    --------
    Basic usage with different types:

    .. code-block:: python

        import brainstate

        # A parameter that can be int or float
        NumericType = brainstate.mixin.OneOfTypes(int, float)

        def process_value(x: NumericType):
            return x * 2

        # Both work
        result1 = process_value(5)      # int
        result2 = process_value(3.14)   # float

    Using with class types:

    .. code-block:: python

        class NumpyArray:
            pass

        class JAXArray:
            pass

        # Accept either numpy or JAX arrays
        ArrayType = brainstate.mixin.OneOfTypes(NumpyArray, JAXArray)

        def compute(arr: ArrayType):
            if isinstance(arr, NumpyArray):
                return "Processing numpy array"
            elif isinstance(arr, JAXArray):
                return "Processing JAX array"

    Combining with None for optional types:

    .. code-block:: python

        # Optional string (equivalent to Optional[str])
        MaybeString = brainstate.mixin.OneOfTypes(str, type(None))

        def format_name(name: MaybeString) -> str:
            if name is None:
                return "Anonymous"
            return name.title()
    """
    if len(types) == 0:
        raise TypeError("Cannot create a OneOfTypes of no types.")

    # Remove duplicates
    seen = set()
    unique_types = []
    for t in types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)

    # If only one type, return it directly
    if len(unique_types) == 1:
        return unique_types[0]

    # Use standard Union from typing
    # We need to use eval to dynamically create Union with variable number of types
    # This avoids using internal APIs while maintaining functionality
    import typing
    # Build the Union dynamically
    result = unique_types[0]
    for t in unique_types[1:]:
        result = Union[result, t]
    return result


class Mode(Mixin):
    """
    Base class for computation behavior modes.

    Modes are used to represent different computational contexts or behaviors,
    such as training vs evaluation, batched vs single-sample processing, etc.
    They provide a flexible way to configure how models and components behave
    in different scenarios.

    Examples
    --------
    Creating a custom mode:

    .. code-block:: python

        import brainstate

        class InferenceMode(brainstate.mixin.Mode):
            def __init__(self, use_cache=True):
                self.use_cache = use_cache

        # Create mode instances
        inference = InferenceMode(use_cache=True)
        print(inference)  # Output: InferenceMode

    Checking mode types:

    .. code-block:: python

        class FastMode(brainstate.mixin.Mode):
            pass

        class SlowMode(brainstate.mixin.Mode):
            pass

        fast = FastMode()
        slow = SlowMode()

        # Check exact mode type
        assert fast.is_a(FastMode)
        assert not fast.is_a(SlowMode)

        # Check if mode is an instance of a type
        assert fast.has(brainstate.mixin.Mode)

    Using modes in a model:

    .. code-block:: python

        class Model:
            def __init__(self):
                self.mode = brainstate.mixin.Training()

            def forward(self, x):
                if self.mode.has(brainstate.mixin.Training):
                    # Training-specific logic
                    return self.train_forward(x)
                else:
                    # Inference logic
                    return self.eval_forward(x)

            def train_forward(self, x):
                return x + 0.1  # Add noise during training

            def eval_forward(self, x):
                return x  # No noise during evaluation
    """

    def __repr__(self):
        """
        String representation of the mode.

        Returns
        -------
        str
            The class name of the mode.
        """
        return self.__class__.__name__

    def __eq__(self, other: 'Mode'):
        """
        Check equality of modes based on their type.

        Parameters
        ----------
        other : Mode
            Another mode to compare with.

        Returns
        -------
        bool
            True if both modes are of the same class.
        """
        assert isinstance(other, Mode)
        return other.__class__ == self.__class__

    def is_a(self, mode: type):
        """
        Check whether the mode is exactly the desired mode type.

        This performs an exact type match, not checking for subclasses.

        Parameters
        ----------
        mode : type
            The mode type to check against.

        Returns
        -------
        bool
            True if this mode is exactly of the specified type.

        Examples
        --------
        .. code-block:: python

            import brainstate

            training_mode = brainstate.mixin.Training()
            assert training_mode.is_a(brainstate.mixin.Training)
            assert not training_mode.is_a(brainstate.mixin.Batching)
        """
        assert isinstance(mode, type), 'Must be a type.'
        return self.__class__ == mode

    def has(self, mode: type):
        """
        Check whether the mode includes the desired mode type.

        This checks if the current mode is an instance of the specified type,
        including checking for subclasses.

        Parameters
        ----------
        mode : type
            The mode type to check for.

        Returns
        -------
        bool
            True if this mode is an instance of the specified type.

        Examples
        --------
        .. code-block:: python

            import brainstate

            # Create a custom mode that extends Training
            class AdvancedTraining(brainstate.mixin.Training):
                pass

            advanced = AdvancedTraining()
            assert advanced.has(brainstate.mixin.Training)  # True (subclass)
            assert advanced.has(brainstate.mixin.Mode)      # True (base class)
        """
        assert isinstance(mode, type), 'Must be a type.'
        return isinstance(self, mode)


class JointMode(Mode):
    """
    A mode that combines multiple modes simultaneously.

    JointMode allows expressing that a computation is in multiple modes at once,
    such as being both in training mode and batching mode. This is useful for
    complex scenarios where multiple behavioral aspects need to be active.

    Parameters
    ----------
    *modes : Mode
        The modes to combine.

    Attributes
    ----------
    modes : tuple of Mode
        The individual modes that are combined.
    types : set of type
        The types of the combined modes.

    Raises
    ------
    TypeError
        If any of the provided arguments is not a Mode instance.

    Examples
    --------
    Combining training and batching modes:

    .. code-block:: python

        import brainstate

        # Create individual modes
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching(batch_size=32)

        # Combine them
        joint = brainstate.mixin.JointMode(training, batching)
        print(joint)  # JointMode(Training, Batching(in_size=32, axis=0))

        # Check if specific modes are present
        assert joint.has(brainstate.mixin.Training)
        assert joint.has(brainstate.mixin.Batching)

        # Access attributes from combined modes
        print(joint.batch_size)  # 32 (from Batching mode)

    Using in model configuration:

    .. code-block:: python

        class NeuralNetwork:
            def __init__(self):
                self.mode = None

            def set_train_mode(self, batch_size=1):
                # Set both training and batching modes
                training = brainstate.mixin.Training()
                batching = brainstate.mixin.Batching(batch_size=batch_size)
                self.mode = brainstate.mixin.JointMode(training, batching)

            def forward(self, x):
                if self.mode.has(brainstate.mixin.Training):
                    x = self.apply_dropout(x)

                if self.mode.has(brainstate.mixin.Batching):
                    # Process in batches
                    batch_size = self.mode.batch_size
                    return self.batch_process(x, batch_size)

                return self.process(x)

        model = NeuralNetwork()
        model.set_train_mode(batch_size=64)
    """

    def __init__(self, *modes: Mode):
        # Validate that all arguments are Mode instances
        for m_ in modes:
            if not isinstance(m_, Mode):
                raise TypeError(f'The supported type must be a tuple/list of Mode. But we got {m_}')

        # Store the modes as a tuple
        self.modes = tuple(modes)

        # Store the types of the modes for quick lookup
        self.types = set([m.__class__ for m in modes])

    def __repr__(self):
        """
        String representation showing all combined modes.

        Returns
        -------
        str
            A string showing the joint mode and its components.
        """
        return f'{self.__class__.__name__}({", ".join([repr(m) for m in self.modes])})'

    def has(self, mode: type):
        """
        Check whether any of the combined modes includes the desired type.

        Parameters
        ----------
        mode : type
            The mode type to check for.

        Returns
        -------
        bool
            True if any of the combined modes is or inherits from the specified type.

        Examples
        --------
        .. code-block:: python

            import brainstate

            training = brainstate.mixin.Training()
            batching = brainstate.mixin.Batching(batch_size=16)
            joint = brainstate.mixin.JointMode(training, batching)

            assert joint.has(brainstate.mixin.Training)
            assert joint.has(brainstate.mixin.Batching)
            assert joint.has(brainstate.mixin.Mode)  # Base class
        """
        assert isinstance(mode, type), 'Must be a type.'
        # Check if any of the combined mode types is a subclass of the target mode
        return any([issubclass(cls, mode) for cls in self.types])

    def is_a(self, cls: type):
        """
        Check whether the joint mode is exactly the desired combined type.

        This is a complex check that verifies the joint mode matches a specific
        combination of types.

        Parameters
        ----------
        cls : type
            The combined type to check against.

        Returns
        -------
        bool
            True if the joint mode exactly matches the specified type combination.
        """
        # Use JointTypes to create the expected type from our mode types
        return JointTypes(*tuple(self.types)) == cls

    def __getattr__(self, item):
        """
        Get attributes from the combined modes.

        This method searches through all combined modes to find the requested
        attribute, allowing transparent access to properties of any of the
        combined modes.

        Parameters
        ----------
        item : str
            The attribute name to search for.

        Returns
        -------
        Any
            The attribute value from the first mode that has it.

        Raises
        ------
        AttributeError
            If the attribute is not found in any of the combined modes.

        Examples
        --------
        .. code-block:: python

            import brainstate

            batching = brainstate.mixin.Batching(batch_size=32, batch_axis=1)
            training = brainstate.mixin.Training()
            joint = brainstate.mixin.JointMode(batching, training)

            # Access batching attributes directly
            print(joint.batch_size)  # 32
            print(joint.batch_axis)  # 1
        """
        # Don't interfere with accessing modes and types attributes
        if item in ['modes', 'types']:
            return super().__getattribute__(item)

        # Search for the attribute in each combined mode
        for m in self.modes:
            if hasattr(m, item):
                return getattr(m, item)

        # If not found, fall back to default behavior (will raise AttributeError)
        return super().__getattribute__(item)


class Batching(Mode):
    """
    Mode indicating batched computation.

    This mode specifies that computations should be performed on batches of data,
    including information about the batch size and which axis represents the batch
    dimension.

    Parameters
    ----------
    batch_size : int, default 1
        The size of each batch.
    batch_axis : int, default 0
        The axis along which batching occurs.

    Attributes
    ----------
    batch_size : int
        The number of samples in each batch.
    batch_axis : int
        The axis index representing the batch dimension.

    Examples
    --------
    Basic batching configuration:

    .. code-block:: python

        import brainstate

        # Create a batching mode
        batching = brainstate.mixin.Batching(batch_size=32, batch_axis=0)
        print(batching)  # Batching(in_size=32, axis=0)

        # Access batch parameters
        print(f"Processing {batching.batch_size} samples at once")
        print(f"Batch dimension is axis {batching.batch_axis}")

    Using in a model:

    .. code-block:: python

        import jax.numpy as jnp

        class BatchedModel:
            def __init__(self):
                self.mode = None

            def set_batch_mode(self, batch_size, batch_axis=0):
                self.mode = brainstate.mixin.Batching(batch_size, batch_axis)

            def process(self, x):
                if self.mode is not None and self.mode.has(brainstate.mixin.Batching):
                    # Process in batches
                    batch_size = self.mode.batch_size
                    axis = self.mode.batch_axis
                    return jnp.mean(x, axis=axis, keepdims=True)
                return x

        model = BatchedModel()
        model.set_batch_mode(batch_size=64)

        # Process batched data
        data = jnp.random.randn(64, 100)  # 64 samples, 100 features
        result = model.process(data)

    Combining with other modes:

    .. code-block:: python

        # Combine batching with training mode
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching(batch_size=128)
        combined = brainstate.mixin.JointMode(training, batching)

        # Use in a training loop
        def train_step(model, data, mode):
            if mode.has(brainstate.mixin.Batching):
                # Split data into batches
                batch_size = mode.batch_size
                # ... batched processing ...
            if mode.has(brainstate.mixin.Training):
                # Apply training-specific operations
                # ... training logic ...
            pass
    """

    def __init__(self, batch_size: int = 1, batch_axis: int = 0):
        self.batch_size = batch_size
        self.batch_axis = batch_axis

    def __repr__(self):
        """
        String representation showing batch configuration.

        Returns
        -------
        str
            A string showing the batch size and axis.
        """
        return f'{self.__class__.__name__}(in_size={self.batch_size}, axis={self.batch_axis})'


class Training(Mode):
    """
    Mode indicating training computation.

    This mode specifies that the model is in training mode, which typically
    enables behaviors like dropout, batch normalization in training mode,
    gradient computation, etc.

    Examples
    --------
    Basic training mode:

    .. code-block:: python

        import brainstate

        # Create training mode
        training = brainstate.mixin.Training()
        print(training)  # Training

        # Check mode
        assert training.is_a(brainstate.mixin.Training)
        assert training.has(brainstate.mixin.Mode)

    Using in a model with dropout:

    .. code-block:: python

        import brainstate
        import jax
        import jax.numpy as jnp

        class ModelWithDropout:
            def __init__(self, dropout_rate=0.5):
                self.dropout_rate = dropout_rate
                self.mode = None

            def set_training(self, is_training=True):
                if is_training:
                    self.mode = brainstate.mixin.Training()
                else:
                    self.mode = brainstate.mixin.Mode()  # Evaluation mode

            def forward(self, x, rng_key):
                # Apply dropout only during training
                if self.mode is not None and self.mode.has(brainstate.mixin.Training):
                    keep_prob = 1.0 - self.dropout_rate
                    mask = jax.random.bernoulli(rng_key, keep_prob, x.shape)
                    x = jnp.where(mask, x / keep_prob, 0)
                return x

        model = ModelWithDropout()

        # Training mode
        model.set_training(True)
        key = jax.random.PRNGKey(0)
        x_train = jnp.ones((10, 20))
        out_train = model.forward(x_train, key)  # Dropout applied

        # Evaluation mode
        model.set_training(False)
        out_eval = model.forward(x_train, key)  # No dropout

    Combining with batching:

    .. code-block:: python

        # Create combined training and batching mode
        training = brainstate.mixin.Training()
        batching = brainstate.mixin.Batching(batch_size=32)
        mode = brainstate.mixin.JointMode(training, batching)

        # Use in training configuration
        class Trainer:
            def __init__(self, model, mode):
                self.model = model
                self.mode = mode

            def train_epoch(self, data):
                if self.mode.has(brainstate.mixin.Training):
                    # Enable training-specific behaviors
                    self.model.set_training(True)

                if self.mode.has(brainstate.mixin.Batching):
                    # Process in batches
                    batch_size = self.mode.batch_size
                    # ... batched training loop ...
                pass
    """
    pass