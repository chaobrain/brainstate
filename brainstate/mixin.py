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

# -*- coding: utf-8 -*-

"""
Mixin classes and utility types for brainstate.

This module provides various mixin classes and custom type definitions that
enhance the functionality of brainstate components. It includes parameter
description mixins, alignment interfaces, and custom type definitions for
expressing complex type requirements.
"""

from typing import Sequence, Optional, TypeVar, Union, _GenericAlias

import jax

__all__ = [
    'Mixin',
    'ParamDesc',
    'ParamDescriber',

    # types
    'JointTypes',
    'OneOfTypes',
    '_JointGenericAlias',
    '_OneOfGenericAlias',
]

T = TypeVar('T')
ArrayLike = jax.typing.ArrayLike


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

        >>> import brainstate
        >>>
        >>> # Hashable objects
        >>> assert brainstate.mixin.hashable(42) == True
        >>> assert brainstate.mixin.hashable("string") == True
        >>> assert brainstate.mixin.hashable((1, 2, 3)) == True
        >>>
        >>> # Non-hashable objects
        >>> assert brainstate.mixin.hashable([1, 2, 3]) == False
        >>> assert brainstate.mixin.hashable({"key": "value"}) == False
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

        >>> import brainstate
        >>>
        >>> class LoggingMixin(brainstate.mixin.Mixin):
        ...     def log(self, message):
        ...         print(f"[{self.__class__.__name__}] {message}")

        >>> class MyComponent(brainstate.nn.Module, LoggingMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def process(self):
        ...         self.log("Processing data...")
        ...         return "Done"
        >>>
        >>> component = MyComponent()
        >>> component.process()  # Prints: [MyComponent] Processing data...
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

        >>> import brainstate
        >>>
        >>> class NeuronModel(brainstate.mixin.ParamDesc):
        ...     def __init__(self, size, tau=10.0, threshold=1.0):
        ...         self.size = size
        ...         self.tau = tau
        ...         self.threshold = threshold
        >>>
        >>> # Create a parameter descriptor
        >>> neuron_desc = NeuronModel.desc(size=100, tau=20.0)
        >>>
        >>> # Use the descriptor to create instances
        >>> neuron1 = neuron_desc(threshold=0.8)  # Creates with threshold=0.8
        >>> neuron2 = neuron_desc(threshold=1.2)  # Creates with threshold=1.2
        >>>
        >>> # Both neurons share size=100, tau=20.0 but have different thresholds

    Creating reusable templates:

    .. code-block:: python

        >>> # Define a template for excitatory neurons
        >>> exc_neuron_template = NeuronModel.desc(size=1000, tau=10.0, threshold=1.0)
        >>>
        >>> # Define a template for inhibitory neurons
        >>> inh_neuron_template = NeuronModel.desc(size=250, tau=5.0, threshold=0.5)
        >>>
        >>> # Create multiple instances from templates
        >>> exc_population = [exc_neuron_template() for _ in range(5)]
        >>> inh_population = [inh_neuron_template() for _ in range(2)]
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

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Regular dict with non-hashable values cannot be hashed
        >>> regular_dict = {"array": jnp.array([1, 2, 3]), "value": 42}
        >>> # hash(regular_dict)  # This would raise TypeError
        >>>
        >>> # HashableDict can be hashed
        >>> hashable = brainstate.mixin.HashableDict(regular_dict)
        >>> key = hash(hashable)  # This works!
        >>>
        >>> # Can be used in sets or as dict keys
        >>> cache = {hashable: "result"}
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

        >>> import brainstate
        >>>
        >>> class Network:
        ...     def __init__(self, n_neurons, learning_rate=0.01):
        ...         self.n_neurons = n_neurons
        ...         self.learning_rate = learning_rate
        >>>
        >>> # Create a descriptor
        >>> network_desc = brainstate.mixin.ParamDescriber(
        ...     Network, n_neurons=1000, learning_rate=0.001
        ... )
        >>>
        >>> # Use the descriptor to create instances with additional args
        >>> net1 = network_desc()
        >>> net2 = network_desc()  # Same configuration

    Using with ParamDesc mixin:

    .. code-block:: python

        >>> class Network(brainstate.mixin.ParamDesc):
        ...     def __init__(self, n_neurons, learning_rate=0.01):
        ...         self.n_neurons = n_neurons
        ...         self.learning_rate = learning_rate
        >>>
        >>> # More concise syntax using the desc() classmethod
        >>> network_desc = Network.desc(n_neurons=1000)
        >>> net = network_desc(learning_rate=0.005)  # Override learning_rate
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

        >>> import brainstate
        >>>
        >>> class BaseModel:
        ...     @brainstate.mixin.not_implemented
        ...     def process(self, x):
        ...         pass
        >>>
        >>> model = BaseModel()
        >>> # model.process(10)  # Raises: NotImplementedError: process is not implemented.
        >>>
        >>> # Check if a method is not implemented
        >>> assert hasattr(BaseModel.process, 'not_implemented')
    """

    def wrapper(*args, **kwargs):
        raise NotImplementedError(f'{func.__name__} is not implemented.')

    # Mark the wrapper so we can detect not-implemented methods
    wrapper.not_implemented = True
    return wrapper


class _JointGenericAlias(_GenericAlias, _root=True):
    """
    Generic alias for JointTypes (intersection types).

    This class represents a type that requires all specified types to be satisfied.
    Unlike _MetaUnionType which creates actual classes with metaclass conflicts,
    this uses typing's generic alias system to avoid metaclass issues.
    """

    def __instancecheck__(self, obj):
        """
        Check if an instance is an instance of all component types.
        """
        return all(isinstance(obj, cls) for cls in self.__args__)

    def __subclasscheck__(self, subclass):
        """
        Check if a class is a subclass of all component types.
        """
        return all(issubclass(subclass, cls) for cls in self.__args__)

    def __eq__(self, other):
        """
        Check equality with another type.

        Two JointTypes are equal if they have the same component types,
        regardless of order.
        """
        if not isinstance(other, _JointGenericAlias):
            return NotImplemented
        return set(self.__args__) == set(other.__args__)

    def __hash__(self):
        """
        Return hash of the JointType.

        The hash is based on the frozenset of component types to ensure
        that JointTypes with the same types (regardless of order) have
        the same hash.
        """
        return hash(frozenset(self.__args__))

    def __repr__(self):
        """
        Return string representation of the JointType.

        Returns a readable representation showing all component types.
        """
        args_str = ', '.join(
            arg.__module__ + '.' + arg.__name__ if hasattr(arg, '__module__') and hasattr(arg, '__name__')
            else str(arg)
            for arg in self.__args__
        )
        return f'JointTypes[{args_str}]'

    def __reduce__(self):
        """
        Support for pickling.

        Returns the necessary information to reconstruct the JointType
        when unpickling.
        """
        return (_JointGenericAlias, (self.__origin__, self.__args__))


class _OneOfGenericAlias(_GenericAlias, _root=True):
    """
    Generic alias for OneOfTypes (union types).

    This class represents a type that requires at least one of the specified
    types to be satisfied. It's similar to typing.Union but provides a consistent
    interface with JointTypes and avoids potential metaclass conflicts.
    """

    def __instancecheck__(self, obj):
        """
        Check if an instance is an instance of any component type.
        """
        return any(isinstance(obj, cls) for cls in self.__args__)

    def __subclasscheck__(self, subclass):
        """
        Check if a class is a subclass of any component type.
        """
        return any(issubclass(subclass, cls) for cls in self.__args__)

    def __eq__(self, other):
        """
        Check equality with another type.

        Two OneOfTypes are equal if they have the same component types,
        regardless of order.
        """
        if not isinstance(other, _OneOfGenericAlias):
            return NotImplemented
        return set(self.__args__) == set(other.__args__)

    def __hash__(self):
        """
        Return hash of the OneOfType.

        The hash is based on the frozenset of component types to ensure
        that OneOfTypes with the same types (regardless of order) have
        the same hash.
        """
        return hash(frozenset(self.__args__))

    def __repr__(self):
        """
        Return string representation of the OneOfType.

        Returns a readable representation showing all component types.
        """
        args_str = ', '.join(
            arg.__module__ + '.' + arg.__name__ if hasattr(arg, '__module__') and hasattr(arg, '__name__')
            else str(arg)
            for arg in self.__args__
        )
        return f'OneOfTypes[{args_str}]'

    def __reduce__(self):
        """
        Support for pickling.

        Returns the necessary information to reconstruct the OneOfType
        when unpickling.
        """
        return (_OneOfGenericAlias, (self.__origin__, self.__args__))


class _JointTypesClass:
    """Helper class to enable subscript syntax for JointTypes."""

    def __call__(self, *types):
        """
        Create a type that requires all specified types (intersection type).

        This function creates a type hint that indicates a value must satisfy all
        the specified types simultaneously. It's useful for expressing complex
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

            >>> import brainstate
            >>> from typing import Protocol
            >>>
            >>> class Trainable(Protocol):
            ...     def train(self): ...
            >>>
            >>> class Evaluable(Protocol):
            ...     def evaluate(self): ...
            >>>
            >>> # A model that is both trainable and evaluable
            >>> TrainableEvaluableModel = brainstate.mixin.JointTypes(Trainable, Evaluable)
            >>> # Or using subscript syntax
            >>> TrainableEvaluableModel = brainstate.mixin.JointTypes[Trainable, Evaluable]
            >>>
            >>> class NeuralNetwork(Trainable, Evaluable):
            ...     def train(self):
            ...         return "Training..."
            ...
            ...     def evaluate(self):
            ...         return "Evaluating..."
            >>>
            >>> model = NeuralNetwork()
            >>> # model satisfies JointTypes(Trainable, Evaluable)

        Using with mixin classes:

        .. code-block:: python

            >>> class Serializable:
            ...     def save(self): pass
            >>>
            >>> class Visualizable:
            ...     def plot(self): pass
            >>>
            >>> # Require both serialization and visualization
            >>> FullFeaturedModel = brainstate.mixin.JointTypes[Serializable, Visualizable]
            >>>
            >>> class MyModel(Serializable, Visualizable):
            ...     def save(self):
            ...         return "Saved"
            ...
            ...     def plot(self):
            ...         return "Plotted"
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

        # Create a generic alias for the joint type
        # This avoids metaclass conflicts by using typing's generic alias system
        return _JointGenericAlias(object, tuple(unique_types))

    def __getitem__(self, item):
        """Enable subscript syntax: JointTypes[Type1, Type2]."""
        if isinstance(item, tuple):
            return self(*item)
        else:
            return self(item)


# Create singleton instance that acts as both a callable and supports subscript
JointTypes = _JointTypesClass()


class _OneOfTypesClass:
    """Helper class to enable subscript syntax for OneOfTypes."""

    def __call__(self, *types):
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

            >>> import brainstate
            >>>
            >>> # A parameter that can be int or float
            >>> NumericType = brainstate.mixin.OneOfTypes(int, float)
            >>> # Or using subscript syntax
            >>> NumericType = brainstate.mixin.OneOfTypes[int, float]
            >>>
            >>> def process_value(x: NumericType):
            ...     return x * 2
            >>>
            >>> # Both work
            >>> result1 = process_value(5)      # int
            >>> result2 = process_value(3.14)   # float

        Using with class types:

        .. code-block:: python

            >>> class NumpyArray:
            ...     pass
            >>>
            >>> class JAXArray:
            ...     pass
            >>>
            >>> # Accept either numpy or JAX arrays
            >>> ArrayType = brainstate.mixin.OneOfTypes[NumpyArray, JAXArray]
            >>>
            >>> def compute(arr: ArrayType):
            ...     if isinstance(arr, NumpyArray):
            ...         return "Processing numpy array"
            ...     elif isinstance(arr, JAXArray):
            ...         return "Processing JAX array"

        Combining with None for optional types:

        .. code-block:: python

            >>> # Optional string (equivalent to Optional[str])
            >>> MaybeString = brainstate.mixin.OneOfTypes[str, type(None)]
            >>>
            >>> def format_name(name: MaybeString) -> str:
            ...     if name is None:
            ...         return "Anonymous"
            ...     return name.title()
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

        # Create a generic alias for the union type
        # This provides consistency with JointTypes and avoids metaclass conflicts
        return _OneOfGenericAlias(Union, tuple(unique_types))

    def __getitem__(self, item):
        """Enable subscript syntax: OneOfTypes[Type1, Type2]."""
        if isinstance(item, tuple):
            return self(*item)
        else:
            return self(item)


# Create singleton instance that acts as both a callable and supports subscript
OneOfTypes = _OneOfTypesClass()


