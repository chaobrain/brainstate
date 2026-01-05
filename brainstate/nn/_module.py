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
All the basic classes for neural networks in ``brainstate``.

The basic classes include:

- ``Module``: The base class for all the objects in the ecosystem.
- ``Sequential``: The class for a sequential of modules, which update the modules sequentially.

"""

from contextlib import contextmanager
from typing import Sequence, Optional, Tuple, Union, TYPE_CHECKING, Callable, Iterable, Iterator

import numpy as np

from brainstate._error import BrainStateError
from brainstate._state import catch_new_states, NewStateCatcher, ParamState, ShortTermState
from brainstate.graph import Node, states, nodes, flatten
from brainstate.mixin import ParamDescriber, ParamDesc, not_implemented, is_not_implemented
from brainstate.typing import Size
from brainstate.util import FlattedDict, NestedDict

__all__ = [
    'Module',
    'ElementWiseBlock',
    'Sequential',
]

# maximum integer
max_int = np.iinfo(np.int32).max


class Module(Node, ParamDesc):
    """
    Base class for neural network modules in BrainState.

    ``Module`` is a graph node with utilities for traversing submodules,
    collecting state, and exposing parameters. Subclasses implement
    ``update()`` to define the module's behavior; calling a module invokes
    ``update()`` directly.

    Parameters
    ----------
    name : str, optional
        Optional display name for the module. Read-only after construction.

    Attributes
    ----------
    name : str
        Module name (read-only).
    in_size : Size or None
        Expected input size tuple if known.
    out_size : Size or None
        Expected output size tuple if known.

    Notes
    -----
    - ``states()`` and ``state_trees()`` collect ``State`` objects from this
      module and its children (with optional filters).
    - ``nodes()``, ``children()``, and ``named_children()`` traverse submodules.
    - ``par_modules()``, ``parameters()``, and ``named_parameters()`` expose
      parameter containers for training or inspection.
    - ``init_state()`` and ``reset_state()`` are optional hooks for stateful
      modules.
    - ``__call__`` forwards to ``update()`` and ``x >> module`` is supported.

    Examples
    --------
    >>> import brainstate
    >>>
    >>> class Scale(brainstate.nn.Module):
    ...     def __init__(self, scale):
    ...         super().__init__()
    ...         self.scale = scale
    ...     def update(self, x):
    ...         return x * self.scale
    >>>
    >>> layer = Scale(2.0)
    >>> layer(3.0)
    6.0
    """

    __module__ = 'brainstate.nn'

    _in_size: Optional[Size]
    _out_size: Optional[Size]
    _name: Optional[str]

    if not TYPE_CHECKING:
        def __init__(self, name: str = None):
            # check the name
            if name is not None:
                assert isinstance(name, str), f'The name must be a string, but we got {type(name)}: {name}'
            self._name = name

            # input and output size
            self._in_size = None
            self._out_size = None

    @property
    def name(self):
        """Name of the model."""
        return self._name

    @name.setter
    def name(self, name: str = None):
        raise AttributeError('The name of the model is read-only.')

    @property
    def in_size(self) -> Size:
        return self._in_size

    @in_size.setter
    def in_size(self, in_size: Sequence[int] | int):
        if isinstance(in_size, int):
            in_size = (in_size,)
        elif isinstance(in_size, np.generic):
            if np.issubdtype(in_size, np.integer) and in_size.ndim == 0:
                in_size = (int(in_size),)
        assert isinstance(in_size, (tuple, list)), f"Invalid type of in_size: {in_size} {type(in_size)}"
        self._in_size = tuple(in_size)

    @property
    def out_size(self) -> Size:
        return self._out_size

    @out_size.setter
    def out_size(self, out_size: Sequence[int] | int):
        if isinstance(out_size, int):
            out_size = (out_size,)
        elif isinstance(out_size, np.ndarray):
            if np.issubdtype(out_size, np.integer) and out_size.ndim == 0:
                out_size = (int(out_size),)
        assert isinstance(out_size, (tuple, list)), f"Invalid type of out_size: {type(out_size)}"
        self._out_size = tuple(out_size)

    @not_implemented
    def update(self, *args, **kwargs):
        """
        The function to specify the updating rule.

        Default implementation returns first argument unchanged (identity function).
        Override this method in subclasses to implement custom behavior.

        Parameters
        ----------
        *args : Any
            Positional arguments (typically input data).
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        output : Any
            Transformed output. Default implementation returns args[0] if available,
            otherwise None.
        """
        if is_not_implemented(self.forward):
            raise NotImplementedError(
                f'Subclass of {self.__class__.__name__} must implement "update" function. \n'
                f'This instance is: \n'
                f'{self}'
            )
        else:
            states = self.get_states()
            params = self.get_params()
            states, out = self.forward(states, params, *args, **kwargs)
            for path, state in self.states(ShortTermState).items():
                name = '.'.join([str(n) for n in path])
                if name not in states:
                    raise BrainStateError(f'State "{name}" not found in provided states.')
                state.value = states[name]
            return out

    def __pretty_repr_item__(self, name, value):
        if name.startswith('_'):
            return None if value is None else (name[1:], value)  # skip the first `_`
        return name, value

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def __rrshift__(self, other):
        """
        Support using right shift operator to call modules.

        Examples
        --------

        >>> import brainstate as brainstate
        >>> x = brainstate.random.rand((10, 10))
        >>> l = brainstate.nn.Dropout(0.5)
        >>> y = x >> l
        """
        return self.__call__(other)

    def states(
        self,
        *filters,
        allowed_hierarchy: Tuple[int, int] = (0, max_int),
    ) -> FlattedDict | Tuple[FlattedDict, ...]:
        """
        Collect all states in this node and the children nodes.

        Parameters
        ----------
        filters : Any
          The filters to select the states.
        allowed_hierarchy : tuple of int
          The hierarchy of the states to be collected.
        level : int
          The level of the states to be collected. Has been deprecated.

        Returns
        -------
        states : FlattedDict, tuple of FlattedDict
          The collection contained (the path, the state).
        """
        return states(self, *filters, allowed_hierarchy=allowed_hierarchy)

    def state_trees(
        self,
        *filters,
    ) -> NestedDict | Tuple[NestedDict, ...]:
        """
        Collect all states in this node and the children nodes.

        Parameters
        ----------
        filters : tuple
          The filters to select the states.

        Returns
        -------
        states : FlattedDict, tuple of FlattedDict
          The collection contained (the path, the state).
        """
        graph_def, state_tree = flatten(self)
        if len(filters):
            return state_tree.filter(*filters)
        return state_tree

    def nodes(
        self,
        *filters,
        allowed_hierarchy: Tuple[int, int] = (0, max_int),
    ) -> FlattedDict | Tuple[FlattedDict, ...]:
        """
        Collect all children nodes.

        Parameters
        ----------
        filters : Any
          The filters to select the states.
        allowed_hierarchy : tuple of int
          The hierarchy of the states to be collected.

        Returns
        -------
        nodes : FlattedDict, tuple of FlattedDict
          The collection contained (the path, the node).
        """
        return nodes(self, *filters, allowed_hierarchy=allowed_hierarchy)

    def children(self) -> Iterator['Module']:
        """
        Return immediate child modules.

        Similar to PyTorch's nn.Module.children().

        Returns
        -------
        children : Iterable
            Dictionary of immediate child modules.

        Examples
        --------
        >>> for child in model.children():
        ...     print(type(child))
        """
        for _name, module in self.named_children():
            yield module

    def named_children(self):
        """
        Return an iterator over immediate child modules, yielding name and module.

        Similar to PyTorch's nn.Module.named_children().

        Yields
        ------
        name : str
            Name of the child module.
        module : Module
            Child module.

        Examples
        --------
        >>> for name, child in model.named_children():
        ...     print(f"{name}: {type(child).__name__}")
        """
        children_dict = self.nodes(Module, allowed_hierarchy=(1, 1))
        for path, child in children_dict.items():
            # Convert path tuple to dot-separated string
            name = '.'.join(str(p) for p in path)
            yield name, child

    def modules(self, include_self: bool = True) -> Iterator['Module']:
        """
        Return all modules in the network.

        Similar to PyTorch's nn.Module.modules().

        Parameters
        ----------
        include_self : bool
            Whether to include the module itself. Default is True.

        Returns
        -------
        modules : Iterator
            Dictionary of all modules in the tree.

        Examples
        --------
        >>> for module in model.modules().values():
        ...     print(type(module))
        """
        for _, module in self.named_modules(include_self=include_self):
            yield module

    def named_modules(self, prefix: str = '', include_self: bool = True):
        """
        Return an iterator over all modules in the network, yielding name and module.

        Similar to PyTorch's nn.Module.named_modules().

        Parameters
        ----------
        prefix : str
            Prefix to prepend to all module names. Default is ''.
        include_self : bool
            Whether to include the module itself. Default is True.

        Yields
        ------
        name : str
            Name of the module (with prefix if provided).
        module : Module
            Module in the tree.

        Examples
        --------
        >>> for name, module in model.named_modules():
        ...     print(f"{name}: {type(module).__name__}")
        """
        if include_self:
            modules_dict = self.nodes(allowed_hierarchy=(0, max_int))
        else:
            modules_dict = self.nodes(allowed_hierarchy=(1, max_int))

        for path, module in modules_dict.items():
            # Convert path tuple to dot-separated string
            name = '.'.join(str(p) for p in path) if path else ''

            # Add prefix if provided
            if prefix:
                name = f"{prefix}.{name}" if name else prefix

            yield name, module

    def param_modules(
        self, allowed_hierarchy: Tuple[int, int] = (0, max_int),
    ) -> Iterator['Param']:
        """
        Collect all Param parameters in this module and children.

        Parameters
        ----------
        allowed_hierarchy : tuple of int
          The hierarchy of the parameters to be collected.

        Returns
        -------
        params : FlattedDict, tuple of FlattedDict
          The collection contained (the path, the Param parameter).

        Examples
        --------
        >>> # Get all parameters
        >>> all_params = model.param_modules()
        >>>
        >>> # Get parameters with transforms
        >>> from brainstate.nn import IdentityT
        >>> transformed = model.param_modules(lambda path, p: not isinstance(p.t, IdentityT))
        >>>
        >>> # Get parameters with regularization
        >>> regularized = model.param_modules(lambda path, p: p.reg is not None)
        """
        for name, module in self.named_param_modules(allowed_hierarchy=allowed_hierarchy):
            yield module

    def named_param_modules(self, allowed_hierarchy: Tuple[int, int] = (0, max_int), ):
        """
        Iterate over (name, parameter) pairs.

        Parameters
        ----------
        allowed_hierarchy: tuple of int
            The hierarchy of the parameters to be collected.

        Yields
        ------
        name : str
            Dot-separated path to the parameter.
        param : Param
            The parameter instance.

        Examples
        --------
        >>> for name, param in model.named_param_modules():
        ...     print(f"{name}: {param.value().shape}")
        layer1.weight: (10, 20)
        layer1.bias: (20,)
        layer2.weight: (20, 5)
        """
        from ._param import Param
        params_dict = self.nodes(Param, allowed_hierarchy=allowed_hierarchy)
        for path, param in params_dict.items():
            # Convert path tuple to dot-separated string
            name = '.'.join(str(p) for p in path)
            yield name, param

    def reg_loss(self):
        """
        Compute total regularization loss from all Param parameters.

        Returns
        -------
        loss : array_like
            Scalar total regularization loss (sum of all reg losses).

        Examples
        --------
        >>> # Get total regularization loss
        >>> reg_penalty = model.reg_loss()
        >>> total_loss = data_loss + reg_penalty
        >>>
        >>> # Get loss only from L1-regularized params
        >>> from brainstate.nn import L1Reg
        >>> l1_loss = model.reg_loss(lambda path, p: isinstance(p.reg, L1Reg))
        """
        param_dict = tuple(self.param_modules())
        if len(param_dict) == 0:
            return 0.0
        losses = [param.reg_loss() for param in param_dict]
        return sum(losses)

    @contextmanager
    def param_precompute(
        self,
        allowed_hierarchy: Tuple[int, int] = (0, max_int),
    ):
        """
        Context manager to temporarily cache all Param parameters.

        This context manager warms up (caches) all parameter transformations
        on entry and clears all caches on exit, ensuring efficient computation
        within the context while maintaining clean state outside.

        Parameters
        ----------
        allowed_hierarchy : tuple of int, optional
            The hierarchy range of parameters to cache, specified as (min_level, max_level).
            Default is (0, max_int) to cache all parameters at all levels.

        Yields
        ------
        None
            This context manager doesn't yield any value but provides
            a cached parameter environment for the enclosed code block.

        Examples
        --------
        Cache all parameters during computation:

        >>> class MyModule(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.param = brainstate.nn.Param(
        ...             jnp.ones(100),
        ...             t=brainstate.nn.SoftplusT()
        ...         )

        >>> model = MyModule()
        >>> with model.param_precompute():
        ...     # First access computes and caches
        ...     val1 = model.param.value()
        ...     # Subsequent accesses use cache (fast!)
        ...     val2 = model.param.value()
        ...     val3 = model.param.value()
        >>> # Cache is automatically cleared here

        Cache only immediate child parameters:

        >>> with model.param_precompute(allowed_hierarchy=(1, 1)):
        ...     # Only level-1 params are cached
        ...     result = model(input_data)

        Exception safety - cache is cleared even on errors:

        >>> try:
        ...     with model.param_precompute():
        ...         result = model(data)
        ...         raise ValueError("Something went wrong")
        ... except ValueError:
        ...     pass
        >>> # Parameter caches are still cleared

        Notes
        -----
        - The context manager is thread-safe (Param's cache uses RLock)
        - Caches are automatically invalidated on parameter updates
        - Exception safety is guaranteed - caches are cleared even if
          exceptions occur within the context
        - For performance-critical code, cache all parameters before
          entering a tight loop or JIT-compiled function

        See Also
        --------
        Param.cache : Manually cache a single parameter
        Param.clear_cache : Manually clear a single parameter's cache
        """
        # Cache all parameters in the hierarchy
        for par_module in self.param_modules(allowed_hierarchy=allowed_hierarchy):
            par_module.cache()

        try:
            yield
        finally:
            # Always clear caches on exit, even if exception occurred
            for par_module in self.param_modules(allowed_hierarchy=allowed_hierarchy):
                par_module.clear_cache()

    def init_state(self, *args, **kwargs):
        """
        State initialization function.
        """
        pass

    def reset_state(self, *args, **kwargs):
        """
        State resetting function.
        """
        pass

    def init_all_states(
        self, *args, state_tag: str = None, **kwargs
    ) -> NewStateCatcher:
        from ._collective_ops import init_all_states
        with catch_new_states(state_tag=state_tag) as catcher:
            init_all_states(self, *args, **kwargs)
        return catcher

    def parameters(self, recurse: bool = True) -> Iterator[ParamState]:
        """
        Return module parameters.

        PyTorch-compatible alias for para_modules(). Returns Param instances.

        Parameters
        ----------
        recurse : bool
            If True, yields parameters of this module and all submodules.
            Otherwise, yields only parameters that are direct attributes of this module.
            Default is True.

        Returns
        -------
        parameters : FlattedDict
            Dictionary of parameters.

        Examples
        --------
        >>> for param in model.parameters():
        ...     print(param.value.shape)

        See Also
        --------
        para_modules : Native brainstate method for parameter discovery
        named_parameters : Returns (name, parameter) pairs
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Return an iterator over module parameters, yielding name and parameter.

        PyTorch-compatible alias for named_para_modules().

        Parameters
        ----------
        prefix : str
            Prefix to prepend to all parameter names. Default is ''.
        recurse : bool
            If True, yields parameters of this module and all submodules.
            Otherwise, yields only parameters that are direct attributes of this module.
            Default is True.

        Yields
        ------
        name : str
            Name of the parameter (with prefix if provided).
        param : Param
            Parameter instance.

        Examples
        --------
        >>> for name, param in model.named_parameters():
        ...     print(f"{name}: {param.value().shape}")

        See Also
        --------
        named_para_modules : Native brainstate method for named parameter iteration
        parameters : Returns parameters only
        """
        if recurse:
            params_dict = self.states(ParamState)
        else:
            params_dict = self.states(ParamState, allowed_hierarchy=(1, 1))

        for path, param in params_dict.items():
            # Convert path tuple to dot-separated string
            name = '.'.join(str(p) for p in path)

            # Add prefix if provided
            if prefix:
                name = f"{prefix}.{name}"

            yield name, param


class ElementWiseBlock(Module):
    __module__ = 'brainstate.nn'


class Sequential(Module):
    """
    A sequential `input-output` module.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``dict`` of modules can be
    passed in. The ``update()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :py:class:`Container`? A ``Container`` is exactly what it
    sounds like--a container to store :py:class:`DynamicalSystem` s!
    On the other hand, the layers in a ``Sequential`` are connected
    in a cascading way.

    Examples
    --------

    >>> import jax
    >>> import brainstate as brainstate
    >>> import brainstate.nn as nn
    >>>
    >>> # composing ANN models
    >>> l = nn.Sequential(nn.Linear(100, 10),
    >>>                   jax.nn.relu,
    >>>                   nn.Linear(10, 2))
    >>> l(brainstate.random.random((256, 100)))

    Args:
      modules_as_tuple: The children modules.
      modules_as_dict: The children modules.
      name: The object name.
    """
    __module__ = 'brainstate.nn'

    def __init__(self, first: Module, *layers):
        super().__init__()
        self.layers = []

        # add all modules
        assert isinstance(first, Module), 'The first module should be an instance of Module.'
        in_size = first.out_size
        self.layers.append(first)
        for module in layers:
            module, in_size = self._format_module(module, in_size)
            self.layers.append(module)

        # the input and output shape
        if first.in_size is not None:
            self.in_size = first.in_size
        if in_size is not None:
            self.out_size = tuple(in_size)

    def update(self, x):
        """Update function of a sequential model.
        """
        for m in self.layers:
            try:
                x = m(x)
            except Exception as e:
                raise BrainStateError(
                    f'The module \n'
                    f'{m}\n'
                    f'failed to update with input {x}\n'
                ) from e
        return x

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, slice):
            return Sequential(*self.layers[key])
        elif isinstance(key, int):
            return self.layers[key]
        elif isinstance(key, (tuple, list)):
            return Sequential(*[self.layers[k] for k in key])
        else:
            raise KeyError(f'Unknown type of key: {type(key)}')

    def append(self, layer: Callable):
        """
        Append a layer to the sequential model.

        This method adds a new layer to the end of the sequential model. The layer can be
        either a Module instance, an ElementWiseBlock instance, or a callable function. If the
        layer is a callable function, it will be wrapped in an ElementWiseBlock instance.

        Parameters:
        ----------
        layer : Callable
            The layer to be appended to the sequential model. It can be a Module instance,
            an ElementWiseBlock instance, or a callable function.

        Raises:
        -------
        ValueError
            If the sequential model is empty and the first layer is a callable function.

        Returns:
        --------
        None
            The method does not return any value. It modifies the sequential model by adding
            the new layer to the end.
        """
        if len(self.layers) == 0:
            raise ValueError('The first layer should be a module, not a function.')
        module, in_size = self._format_module(layer, self.out_size)
        self.layers.append(module)
        self.out_size = in_size

    def extend(self, modules):
        """
        Append modules from an iterable to the end of the sequential model.

        This method adds multiple modules to the end of the sequential model. Each module
        is processed and validated, with automatic size inference between layers.

        Parameters:
        ----------
        modules : iterable
            An iterable of modules to append (e.g., list, tuple). Each element can be
            a Module instance, an ElementWiseBlock instance, or a callable function.

        Raises:
        -------
        ValueError
            If the sequential model is empty and the first module is not a Module instance.

        Returns:
        --------
        None
            The method does not return any value. It modifies the sequential model by adding
            the new modules to the end.

        Examples:
        --------
        >>> import brainstate
        >>> seq = brainstate.nn.Sequential(brainstate.nn.Linear(10, 20))
        >>> seq.extend([brainstate.nn.ReLU(), brainstate.nn.Linear(20, 5)])
        """
        if len(self.layers) == 0:
            raise ValueError('Cannot extend an empty Sequential. Use __init__ to add the first module.')

        current_size = self.out_size
        for module in modules:
            module, current_size = self._format_module(module, current_size)
            self.layers.append(module)

        if current_size is not None:
            self.out_size = current_size

    def insert(self, index: int, module):
        """
        Insert a module at a specific position in the sequential model.

        This method inserts a module at the specified index position. After insertion,
        all output sizes for modules from the insertion point onwards are recalculated
        to maintain the size inference chain.

        Parameters:
        ----------
        index : int
            Position to insert the module. Supports negative indices following Python
            list convention (e.g., -1 for before the last element).
        module : Module or Callable
            The module to insert. Can be a Module instance, an ElementWiseBlock instance,
            or a callable function.

        Raises:
        -------
        ValueError
            If the sequential model is empty and index is not 0.
        IndexError
            If the index is out of range.

        Returns:
        --------
        None
            The method does not return any value. It modifies the sequential model by inserting
            the module at the specified position.

        Examples:
        --------
        >>> import brainstate
        >>> seq = brainstate.nn.Sequential(brainstate.nn.Linear(10, 20), brainstate.nn.Linear(20, 5))
        >>> seq.insert(1, brainstate.nn.ReLU())  # Insert ReLU between the two Linear layers
        >>> seq.insert(-1, brainstate.nn.Dropout(0.5))  # Insert Dropout before the last layer
        """
        # Handle empty Sequential
        if len(self.layers) == 0:
            if index != 0:
                raise ValueError('Cannot insert into empty Sequential at non-zero index. Use __init__ or index=0.')
            if not isinstance(module, Module):
                raise ValueError('The first module in Sequential must be a Module instance, not a callable.')
            self.layers.append(module)
            if module.in_size is not None:
                self.in_size = module.in_size
            if module.out_size is not None:
                self.out_size = tuple(module.out_size)
            return

        # Normalize negative index
        n = len(self.layers)
        if index < 0:
            index = max(0, n + index + 1)

        # Validate index range [0, n] (inclusive of n for appending)
        if index < 0 or index > n:
            raise IndexError(f'Index {index} is out of range for Sequential with {n} layers.')

        # Determine input size for the new module
        if index == 0:
            # Inserting at the beginning
            in_size = self.in_size
        else:
            # Inserting after an existing module
            in_size = self.layers[index - 1].out_size

        # Format and insert the new module
        formatted_module, out_size = self._format_module(module, in_size)
        self.layers.insert(index, formatted_module)

        # Recalculate sizes for all modules from insertion point onwards
        current_size = out_size
        for i in range(index + 1, len(self.layers)):
            self.layers[i], current_size = self._format_module(self.layers[i], current_size)

        # Update in_size if inserted at beginning
        if index == 0 and formatted_module.in_size is not None:
            self.in_size = formatted_module.in_size

        # Update out_size with final output
        if current_size is not None:
            self.out_size = tuple(current_size)

    def _format_module(self, module, in_size):
        try:
            if isinstance(module, ParamDescriber):
                if in_size is None:
                    raise ValueError(
                        'The input size should be specified. '
                        f'Please set the in_size attribute of the previous module: \n'
                        f'{self.layers[-1]}'
                    )
                module = module(in_size=in_size)
                assert isinstance(module, Module), 'The module should be an instance of Module.'
                out_size = module.out_size
            elif isinstance(module, ElementWiseBlock):
                out_size = in_size
            elif isinstance(module, Module):
                out_size = module.out_size
            elif callable(module):
                out_size = in_size
            else:
                raise TypeError(f"Unsupported type {type(module)}. ")
        except Exception as e:
            raise BrainStateError(
                f'Failed to format the module: \n'
                f'{module}\n'
                f'with input size: {in_size}\n'
            ) from e
        return module, out_size
