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

import warnings
from typing import Sequence, Optional, Tuple, Union, TYPE_CHECKING, Callable

import numpy as np

from brainstate._error import BrainStateError
from brainstate._state import State
from brainstate.graph import Node, states, nodes, flatten
from brainstate.mixin import ParamDescriber, ParamDesc
from brainstate.typing import PathParts, Size
from brainstate.util import FlattedDict, NestedDict

# maximum integer
max_int = np.iinfo(np.int32).max

__all__ = [
    'Module', 'ElementWiseBlock', 'Sequential',
]


class Module(Node, ParamDesc):
    """
    The Module class for the whole ecosystem.

    The ``Module`` is the base class for all the objects in the ecosystem. It
    provides the basic functionalities for the objects, including:

    - ``states()``: Collect all states in this node and the children nodes.
    - ``nodes()``: Collect all children nodes.
    - ``update()``: The function to specify the updating rule.
    - ``init_state()``: State initialization function.
    - ``reset_state()``: State resetting function.

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

    def update(self, *args, **kwargs):
        """
        The function to specify the updating rule.
        """
        raise NotImplementedError(
            f'Subclass of {self.__class__.__name__} must implement "update" function. \n'
            f'This instance is: \n'
            f'{self}'
        )

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
        level: int = None,
    ) -> FlattedDict[PathParts, State] | Tuple[FlattedDict[PathParts, State], ...]:
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
        if level is not None:
            allowed_hierarchy = (0, level)
            warnings.warn('The "level" argument is deprecated. Please use "allowed_hierarchy" instead.',
                          DeprecationWarning)

        return states(self, *filters, allowed_hierarchy=allowed_hierarchy)

    def state_trees(
        self,
        *filters,
    ) -> NestedDict[PathParts, State] | Tuple[NestedDict[PathParts, State], ...]:
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
        level: int = None,
    ) -> FlattedDict[PathParts, Node] | Tuple[FlattedDict[PathParts, Node], ...]:
        """
        Collect all children nodes.

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
        nodes : FlattedDict, tuple of FlattedDict
          The collection contained (the path, the node).
        """
        if level is not None:
            allowed_hierarchy = (0, level)
            warnings.warn('The "level" argument is deprecated. Please use "allowed_hierarchy" instead.',
                          DeprecationWarning)

        return nodes(self, *filters, allowed_hierarchy=allowed_hierarchy)

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

    def __pretty_repr_item__(self, name, value):
        if name.startswith('_'):
            return None if value is None else (name[1:], value)  # skip the first `_`
        return name, value


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
