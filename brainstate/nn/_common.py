# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

from collections import defaultdict
from typing import Any, Sequence, Hashable, Dict

from brainstate import environ
from brainstate.transform import vmap
from brainstate.typing import Filter
from ._module import Module

AxisName = Hashable

__all__ = [
    'EnvironContext',
    'Vmap',
]


class EnvironContext(Module):
    """Wrap a module so it executes inside a brainstate environment context.

    Parameters
    ----------
    layer : Module
        Module executed within the environment context.
    **context
        Keyword arguments forwarded to ``brainstate.environ.context``.

    Attributes
    ----------
    layer : Module
        Wrapped module executed inside the context.
    context : dict
        Environment arguments applied to the wrapped module.

    Examples
    --------
    .. code-block:: python

       >>> import brainstate
       >>> from brainstate.nn import EnvironContext
       >>> wrapped = EnvironContext(layer, fit=True)
       >>> result = wrapped.update(inputs)
    """

    def __init__(self, layer: Module, **context):
        """Initialize the wrapper with a module and environment arguments.

        Parameters
        ----------
        layer : Module
            Module executed inside the environment context.
        **context
            Keyword arguments forwarded to ``brainstate.environ.context``.
        """
        super().__init__()

        assert isinstance(layer, Module), 'The layer must be an instance of Module.'
        self.layer = layer
        self.context = context

    def update(self, *args, context: Dict = None, **kwargs):
        """Execute the wrapped module inside the environment context.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the wrapped module.
        **kwargs
            Keyword arguments forwarded to the wrapped module.
        context: dict, optional
            Additional environment settings for this call. Merged with the
            stored context.

        Returns
        -------
        Any
            Result returned by the wrapped module.
        """
        if context is None:
            context = dict()
        with environ.context(**self.context, **context):
            return self.layer(*args, **kwargs)

    def add_context(self, **context):
        """Add more environment settings to the wrapped module.

        Parameters
        ----------
        **context
            Keyword arguments merged into the stored environment context.
        """
        self.context.update(context)


def _filter_states(
    module: Module,
    filters: Filter | Dict[Filter, int],
) -> Dict:
    """Normalize state filter specifications for ``Module.states``.

    Parameters
    ----------
    module : Module
        Module providing the states interface.
    filters : Filter or dict[Filter, int]
        Filters passed by the caller. Dictionary keys are filters and values
        are the axes they should map over.

    Returns
    -------
    dict[int, Any] or Any or None
        Structured filters to pass to ``Module.states``. Returns ``None`` when
        no filtering is requested.
    """
    if filters is None:
        filtered_states = None
    elif isinstance(filters, dict):
        in_states_filter = defaultdict(list)
        for filter_, axis in filters:
            assert isinstance(axis, int), 'The value of in_states must be the map axis, which should be an integer.'
            in_states_filter[axis].append(filter_)
        filtered_states = module.states(*in_states_filter.values())
        in_states_axis = tuple(in_states_filter.keys())
        filtered_states = {axis: states for axis, states in zip(in_states_axis, filtered_states)}
    else:
        filtered_states = module.states(filters)
    return filtered_states


class Vmap(Module):
    """Vectorize a module with ``brainstate.transform.vmap``.

    Parameters
    ----------
    module : Module
        Module to wrap with vectorized mapping.
    in_axes : int or None or Sequence[Any], optional
        Specification for mapping over inputs. Defaults to ``0``.
    out_axes : Any, optional
        Specification for mapping over outputs. Defaults to ``0``.
    axis_name : AxisName or None, optional
        Name of the axis being mapped. Defaults to ``None``.
    axis_size : int or None, optional
        Size of the mapped axis. Defaults to ``None``.

    Examples
    --------
    .. code-block:: python

       >>> from brainstate.nn import Vmap
       >>> vmapped = Vmap(module, in_axes=0, axis_name="batch")
       >>> outputs = vmapped.update(inputs)
    """

    def __init__(
        self,
        module: Module,
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
        vmap_states: Filter | Dict[int, Filter] = None,
        vmap_out_states: Dict[int, Dict] | Any | None = None,
        axis_name: AxisName | None = None,
        axis_size: int | None = None,
    ):
        super().__init__()

        # parameters
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.axis_name = axis_name
        self.axis_size = axis_size
        assert isinstance(module, Module), 'The module must be an instance of Module.'
        self.module = module
        vmap_states = _filter_states(module, vmap_states)
        vmap_out_states = _filter_states(module, vmap_out_states)

        @vmap(
            in_axes=in_axes,
            out_axes=out_axes,
            in_states=vmap_states,
            out_states=vmap_out_states,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        def vmap_run(*args, **kwargs):
            return module(*args, **kwargs)

        # vmapped module
        self.vmapped_fn = vmap_run

    def update(self, *args, **kwargs):
        """Execute the vmapped module with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the vmapped module.
        **kwargs
            Keyword arguments forwarded to the vmapped module.

        Returns
        -------
        Any
            Result of executing the vmapped module.
        """
        return self.vmapped_fn(*args, **kwargs)
