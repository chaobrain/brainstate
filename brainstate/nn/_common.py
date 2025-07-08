# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
from brainstate.augment._mapping import vmap
from brainstate.typing import Filter
from ._module import Module

AxisName = Hashable

__all__ = [
    'EnvironContext',
    'Vmap',
]


class EnvironContext(Module):
    """
    A wrapper class that provides an environment context for a given layer.

    This class allows execution of a layer within a specific environment context,
    which can be useful for controlling the execution environment of neural network layers.

    This class is equivalent to the following code snippet:

    ```python

    import brainstate

    with brainstate.environ.context(**context):
        result = layer(*args, **kwargs)

    ```

    Attributes:
        layer (Module): The layer to be executed within the environment context.
        context (dict): The environment context parameters.
    """

    def __init__(self, layer: Module, **context):
        """
        Initialize the EnvironContext.

        Args:
            layer (Module): The layer to be wrapped with the environment context.
            **context: Arbitrary keyword arguments representing the environment context parameters.
        """
        super().__init__()

        assert isinstance(layer, Module), 'The layer must be an instance of Module.'
        self.layer = layer
        self.context = context

    def update(self, *args, **kwargs):
        """
        Execute the wrapped layer within the specified environment context.

        Args:
            *args: Variable length argument list to be passed to the wrapped layer.
            **kwargs: Arbitrary keyword arguments to be passed to the wrapped layer.

        Returns:
            The result of executing the wrapped layer within the environment context.
        """
        with environ.context(**self.context):
            return self.layer(*args, **kwargs)

    def add_context(self, **context):
        """
        Add additional environment context parameters to the existing context.

        Args:
            **context: Arbitrary keyword arguments representing the additional environment context parameters.
        """
        self.context.update(context)


def _filter_states(
    module: Module,
    filters: Filter | Dict[Filter, int],
) -> Dict:
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
    """
    A class that applies vectorized mapping (vmap) to a given module.

    This class wraps a module and applies vectorized mapping to its execution,
    allowing for efficient parallel processing across specified axes.

    Args:
        module (Module): The module to be vmapped.
        in_axes (int | None | Sequence[Any], optional): Specifies how to map over inputs. Defaults to 0.
        out_axes (Any, optional): Specifies how to map over outputs. Defaults to 0.
        vmap_states (Filter | Dict[Filter, int], optional): Specifies which states to vmap and on which axes. Defaults to None.
        vmap_out_states (Filter | Dict[Filter, int], optional): Specifies which output states to vmap and on which axes. Defaults to None.
        axis_name (AxisName | None, optional): Name of the axis being mapped over. Defaults to None.
        axis_size (int | None, optional): Size of the axis being mapped over. Defaults to None.
    """

    def __init__(
        self,
        module: Module,
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
        vmap_states: Filter | Dict[Filter, int] = None,
        vmap_out_states: Filter | Dict[Filter, int] = None,
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
        """
        Execute the vmapped module with the given arguments.

        Args:
            *args: Variable length argument list to be passed to the vmapped module.
            **kwargs: Arbitrary keyword arguments to be passed to the vmapped module.

        Returns:
            The result of executing the vmapped module.
        """
        return self.vmapped_fn(*args, **kwargs)
