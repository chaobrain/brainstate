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
from typing import Any, Sequence, Hashable, Dict, Optional, Callable

from brainstate import environ
from brainstate._state import State
from brainstate.transform import vmap, vmap2, vmap2_new_states, pmap2, pmap2_new_states
from brainstate.typing import Filter
from brainstate.util import filter
from ._module import Module

AxisName = Hashable

__all__ = [
    'EnvironContext',
    'Vmap',
    'ModuleMapper',
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
    """
    Vectorize a module with ``brainstate.transform.vmap``.

    This wrapper applies vectorized mapping over a module, enabling efficient
    batch processing by automatically mapping over specified axes of inputs
    and states.

    Parameters
    ----------
    module : Module
        Module to wrap with vectorized mapping.
    in_axes : int or None or Sequence[Any], optional
        Specification for mapping over inputs. Defaults to ``0``.
    out_axes : Any, optional
        Specification for mapping over outputs. Defaults to ``0``.
    vmap_states : Filter or Dict[int, Filter], optional
        Filter specifying which states should be mapped. Can be a single filter
        or a dictionary mapping axes (int) to filters. Defaults to ``None``.
    vmap_out_states : Dict[int, Dict] or Any or None, optional
        Specification for how to map output states. Can be a dictionary mapping
        axes to state specifications. Defaults to ``None``.
    axis_name : AxisName or None, optional
        Name of the axis being mapped. Defaults to ``None``.
    axis_size : int or None, optional
        Size of the mapped axis. Defaults to ``None``.

    Attributes
    ----------
    module : Module
        The wrapped module being vectorized.
    vmapped_fn : Callable
        The vectorized function that executes the module.

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

        assert isinstance(module, Module), 'The module must be an instance of Module.'
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.axis_name = axis_name
        self.axis_size = axis_size
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


class ToPredicate:
    """Helper predicate class for filtering states by identity.

    This class creates a predicate that matches states based on their object
    identity (id), used internally for state filtering in vectorized mapping.

    Parameters
    ----------
    states : Iterable[State]
        Collection of states to match against.

    Attributes
    ----------
    state_ids : set
        Set of state object IDs for efficient lookup.
    """

    def __init__(self, states):
        self.state_ids = set([id(st) for st in states])

    def __call__(self, path, st: State):
        """Check if a state matches the predicate.

        Parameters
        ----------
        path : Any
            Path to the state (unused).
        st : State
            State to check.

        Returns
        -------
        bool
            True if the state's ID is in the predicate's state set.
        """
        return id(st) in self.state_ids


class _ModuleMapperCalling:
    def __init__(
        self,
        fn: Callable,
        behavior: str,
        in_axes: Any = 0,
        out_axes: Any = 0,
        axis_name: Optional[str] = None,
        state_axes: Dict[int, Filter] = None,
    ):
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.axis_name = axis_name
        self.state_axes = state_axes
        self.behavior = behavior

        if behavior == 'vmap':
            map_fn = vmap2
        elif behavior == 'pmap':
            map_fn = pmap2
        else:
            raise ValueError(
                'Invalid behavior specified. Must be "vmap" or "pmap".'
            )

        self.map_fn = map_fn(
            fn,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            state_in_axes=state_axes,
            state_out_axes=state_axes,
        )

    def __call__(self, *args, **kwargs):
        return self.map_fn(*args, **kwargs)


class ModuleMapper(Module):
    """
    Vectorize or parallelize a module using ``brainstate.transform.vmap2`` or ``pmap2``.

    This wrapper provides enhanced control over state management during vectorized
    or parallel mapping operations. Unlike ``Vmap``, ``ModuleMapper`` requires
    explicit initialization of vectorized states before use, enabling fine-grained
    control over how states are distributed across mapping axes.

    The class supports two modes of operation:

    - ``behavior='vmap'``: Vectorized mapping using :func:`vmap2` for single-device batching
    - ``behavior='pmap'``: Parallel mapping using :func:`pmap2` for multi-device parallelization

    Parameters
    ----------
    module : Module
        Module to wrap with vectorized or parallel mapping.
    init_map_size : int
        Size of the mapping axis used during state initialization. This determines
        how many copies of each state will be created.
    init_state_axes : Dict[int, Filter], optional
        Dictionary mapping axis indices to filters for state initialization.
        Controls how newly created states are distributed across axes. Defaults to
        ``None``, which assigns all states to axis 0 except :class:`NonBatchState`.
    state_tag : str, optional
        Tag for identifying and grouping states during vectorization.
        Defaults to ``None``.
    in_axes : int or Sequence[Any], optional
        Specification for mapping over inputs during ``update`` calls, following
        the semantics of :func:`jax.vmap`. Defaults to ``0``.
    out_axes : Any, optional
        Specification for mapping over outputs during ``update`` calls.
        Defaults to ``0``.
    axis_name : AxisName or None, optional
        Name of the mapped axis used by collective primitives (e.g., ``lax.psum``).
        Defaults to ``None``.
    spmd_axis_name : AxisName or None, optional
        Name for SPMD (Single Program Multiple Data) axis when using nested
        mapping transforms. Defaults to ``None``.
    call_state_axes : Dict[int, Filter], optional
        Dictionary mapping axes to filters for states during ``update`` calls.
        Specifies how states should be mapped over different axes. This is
        automatically integrated with states created during initialization.
        Defaults to ``None``.
    behavior : {'vmap', 'pmap'}, default 'vmap'
        Type of parallelization to use. ``'vmap'`` for vectorized single-device
        mapping, ``'pmap'`` for multi-device parallel mapping.

    Attributes
    ----------
    module : Module
        The wrapped module being vectorized or parallelized.
    init_map_size : int
        Size of the mapping axis for state initialization.
    dict_vmap_states : dict[int, list[State]] or None
        Dictionary mapping axis indices to lists of vectorized states, populated
        after calling :meth:`init_all_states`.

    Raises
    ------
    ValueError
        If ``behavior`` is not ``'vmap'`` or ``'pmap'``, or if ``update`` is called
        before :meth:`init_all_states`.
    AssertionError
        If ``init_map_size`` is not an integer.

    Notes
    -----
    This module requires calling :meth:`init_all_states` before the first
    :meth:`update` call. The initialization process:

    1. Calls ``module.init_all_states(**kwargs)`` under vectorized/parallel mapping
    2. Captures all newly created states
    3. Distributes states across axes based on ``init_state_axes``
    4. Integrates these states into ``call_state_axes`` for subsequent ``update`` calls

    Examples
    --------
    **Basic vectorized mapping:**

    .. code-block:: python

       >>> import brainstate
       >>> from brainstate.nn import ModuleMapper
       >>> from brainstate.util.filter import OfType
       >>>
       >>> class MyModule(brainstate.nn.Module):
       ...     def init_state(self, size):
       ...         self.weight = brainstate.ParamState(jnp.zeros(size))
       ...     def update(self, x):
       ...         return x @ self.weight.value
       >>>
       >>> module = MyModule()
       >>> vmapper = ModuleMapper(
       ...     module,
       ...     init_map_size=10,
       ...     in_axes=0,
       ...     axis_name="batch"
       ... )
       >>> vmapper.init_all_states(size=(5,))  # Creates 10 copies of the state
       >>> outputs = vmapper.update(inputs)  # inputs.shape = (10, 5)

    **Parallel mapping across devices:**

    .. code-block:: python

       >>> import jax
       >>> pmapper = ModuleMapper(
       ...     module,
       ...     init_map_size=jax.device_count(),
       ...     behavior='pmap',
       ...     axis_name="devices"
       ... )
       >>> pmapper.init_all_states(size=(5,))
       >>> # inputs replicated across devices
       >>> outputs = pmapper.update(inputs)

    **Mapping custom module methods:**

    .. code-block:: python

       >>> vmapper = ModuleMapper(module, init_map_size=10)
       >>> vmapper.init_all_states(size=(5,))
       >>> # Call a specific method with custom mapping
       >>> predictions = vmapper.map('predict', in_axes=0)(inputs)

    See Also
    --------
    brainstate.transform.vmap2 : Vectorized mapping with state semantics.
    brainstate.transform.pmap2 : Parallel mapping across devices.
    brainstate.transform.vmap2_new_states : Initialize vectorized states.
    brainstate.transform.pmap2_new_states : Initialize parallel states.
    Vmap : Simpler vectorization wrapper without explicit state initialization.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        module: 'Module',

        # vmap parameters for init_all_states
        init_map_size: int,
        init_state_axes: Dict[int, Filter] = None,
        state_tag: str = None,

        # vmap parameters for update calls
        in_axes=0,
        out_axes=0,
        axis_name=None,
        spmd_axis_name=None,
        call_state_axes: Dict[int, Filter] = None,

        # type to parallelize
        behavior: str = 'vmap',
    ):
        super().__init__()
        assert isinstance(init_map_size, int), 'init_map_size must be an integer.'
        assert behavior in ['vmap', 'pmap'], 'behavior must be either "vmap" or "pmap".'
        self.init_map_size = init_map_size
        self.module = module
        self.state_tag = state_tag
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.axis_name = axis_name
        self.spmd_axis_name = spmd_axis_name
        self.call_state_axes = call_state_axes
        self.init_state_axes = init_state_axes
        self.dict_vmap_states = None
        self.behavior = behavior

        self._init = False
        self._call_state_axes = None

    def __pretty_repr_item__(self, name, value):
        if name in [
            '_init',
            'dict_vmap_states',
            '_call_state_axes'
        ]:
            return None
        return name, value

    def _integrate_state_axes(self, call_state_axes):
        if call_state_axes is None:
            call_state_axes = dict()
        call_state_axes = dict(call_state_axes)
        for k, v in tuple(call_state_axes.items()):
            if k in self.dict_vmap_states:
                call_state_axes[k] = filter.Any(v, ToPredicate(self.dict_vmap_states[k]))
        for k, v in self.dict_vmap_states.items():
            if k not in call_state_axes:
                call_state_axes[k] = ToPredicate(v)
        return call_state_axes

    def init_all_states(self, **kwargs):
        """Initialize vectorized states for the wrapped module.

        This method must be called before the first ``update`` call. It creates
        and configures vectorized versions of the module's states based on the
        specified axis size.
        """
        if self.behavior == 'vmap':
            map_fn = vmap2_new_states
        elif self.behavior == 'pmap':
            map_fn = pmap2_new_states
        else:
            raise ValueError(
                'Invalid behavior specified. Must be "vmap" or "pmap".'
            )

        self.dict_vmap_states = map_fn(
            self.module,
            kwargs,
            state_tag=self.state_tag,
            axis_size=self.init_map_size,
            state_out_axes=self.init_state_axes,
        )
        self._call_state_axes = self._integrate_state_axes(self.call_state_axes)
        self._init = True

    def update(self, *args, **kwargs):
        """Execute the vectorized module with the given arguments.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the vectorized module.
        **kwargs
            Keyword arguments forwarded to the vectorized module.

        Returns
        -------
        Any
            Result of executing the vectorized module.

        Raises
        ------
        ValueError
            If ``init_all_states`` has not been called before this method.
        """
        if not self._init:
            raise ValueError(
                'ModuleMapper.update called before init_all_states. Please call init_all_states first.'
            )
        if self.behavior == 'vmap':
            map_fn = vmap2
        elif self.behavior == 'pmap':
            map_fn = pmap2
        else:
            raise ValueError(
                'Invalid behavior specified. Must be "vmap" or "pmap".'
            )

        return map_fn(
            self.module,
            in_axes=self.in_axes,
            out_axes=self.out_axes,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
            state_in_axes=self._call_state_axes,
            state_out_axes=self._call_state_axes,
        )(*args, **kwargs)

    def map(
        self,
        fn: str,
        in_axes: Any = 0,
        out_axes: Any = 0,
        axis_name: Optional[str] = None,
        state_axes: Dict[int, Filter] = None,
    ) -> _ModuleMapperCalling:
        """
        Access the wrapped module's methods with vectorized mapping.

        Returns
        -------
        _ModuleMapperCalling
            An object that allows calling methods of the wrapped module
            with vectorized mapping.

        Examples
        --------
        .. code-block:: python

           >>> vmapped = ModuleMapper(module, in_axes=0)
           >>> vmapped.init_all_states(vmap_axis=10)
           >>> outputs = vmapped.vmap('predict')(inputs)
        """
        try:
            fn = getattr(self.module, fn)
        except AttributeError:
            raise AttributeError(f'Module has no method named {fn}.') from None
        if not self._init:
            raise ValueError(
                'ModuleMapper.update called before init_all_states. Please call init_all_states first.'
            )
        return _ModuleMapperCalling(
            fn,
            behavior=self.behavior,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            state_axes=self._integrate_state_axes(state_axes),
        )
