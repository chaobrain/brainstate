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

from typing import Any, Callable, Optional, Sequence, Union

import jax
from jax.experimental import io_callback as _jax_io_callback

from brainstate._state import State
from brainstate._utils import set_module_as

__all__ = ['pure_callback', 'io_callback']


def _as_state_tuple(states) -> tuple:
    """Normalize None / a single State / a sequence of States to a tuple of States."""
    if states is None:
        return ()
    if isinstance(states, State):
        return (states,)
    states = tuple(states)
    if any(not isinstance(s, State) for s in states):
        raise TypeError("read_states/write_states must be State instances.")
    return states


@set_module_as("brainstate.transform")
def pure_callback(
    callback: Callable,
    result_shape_dtypes: Any,
    *args,
    read_states: Optional[Union[State, Sequence[State]]] = None,
    sharding=None,
    vmap_method: Optional[str] = None,
):
    """Call a pure host-Python function from inside transformed code.

    A state-aware wrapper over :func:`jax.pure_callback`. The values of any
    ``read_states`` are read at call time and appended to ``callback``'s
    positional arguments, so host code can use current ``State`` values. The
    callback must be pure (no side effects, same output for same input).

    Parameters
    ----------
    callback : callable
        Host-side function invoked as ``callback(*args, *read_state_values)``.
        Must return arrays matching ``result_shape_dtypes``.
    result_shape_dtypes : PyTree of jax.ShapeDtypeStruct
        The shape/dtype structure of ``callback``'s return value.
    *args
        Positional arguments (traced arrays) passed to ``callback``.
    read_states : State or sequence of State, optional
        States whose current ``.value`` is appended (in order) to ``callback``'s
        arguments.
    sharding : jax.sharding.Sharding, optional
        Optional output sharding, forwarded to :func:`jax.pure_callback`.
    vmap_method : str, optional
        How the callback behaves under :func:`vmap`, forwarded to
        :func:`jax.pure_callback`.

    Returns
    -------
    PyTree
        The callback's result, matching ``result_shape_dtypes``.

    See Also
    --------
    io_callback

    Notes
    -----
    Because the callback must be pure, the compiler may deduplicate or drop it;
    use :func:`io_callback` for side effects or for writing back into states.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> import numpy as np
        >>> w = brainstate.State(jnp.array([2.0, 3.0]))
        >>> def host(x, w_val):
        ...     return np.asarray(x) * np.asarray(w_val)
        >>> x = jnp.array([5.0, 7.0])
        >>> spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        >>> brainstate.transform.pure_callback(host, spec, x, read_states=w)
        Array([10., 21.], dtype=float32)
    """
    states = _as_state_tuple(read_states)
    if states:
        state_vals = tuple(s.value for s in states)

        def _wrapped(cb_args, cb_state_vals):
            return callback(*cb_args, *cb_state_vals)

        return jax.pure_callback(
            _wrapped, result_shape_dtypes, args, state_vals,
            sharding=sharding, vmap_method=vmap_method,
        )
    return jax.pure_callback(
        callback, result_shape_dtypes, *args,
        sharding=sharding, vmap_method=vmap_method,
    )


@set_module_as("brainstate.transform")
def io_callback(
    callback: Callable,
    result_shape_dtypes: Any,
    *args,
    read_states: Optional[Union[State, Sequence[State]]] = None,
    write_states: Optional[Union[State, Sequence[State]]] = None,
    sharding=None,
    ordered: bool = True,
):
    """Call a side-effecting host-Python function from inside transformed code.

    A state-aware wrapper over :func:`jax.experimental.io_callback`. The values
    of ``read_states`` are appended to ``callback``'s arguments, and the
    callback's result is optionally written back into ``write_states``. Useful
    for external (non-JAX) solvers, host-side data, and in-loop logging.

    Parameters
    ----------
    callback : callable
        Host-side function invoked as ``callback(*args, *read_state_values)``.
        Must return arrays matching ``result_shape_dtypes``.
    result_shape_dtypes : PyTree of jax.ShapeDtypeStruct
        The shape/dtype structure of ``callback``'s return value. When
        ``write_states`` is a sequence, this must be a matching sequence.
    *args
        Positional arguments (traced arrays) passed to ``callback``.
    read_states : State or sequence of State, optional
        States whose current ``.value`` is appended (in order) to ``callback``'s
        arguments.
    write_states : State or sequence of State, optional
        States to overwrite with the callback's result. A single ``State`` is
        assigned the whole result; a sequence is assigned element-by-element
        from a result sequence of the same length.
    sharding : jax.sharding.Sharding, optional
        Optional output sharding, forwarded to the JAX callback.
    ordered : bool, default True
        If ``True``, the callbacks are executed in program order (recommended
        when the callback has side effects or writes state).

    Returns
    -------
    PyTree
        The callback's result, matching ``result_shape_dtypes``.

    See Also
    --------
    pure_callback

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax, jax.numpy as jnp
        >>> import numpy as np
        >>> state = brainstate.State(jnp.array([0.0, 0.0]))
        >>> def host(x):
        ...     return np.asarray(x) + 1.0
        >>> x = jnp.array([1.0, 2.0])
        >>> spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        >>> out = brainstate.transform.io_callback(host, spec, x, write_states=state)
        >>> state.value
        Array([2., 3.], dtype=float32)
    """
    in_states = _as_state_tuple(read_states)
    if in_states:
        in_vals = tuple(s.value for s in in_states)

        def _wrapped(cb_args, cb_state_vals):
            return callback(*cb_args, *cb_state_vals)

        result = _jax_io_callback(
            _wrapped, result_shape_dtypes, args, in_vals,
            sharding=sharding, ordered=ordered,
        )
    else:
        result = _jax_io_callback(
            callback, result_shape_dtypes, *args,
            sharding=sharding, ordered=ordered,
        )

    # Optional write-back into states.
    if write_states is not None:
        if isinstance(write_states, State):
            write_states.value = result
        else:
            for st, val in zip(tuple(write_states), result):
                if not isinstance(st, State):
                    raise TypeError("write_states must be State instances.")
                st.value = val

    return result
