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

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional

import jax

from brainstate._utils import set_module_as

__all__ = ['named_call']


@set_module_as("brainstate.transform")
def named_call(fun: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Annotate a function's computation with a name for traces and profiles.

    Wrap ``fun`` so that its body runs inside :func:`jax.named_scope`, attaching
    ``name`` to the resulting equations' name stack. Unlike
    :func:`named_scope`, this does **not** apply ``jit`` — it adds naming
    only, so it composes inside ``grad``/``scan``/``vmap``/``jit`` and leaves
    state read/write behavior unchanged.

    Can be used as a bare decorator (``@named_call``, name taken from the
    function), a parameterized decorator (``@named_call(name='block')``), or a
    direct wrapper (``named_call(fun, name='block')``).

    Parameters
    ----------
    fun : callable, optional
        The function to wrap. If omitted (``None``), a decorator is returned.
    name : str, optional
        The scope name. Defaults to ``fun.__name__`` when not given.

    Returns
    -------
    callable
        The name-annotated function, or a decorator when ``fun`` is ``None``.

    See Also
    --------
    named_scope, jit

    Notes
    -----
    The name is not shown in the default ``repr`` of a jaxpr; it appears in each
    equation's name stack (``eqn.source_info.name_stack``) and in profiler/HLO
    metadata, which is where it aids debugging and performance analysis.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> @brainstate.transform.named_call(name='my_block')
        ... def block(x):
        ...     return jnp.sin(x) * 2.0
        >>>
        >>> block(jnp.array([1.0, 2.0]))
        Array([1.6829419, 1.8185949], dtype=float32)
    """
    def _wrap(f: Callable) -> Callable:
        scope_name = name if name is not None else getattr(f, '__name__', 'named_call')

        @wraps(f)
        def wrapped(*args, **kwargs):
            with jax.named_scope(scope_name):
                return f(*args, **kwargs)

        return wrapped

    if fun is None:
        return _wrap
    return _wrap(fun)
