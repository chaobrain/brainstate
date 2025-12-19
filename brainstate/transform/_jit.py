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

import functools
from collections.abc import Iterable, Sequence
from typing import (Any, Callable, Union)

import jax
from jax._src import sharding_impls
from jax.stages import Traced

from brainstate._compatible_import import Device
from brainstate._utils import set_module_as
from brainstate.typing import Missing
from ._make_jaxpr import StatefulFunction, _ensure_index_tuple

__all__ = ['jit']


class JittedFunction(Callable):
    """
    A wrapped version of ``fun``, set up for just-in-time compilation.
    """
    __module__ = 'brainstate.transform'

    origin_fun: Callable  # the original function
    stateful_fun: StatefulFunction  # the stateful function for extracting states
    jitted_fun: jax.stages.Wrapped  # the jitted function
    clear_cache: Callable  # clear the cache of the jitted function
    eval_shape: Callable  # evaluate the shape of the jitted function
    compile: Callable  # lower the jitted function
    trace: Callable  # trace the jitted
    lower: Callable  # lower the jitted

    def __call__(self, *args, **kwargs):
        pass


def _get_jitted_fun(
    fun: Callable,
    name: str,
    in_shardings,
    out_shardings,
    static_argnums,
    donate_argnums,
    static_argnames,
    donate_argnames,
    keep_unused,
    device,
    backend,
    inline,
    **kwargs
) -> JittedFunction:
    static_argnums = tuple() if static_argnums is None else _ensure_index_tuple(static_argnums)
    donate_argnums = tuple() if donate_argnums is None else _ensure_index_tuple(donate_argnums)
    fun = StatefulFunction(
        fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        name='jit',
        return_only_write=True
    )

    def run_fn(*args, **kwargs):
        return fun.jaxpr_call(*args, **kwargs)

    if name:
        run_fn.__name__ = name

    jit_fun = jax.jit(
        run_fn if name else fun.jaxpr_call,
        static_argnums=tuple(i + 1 for i in static_argnums),
        static_argnames=static_argnames,
        donate_argnums=tuple(i + 1 for i in donate_argnums),
        donate_argnames=donate_argnames,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        **kwargs
    )

    @functools.wraps(fun.fun)
    def jitted_fn(*args, **params):
        if jax.config.jax_disable_jit:
            return fun.fun(*args, **params)

        # compile the function and get the state trace
        state_trace = fun.get_state_trace(*args, **params, compile_if_miss=True)
        read_state_vals = state_trace.get_read_state_values(True)

        # call the jitted function
        write_state_vals, outs = jit_fun(state_trace.get_state_values(), *args, **params)

        # write the state values back to the states
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
        return outs

    def clear_cache():
        """
        Clear the cache of the jitted function.
        """
        # clear the cache of the stateful function
        fun.clear_cache()
        try:
            # clear the cache of the jitted function
            jit_fun.clear_cache()
        except AttributeError:
            pass

    def eval_shape(*args, **params):
        state_trace = fun.get_state_trace(*args, **params, compile_if_miss=True)
        read_state_vals = state_trace.get_read_state_values(True)
        write_state_vals = state_trace.get_write_state_values(True)
        ret = jit_fun.eval_shape(state_trace.get_state_values(), *args, **params)
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
        return ret

    def lower(*args, **params) -> jax.stages.Lowered:
        """
        Lower this function explicitly for the given arguments.

        A lowered function is staged out of Python and translated to a
        compiler's input language, possibly in a backend-dependent
        manner. It is ready for compilation but not yet compiled.

        Returns:
          A ``Lowered`` instance representing the lowering.
        """
        # compile the function and get the state trace
        state_trace = fun.get_state_trace(*args, **params, compile_if_miss=True)
        read_state_vals = state_trace.get_read_state_values(True)
        write_state_vals = state_trace.get_write_state_values(True)

        # compile the model
        lowered = jit_fun.lower(state_trace.get_state_values(), *args, **params)

        # write the state values back to the states
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
        return lowered

    def trace(*args, **params) -> Traced:
        """
        Trace this function explicitly for the given arguments.

        A traced function is staged out of Python and translated to a jaxpr. It is
        ready for lowering but not yet lowered.

        Returns:
          A ``Traced`` instance representing the tracing.
        """
        # compile the function and get the state trace
        state_trace = fun.get_state_trace(*args, **params, compile_if_miss=True)
        read_state_vals = state_trace.get_read_state_values(True)
        write_state_vals = state_trace.get_write_state_values(True)

        # call the jitted function
        traced = jit_fun.trace(state_trace.get_state_values(), *args, **params)

        # write the state values back to the states
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
        return traced

    def compile(*args, **params):
        """Lower this function explicitly for the given arguments.

        A lowered function is staged out of Python and translated to a
        compiler's input language, possibly in a backend-dependent
        manner. It is ready for compilation but not yet compiled.

        Returns:
          A ``Lowered`` instance representing the lowering.
        """
        # compile the function and get the state trace
        state_trace = fun.get_state_trace(*args, **params, compile_if_miss=True)
        read_state_vals = state_trace.get_read_state_values(replace_writen=True)
        write_state_vals = state_trace.get_write_state_values(replace_read=True)

        # compile the model
        ret = jit_fun.lower(state_trace.get_state_values(), *args, **params).compile()

        # write the state values back to the states
        state_trace.assign_state_vals_v2(read_state_vals, write_state_vals)
        return ret

    jitted_fn: JittedFunction

    # the original function
    jitted_fn.origin_fun = fun.fun

    # the stateful function for extracting states
    jitted_fn.stateful_fun = fun

    # the jitted function
    jitted_fn.jitted_fun = jit_fun

    # clear cache
    jitted_fn.clear_cache = clear_cache

    # compile the jitted function
    jitted_fn.eval_shape = eval_shape
    jitted_fn.compile = compile
    jitted_fn.lower = lower
    jitted_fn.trace = trace

    return jitted_fn


@set_module_as('brainstate.transform')
def jit(
    fun: Callable | Missing = Missing(),
    in_shardings=sharding_impls.UNSPECIFIED,
    out_shardings=sharding_impls.UNSPECIFIED,
    static_argnums: int | Sequence[int] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    name: str = None,
    **kwargs
) -> Union[JittedFunction, Callable[[Callable], JittedFunction]]:
    """
    Sets up ``fun`` for just-in-time compilation with XLA.

    Parameters
    ----------
    fun : callable or Missing, optional
        Function to be jitted.
    in_shardings : pytree, optional
        Pytree of structure matching that of arguments to ``fun``,
        with all actual arguments replaced by resource assignment specifications.
        It is also valid to specify a pytree prefix (e.g. one value in place of a
        whole subtree), in which case the leaves get broadcast to all values in
        that subtree.

        The ``in_shardings`` argument is optional. JAX will infer the shardings
        from the input :py:class:`jax.Array`'s and defaults to replicating the input
        if the sharding cannot be inferred.

        The valid resource assignment specifications are:

        - :py:class:`XLACompatibleSharding`, which will decide how the value
          will be partitioned. With this, using a mesh context manager is not
          required.
        - :py:obj:`None`, will give JAX the freedom to choose whatever sharding
          it wants.
          For in_shardings, JAX will mark is as replicated but this behavior
          can change in the future.
          For out_shardings, we will rely on the XLA GSPMD partitioner to
          determine the output shardings.

        The size of every dimension has to be a multiple of the total number of
        resources assigned to it. This is similar to pjit's in_shardings.
    out_shardings : pytree, optional
        Like ``in_shardings``, but specifies resource
        assignment for function outputs. This is similar to pjit's
        out_shardings.

        The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
        will use GSPMD's sharding propagation to figure out what the sharding of the
        output(s) should be.
    static_argnums : int or sequence of int, optional
        An optional int or collection of ints that specify which
        positional arguments to treat as static (compile-time constant).
        Operations that only depend on static arguments will be constant-folded in
        Python (during tracing), and so the corresponding argument values can be
        any Python object.

        Static arguments should be hashable, meaning both ``__hash__`` and
        ``__eq__`` are implemented, and immutable. Calling the jitted function
        with different values for these constants will trigger recompilation.
        Arguments that are not arrays or containers thereof must be marked as
        static.

        If neither ``static_argnums`` nor ``static_argnames`` is provided, no
        arguments are treated as static. If ``static_argnums`` is not provided but
        ``static_argnames`` is, or vice versa, JAX uses
        :code:`inspect.signature(fun)` to find any positional arguments that
        correspond to ``static_argnames``
        (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
        provided, ``inspect.signature`` is not used, and only actual
        parameters listed in either ``static_argnums`` or ``static_argnames`` will
        be treated as static.
    donate_argnums : int or sequence of int, optional
        Specify which positional argument buffers are "donated" to
        the computation. It is safe to donate argument buffers if you no longer
        need them once the computation has finished. In some cases XLA can make
        use of donated buffers to reduce the amount of memory needed to perform a
        computation, for example recycling one of your input buffers to store a
        result. You should not reuse buffers that you donate to a computation, JAX
        will raise an error if you try to. By default, no argument buffers are
        donated.

        If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
        arguments are donated. If ``donate_argnums`` is not provided but
        ``donate_argnames`` is, or vice versa, JAX uses
        :code:`inspect.signature(fun)` to find any positional arguments that
        correspond to ``donate_argnames``
        (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
        provided, ``inspect.signature`` is not used, and only actual
        parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
        be donated.

        For more details on buffer donation see the
        `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    static_argnames : str or sequence of str, optional
        An optional string or collection of strings specifying
        which named arguments are treated as static (compile-time constant).
        Operations that only depend on static arguments will be constant-folded in
        Python (during tracing), and so the corresponding argument values can be
        any Python object.
    donate_argnames : str or iterable of str, optional
        An optional string or collection of strings specifying
        which named arguments are donated to the computation. See the
        comment on ``donate_argnums`` for details. If not
        provided but ``donate_argnums`` is set, the default is based on calling
        ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused : bool, default False
        If `False` (the default), arguments that JAX determines to be
        unused by `fun` *may* be dropped from resulting compiled XLA executables.
        Such arguments will not be transferred to the device nor provided to the
        underlying executable. If `True`, unused arguments will not be pruned.
    device : Device, optional
        This is an experimental feature and the API is likely to change.
        Optional, the Device the jitted function will run on. (Available devices
        can be retrieved via :py:func:`jax.devices`.) The default is inherited
        from XLA's DeviceAssignment logic and is usually to use
        ``jax.devices()[0]``.
    backend : str, optional
        This is an experimental feature and the API is likely to change.
        Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
        ``'tpu'``.
    inline : bool, default False
        Specify whether this function should be inlined into enclosing
        jaxprs (rather than being represented as an application of the xla_call
        primitive with its own subjaxpr). Default False.
    **kwargs
        Additional keyword arguments passed to the underlying JAX jit function.

    Returns
    -------
    JittedFunction or callable
        A wrapped version of ``fun``, set up for just-in-time compilation.
        The returned object is a :py:class:`JittedFunction` that can be called with the same arguments
        and has the following attributes and methods:

        - ``stateful_fun`` : the stateful function for extracting states, an instance of :py:class:`StatefulFunction`.
        - ``origin_fun(*args, **kwargs)`` : the original function
        - ``jitted_fun(*args, **kwargs)`` : the jitted function
        - ``clear_cache(*args, **kwargs)`` : clear the cache of the jitted function

    Examples
    --------
    Basic usage with a simple function:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> @brainstate.transform.jit
        ... def f(x):
        ...     return x ** 2
        >>>
        >>> result = f(jnp.array([1, 2, 3]))

    Using static arguments:

    .. code-block:: python

        >>> @brainstate.transform.jit(static_argnums=(1,))
        ... def g(x, n):
        ...     return x ** n
        >>>
        >>> result = g(jnp.array([1, 2, 3]), 2)

    Manual jitting:

    .. code-block:: python

        >>> def h(x):
        ...     return x * 2
        >>>
        >>> jitted_h = brainstate.transform.jit(h)
        >>> result = jitted_h(jnp.array([1, 2, 3]))
    """

    if isinstance(fun, Missing):
        def wrapper(fun_again: Callable) -> JittedFunction:
            return _get_jitted_fun(
                fun_again,
                name=name,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                static_argnums=static_argnums,
                donate_argnums=donate_argnums,
                static_argnames=static_argnames,
                donate_argnames=donate_argnames,
                keep_unused=keep_unused,
                device=device,
                backend=backend,
                inline=inline,
                **kwargs
            )

        return wrapper

    else:
        return _get_jitted_fun(
            fun,
            name=name,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            donate_argnums=donate_argnums,
            static_argnames=static_argnames,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            **kwargs
        )
