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

from typing import Callable, Optional, Union, Dict

from brainstate.typing import Missing
from ._impl import *

__all__ = [
    'ForLoop',
    'JIT',
]


class ForLoop:
    """Device-aware decorator/adapter for ``for_loop`` transformations.

    Parameters
    ----------
    fn : Callable or Missing, optional
        Function to wrap immediately. When omitted (the default), the instance
        behaves as a decorator and expects a callable later.
    device : str, default='cpu'
        Target device identifier registered via :func:`register_forloop_impl`.
    **kwargs
        Additional arguments forwarded to the device-specific implementation.

    Notes
    -----
    The object is callable. If it was created without ``fn`` it returns a new
    :class:`ForLoop` configured for the provided function when used as a
    decorator.

    Examples
    --------
    Wrap a function directly:

    >>> loop = ForLoop(step_fn, device='gpu')
    >>> loop(xs, length=100)

    Use as a decorator:

    >>> @ForLoop(device='gpu')
    ... def step(x):
    ...     return x * 2
    >>> step(xs)
    """

    def __init__(
        self,
        fn: Union[Callable, Missing] = Missing(),
        device: str = 'cpu',
        **kwargs
    ):
        self._device = device
        self._call_count = 0
        self.kwargs = kwargs

        if device not in get_registered_devices():
            available = get_registered_devices()
            raise ValueError(
                f"Device '{device}' is not registered.\n"
                f"Available devices: {available}\n"
                f"Register a new device using register_forloop_impl()."
            )

        if isinstance(fn, Missing):
            self._fn = None
            self._is_decorator = True
        else:
            if not callable(fn):
                raise TypeError(
                    f"Expected a callable function, got {type(fn).__name__}. "
                    f"If using as decorator, use @ForLoop(device='{device}')."
                )
            self._fn = fn
            self._is_decorator = False

        self._compiled_fn = None

    @property
    def fn(self) -> Optional[Callable]:
        """The wrapped function."""
        return self._fn

    @property
    def device(self) -> str:
        """The target device."""
        return self._device

    @property
    def call_count(self) -> int:
        """Number of times the loop has been executed."""
        return self._call_count

    def __call__(self, *args, **kwargs):
        if self._is_decorator and self._fn is None:
            # Being used as a decorator
            if len(args) != 1 or not callable(args[0]):
                raise TypeError(
                    "When using ForLoop as a decorator, it must be called with a single callable argument."
                )
            fn = args[0]
            return ForLoop(fn=fn, device=self._device, **self.kwargs)

        if self._fn is None:
            raise RuntimeError("ForLoop function not set. This should not happen.")
        if self._compiled_fn is None:
            self._compiled_fn = get_forloop_impl(self._device)(self._fn, **self.kwargs)

        self._call_count += 1
        return self._compiled_fn(*args, **kwargs)

    def __repr__(self) -> str:
        fn_name = self._fn.__name__ if self._fn and hasattr(self._fn, '__name__') else str(self._fn)
        return f"ForLoop(fn={fn_name}, device='{self._device}')"


class JIT:
    """Device-aware decorator/adapter for just-in-time compilation.

    Parameters
    ----------
    fn : Callable or Missing, optional
        Function to compile immediately. When omitted, the instance works as a
        decorator.
    device : str, default='cpu'
        Target device identifier registered via :func:`register_jit_impl`.
    jit_kwargs : dict, optional
        Keyword arguments passed to the registered JIT factory upon compilation.
    **kwargs
        Additional keyword arguments forwarded to the implementation.

    Notes
    -----
    The class compiles the function during instantiation in decorator mode and
    caches the compiled callable for subsequent invocations.

    Examples
    --------
    Direct usage:

    >>> jit_step = JIT(step_fn, device='gpu')
    >>> jit_step(x, y)

    As a decorator:

    >>> @JIT(device='cpu', jit_kwargs={'static_argnums': (1,)})
    ... def power(x, n):
    ...     return x ** n
    >>> power(2, 3)
    """

    def __init__(
        self,
        fn: Union[Callable, Missing] = Missing(),
        device: str = 'cpu',
        jit_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        self._device = device
        self._jit_kwargs = jit_kwargs
        self._call_count = 0
        self._compiled_fn = None
        self.kwargs = kwargs

        if device not in get_registered_devices():
            available = get_registered_devices()
            raise ValueError(
                f"Device '{device}' is not registered.\n"
                f"Available devices: {available}\n"
                f"Register a new device using register_jit_impl()."
            )

        if isinstance(fn, Missing):
            self._fn = None
            self._is_decorator = True
        else:
            if not callable(fn):
                raise TypeError(
                    f"Expected a callable function, got {type(fn).__name__}. "
                    f"If using as decorator, use @JIT(device='{device}')."
                )
            self._fn = fn
            self._is_decorator = False
            self._compile_function()

    def _compile_function(self) -> None:
        """Compile the function with the device-specific JIT implementation."""
        if self._fn is not None:
            jit_impl = get_jit_impl(self._device)
            if self._jit_kwargs:
                self._compiled_fn = jit_impl(self._fn, **self._jit_kwargs, **self.kwargs)
            else:
                self._compiled_fn = jit_impl(self._fn, **self.kwargs)

    @property
    def fn(self) -> Optional[Callable]:
        """The original (uncompiled) function."""
        return self._fn

    @property
    def device(self) -> str:
        """The target device."""
        return self._device

    @property
    def call_count(self) -> int:
        """Number of times the compiled function has been called."""
        return self._call_count

    @property
    def compiled_fn(self) -> Optional[Callable]:
        """The compiled function."""
        return self._compiled_fn

    def __call__(self, *args, **kwargs):
        if self._is_decorator and self._fn is None:
            # Being used as a decorator
            if len(args) != 1 or not callable(args[0]):
                raise TypeError(
                    "When using JIT as a decorator, it must be called with a single callable argument."
                )
            fn = args[0]
            return JIT(
                fn=fn,
                device=self._device,
                jit_kwargs=self._jit_kwargs,
                **self.kwargs
            )

        if self._compiled_fn is None:
            raise RuntimeError("JIT function not compiled. This should not happen.")

        self._call_count += 1
        return self._compiled_fn(*args, **kwargs)

    def __repr__(self) -> str:
        fn_name = self._fn.__name__ if self._fn and hasattr(self._fn, '__name__') else str(self._fn)
        kwargs_str = ', '.join(f'{k}={v}' for k, v in self._jit_kwargs.items())
        if kwargs_str:
            return f"JIT(fn={fn_name}, device='{self._device}', {kwargs_str})"
        return f"JIT(fn={fn_name}, device='{self._device}')"
