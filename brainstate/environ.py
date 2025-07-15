# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import contextlib
import dataclasses
import functools
import os
import re
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable

import numpy as np
from jax import config, devices, numpy as jnp
from jax.typing import DTypeLike

from .mixin import Mode

__all__ = [
    # functions for environment settings
    'set', 'context', 'get', 'all', 'set_host_device_count', 'set_platform',
    # functions for getting default behaviors
    'get_host_device_count', 'get_platform', 'get_dt', 'get_mode', 'get_precision',
    # functions for default data types
    'dftype', 'ditype', 'dutype', 'dctype',
    # others
    'tolerance', 'register_default_behavior',
]

# Default, there are several shared arguments in the global context.
I = 'i'  # the index of the current computation.
T = 't'  # the current time of the current computation.
JIT_ERROR_CHECK = 'jit_error_check'  # whether to record the current computation.
FIT = 'fit'  # whether to fit the model.


@dataclasses.dataclass
class DefaultContext(threading.local):
    # default environment settings
    settings: Dict[Hashable, Any] = dataclasses.field(default_factory=dict)
    # current environment settings
    contexts: defaultdict[Hashable, Any] = dataclasses.field(default_factory=lambda: defaultdict(list))
    # environment functions
    functions: Dict[Hashable, Any] = dataclasses.field(default_factory=dict)


DFAULT = DefaultContext()
_NOT_PROVIDE = object()


@contextlib.contextmanager
def context(**kwargs):
    r"""
    Context-manager that sets a computing environment for brain dynamics computation.

    In BrainPy, there are several basic computation settings when constructing models,
    including ``mode`` for controlling model computing behavior, ``dt`` for numerical
    integration, ``int_`` for integer precision, and ``float_`` for floating precision.
    :py:class:`~.environment`` provides a context for model construction and
    computation. In this temporal environment, models are constructed with the given
    ``mode``, ``dt``, ``int_``, etc., environment settings.

    For instance::

    >>> import brainstate as brainstate
    >>> with brainstate.environ.context(dt=0.1) as env:
    ...     dt = brainstate.environ.get('dt')
    ...     print(env)

    """
    if 'platform' in kwargs:
        raise ValueError('\n'
                         'Cannot set platform in "context" environment. \n'
                         'You should set platform in the global environment by "set_platform()" or "set()".')
    if 'host_device_count' in kwargs:
        raise ValueError('Cannot set host_device_count in environment context. '
                         'Please use set_host_device_count() or set() for the global setting.')

    if 'precision' in kwargs:
        last_precision = _get_precision()
        _set_jax_precision(kwargs['precision'])

    try:
        for k, v in kwargs.items():

            # update the current environment
            DFAULT.contexts[k].append(v)

            # restore the environment functions
            if k in DFAULT.functions:
                DFAULT.functions[k](v)

        # yield the current all environment information
        yield all()
    finally:

        for k, v in kwargs.items():

            # restore the current environment
            DFAULT.contexts[k].pop()

            # restore the environment functions
            if k in DFAULT.functions:
                DFAULT.functions[k](get(k))

        if 'precision' in kwargs:
            _set_jax_precision(last_precision)


def get(key: str, default: Any = _NOT_PROVIDE, desc: str = None):
    """
    Get one of the default computation environment.

    Returns
    -------
    item: Any
      The default computation environment.
    """
    if key == 'platform':
        return get_platform()

    if key == 'host_device_count':
        return get_host_device_count()

    if key in DFAULT.contexts:
        if len(DFAULT.contexts[key]) > 0:
            return DFAULT.contexts[key][-1]
    if key in DFAULT.settings:
        return DFAULT.settings[key]

    if default is _NOT_PROVIDE:
        if desc is not None:
            raise KeyError(
                f"'{key}' is not found in the context. \n"
                f"You can set it by `brainstate.share.context({key}=value)` "
                f"locally or `brainstate.share.set({key}=value)` globally. \n"
                f"Description: {desc}"
            )
        else:
            raise KeyError(
                f"'{key}' is not found in the context. \n"
                f"You can set it by `brainstate.share.context({key}=value)` "
                f"locally or `brainstate.share.set({key}=value)` globally."
            )
    return default


def all() -> dict:
    """
    Get all the current default computation environment.

    Returns
    -------
    r: dict
      The current default computation environment.
    """
    r = dict()
    for k, v in DFAULT.contexts.items():
        if v:
            r[k] = v[-1]
    for k, v in DFAULT.settings.items():
        if k not in r:
            r[k] = v
    return r


def get_dt():
    """Get the numerical integrator precision.

    Returns
    -------
    dt : float
        Numerical integration precision.
    """
    return get('dt')


def get_mode() -> Mode:
    """Get the default computing mode.

    References
    ----------
    mode: Mode
      The default computing mode.
    """
    return get('mode')


def get_platform() -> str:
    """Get the computing platform.

    Returns
    -------
    platform: str
      Either 'cpu', 'gpu' or 'tpu'.
    """
    return devices()[0].platform


def get_host_device_count():
    """
    Get the number of host devices.

    Returns
    -------
    n: int
      The number of host devices.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    match = re.search(r"--xla_force_host_platform_device_count=(\d+)", xla_flags)
    return int(match.group(1)) if match else 1


def _get_precision() -> int | str:
    """
    Get the default precision.

    Returns
    -------
    precision: int
      The default precision.
    """
    return get('precision')


def get_precision() -> int:
    """
    Get the default precision.

    Returns
    -------
    precision: int
      The default precision.
    """
    precision = get('precision')
    if precision == 'bf16':
        return 16
    if isinstance(precision, int):
        return precision
    if isinstance(precision, str):
        return int(precision)
    raise ValueError(f'Unsupported precision: {precision}')


def set(
    platform: str = None,
    host_device_count: int = None,
    precision: int | str = None,
    mode: Mode = None,
    **kwargs
):
    """
    Set the global default computation environment.



    Args:
      platform: str. The computing platform. Either 'cpu', 'gpu' or 'tpu'.
      host_device_count: int. The number of host devices.
      precision: int, str. The default precision.
      mode: Mode. The computing mode.
      **kwargs: dict. Other environment settings.
    """
    if platform is not None:
        set_platform(platform)
    if host_device_count is not None:
        set_host_device_count(host_device_count)
    if precision is not None:
        _set_jax_precision(precision)
        kwargs['precision'] = precision
    if mode is not None:
        assert isinstance(mode, Mode), 'mode must be a Mode instance.'
        kwargs['mode'] = mode

    # set default environment
    DFAULT.settings.update(kwargs)

    # update the environment functions
    for k, v in kwargs.items():
        if k in DFAULT.functions:
            DFAULT.functions[k](v)


def set_host_device_count(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.

    :param int n: number of devices to use.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(n)] + xla_flags)

    # update the environment functions
    if 'host_device_count' in DFAULT.functions:
        DFAULT.functions['host_device_count'](n)


def set_platform(platform: str):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    Args:
      platform: str. The computing platform. Either 'cpu', 'gpu' or 'tpu'.

    Raises:
      ValueError: If the platform is not in ['cpu', 'gpu', 'tpu'].
    """
    assert platform in ['cpu', 'gpu', 'tpu']
    config.update("jax_platform_name", platform)

    # update the environment functions
    if 'platform' in DFAULT.functions:
        DFAULT.functions['platform'](platform)


def _set_jax_precision(precision: int | str):
    """
    Set the default precision.

    Args:
      precision: int. The default precision.
    """
    # assert precision in [64, 32, 16, 'bf16', 8], f'Precision must be in [64, 32, 16, "bf16", 8]. But got {precision}.'
    if precision in [64, '64']:
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)


@functools.lru_cache()
def _get_uint(precision: int):
    if precision in [64, '64']:
        return np.uint64
    elif precision in [32, '32']:
        return np.uint32
    elif precision in [16, '16', 'bf16']:
        return np.uint16
    elif precision in [8, '8']:
        return np.uint8
    else:
        raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_int(precision: int):
    if precision in [64, '64']:
        return np.int64
    elif precision in [32, '32']:
        return np.int32
    elif precision in [16, '16', 'bf16']:
        return np.int16
    elif precision in [8, '8']:
        return np.int8
    else:
        raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_float(precision: int):
    if precision in [64, '64']:
        return np.float64
    elif precision in [32, '32']:
        return np.float32
    elif precision in [16, '16']:
        return np.float16
    elif precision in ['bf16']:
        return jnp.bfloat16
    elif precision in [8, '8']:
        return jnp.float8_e5m2
    else:
        raise ValueError(f'Unsupported precision: {precision}')


@functools.lru_cache()
def _get_complex(precision: int):
    if precision == [64, '64']:
        return np.complex128
    elif precision == [32, '32']:
        return np.complex64
    elif precision in [16, '16', 'bf16']:
        return np.complex64
    elif precision == [8, '8']:
        return np.complex64
    else:
        raise ValueError(f'Unsupported precision: {precision}')


def dftype() -> DTypeLike:
    """
    Default floating data type.

    This function returns the default floating data type based on the current precision.
    If you want the data dtype is changed with the setting of the precision by ``brainstate.environ.set(precision)``,
    you can use this function to get the default floating data type, and create the data by using ``dtype=dftype()``.

    For example, if the precision is set to 32, the default floating data type is ``np.float32``.

    >>> import brainstate as brainstate
    >>> import numpy as np
    >>> with brainstate.environ.context(precision=32):
    ...    a = np.zeros(1, dtype=brainstate.environ.dftype())
    >>> print(a.dtype)

    Returns
    -------
    float_dtype: DTypeLike
      The default floating data type.
    """
    return _get_float(_get_precision())


def ditype() -> DTypeLike:
    """
    Default integer data type.

    This function returns the default integer data type based on the current precision.
    If you want the data dtype is changed with the setting of the precision by ``brainstate.environ.set(precision)``,
    you can use this function to get the default integer data type, and create the data by using ``dtype=ditype()``.

    For example, if the precision is set to 32, the default integer data type is ``np.int32``.

    >>> import brainstate as brainstate
    >>> import numpy as np
    >>> with brainstate.environ.context(precision=32):
    ...    a = np.zeros(1, dtype=brainstate.environ.ditype())
    >>> print(a.dtype)
    int32

    Returns
    -------
    int_dtype: DTypeLike
      The default integer data type.
    """
    return _get_int(_get_precision())


def dutype() -> DTypeLike:
    """
    Default unsigned integer data type.

    This function returns the default unsigned integer data type based on the current precision.
    If you want the data dtype is changed with the setting of the precision
    by ``brainstate.environ.set(precision)``, you can use this function to get the default
    unsigned integer data type, and create the data by using ``dtype=dutype()``.

    For example, if the precision is set to 32, the default unsigned integer data type is ``np.uint32``.

    >>> import brainstate as brainstate
    >>> import numpy as np
    >>> with brainstate.environ.context(precision=32):
    ...   a = np.zeros(1, dtype=brainstate.environ.dutype())
    >>> print(a.dtype)
    uint32

    Returns
    -------
    uint_dtype: DTypeLike
      The default unsigned integer data type.
    """
    return _get_uint(_get_precision())


def dctype() -> DTypeLike:
    """
    Default complex data type.

    This function returns the default complex data type based on the current precision.
    If you want the data dtype is changed with the setting of the precision by ``brainstate.environ.set(precision)``,
    you can use this function to get the default complex data type, and create the data by using ``dtype=dctype()``.

    For example, if the precision is set to 32, the default complex data type is ``np.complex64``.

    >>> import brainstate as brainstate
    >>> import numpy as np
    >>> with brainstate.environ.context(precision=32):
    ...    a = np.zeros(1, dtype=brainstate.environ.dctype())
    >>> print(a.dtype)
    complex64

    Returns
    -------
    complex_dtype: DTypeLike
      The default complex data type.
    """
    return _get_complex(_get_precision())


def tolerance():
    if get_precision() == 64:
        return jnp.array(1e-12, dtype=np.float64)
    elif get_precision() == 32:
        return jnp.array(1e-5, dtype=np.float32)
    else:
        return jnp.array(1e-2, dtype=np.float16)


def register_default_behavior(key: str, behavior: Callable, replace_if_exist: bool = False):
    """
    Register a default behavior for a specific global key parameter.

    For example, you can register a default behavior for the key 'dt' by::

    >>> import brainstate as brainstate
    >>> def dt_behavior(dt):
    ...     print(f'Set the default dt to {dt}.')
    ...
    >>> brainstate.environ.register_default_behavior('dt', dt_behavior)

    Then, when you set the default dt by `brainstate.environ.set(dt=0.1)`, the behavior
    `dt_behavior` will be called with
    `dt_behavior(0.1)`.

    >>> brainstate.environ.set(dt=0.1)
    Set the default dt to 0.1.
    >>> with brainstate.environ.context(dt=0.2):
    ...     pass
    Set the default dt to 0.2.
    Set the default dt to 0.1.


    Args:
      key: str. The key to register.
      behavior: Callable. The behavior to register. It should be a callable.
      replace_if_exist: bool. Whether to replace the behavior if the key has been registered.

    """
    assert isinstance(key, str), 'key must be a string.'
    assert callable(behavior), 'behavior must be a callable.'
    if not replace_if_exist:
        assert key not in DFAULT.functions, f'{key} has been registered.'
    DFAULT.functions[key] = behavior


set(precision=32)
