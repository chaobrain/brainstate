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


from typing import Callable

from brainstate.transform._jit import jit
from brainstate.transform._loop_collect_return import for_loop

__all__ = [
    "get_registered_devices",
    'get_forloop_impl',
    'get_jit_impl',
    'register_forloop_impl',
    'register_jit_impl',
]

registered_devices = {}


def get_registered_devices():
    return list(registered_devices.keys())


def get_forloop_impl(device: str) -> Callable:
    if device not in registered_devices:
        raise ValueError(f"Device '{device}' is not registered.")
    return registered_devices[device]['forloop']


def get_jit_impl(device: str) -> Callable:
    if device not in registered_devices:
        raise ValueError(f"Device '{device}' is not registered.")
    return registered_devices[device]['jit']


def register_forloop_impl(device: str, impl: Callable):
    if device not in registered_devices:
        registered_devices[device] = {}
    registered_devices[device]['forloop'] = impl


def register_jit_impl(device: str, impl: Callable):
    if device not in registered_devices:
        registered_devices[device] = {}
    registered_devices[device]['jit'] = impl


def _for_loop_wrapper(fn: Callable, **kwargs) -> Callable:
    def run(*args, **kwargs):
        return for_loop(fn, *args, **kwargs)

    return run


def _jit_wrapper(fn: Callable, **jit_kwargs) -> Callable:
    return jit(fn, **jit_kwargs)


for d in ['cpu', 'gpu', 'tpu']:
    register_forloop_impl(d, _for_loop_wrapper)
    register_jit_impl(d, _jit_wrapper)
