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

from typing import Callable

from ._export import *
from ._export import __all__ as export_all

__all__ = ['ForLoop', 'JIT'] + export_all


class ForLoop:
    def __init__(
        self,
        fn: Callable,
        device: str,
    ):
        self.fn = fn
        self.device = device
        if device not in get_registered_devices():
            raise ValueError(f"Device '{device}' is not registered.")

    def __call__(self, *args, **kwargs):
        return get_forloop_impl(self.device)(self.fn, *args, **kwargs)


class JIT:
    def __init__(
        self,
        fn: Callable,
        device: str,
    ):
        self.fn = fn
        self.device = device

        if device not in get_registered_devices():
            raise ValueError(f"Device '{device}' is not registered.")

    def __call__(self, *args, **kwargs):
        return get_jit_impl(self.device)(self.fn, *args, **kwargs)
