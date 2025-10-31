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

__all__ = [
    'ForLoop',
    'JIT',
]

registered_devices = {
    'cpu': None,
    'gpu': None,
    'tpu': None,
}


class ForLoop:
    def __init__(
        self,
        fn: Callable,
        device: str,
    ):
        self.fn = fn
        self.device = device

        if device == 'bpu':
            pass

    def __call__(self, *args, **kwargs):
        pass


class JIT:
    def __init__(
        self,
        fn: Callable,
        device: str,
    ):
        self.fn = fn
        self.device = device

        if device not in registered_devices:
            raise ValueError(f"Device '{device}' is not registered.")
        self.device = device

    def __call__(self, *args, **kwargs):
        pass
