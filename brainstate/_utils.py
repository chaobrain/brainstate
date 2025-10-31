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


import warnings


def set_module_as(module: str):
    def wrapper(fun: callable):
        fun.__module__ = module
        return fun

    return wrapper


def _deprecate(msg):
    warnings.simplefilter('always', DeprecationWarning)  # turn off filter
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    warnings.simplefilter('default', DeprecationWarning)  # reset filter


def deprecation_getattr(module, deprecations):
    def get_attr(name):
        if name in deprecations:
            old_name, new_name, fn = deprecations[name]
            message = f"{old_name} is deprecated. "
            if new_name is not None:
                message += f'Use {new_name} instead.'
            if fn is None:
                raise AttributeError(message)
            _deprecate(message)
            return fn
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return get_attr
