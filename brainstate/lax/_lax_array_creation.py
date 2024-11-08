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

__all__ = [
    # array creation(given shape)
    'full',

    # array creation(given array)
    'full_like', 'zeros_like_array',

    # array creation(misc)
    'asarray', 'iota', 'broadcasted_iota', 'zeros_like_shaped_array',

    # indexing funcs

    # others

]

# array creation (given shape)
def full(shape, fill_value, dtype=None, order='C'): pass

# array creation (given array)
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None): pass
def zeros_like_array(a, dtype=None, order='K', subok=True, shape=None): pass

# array creation (misc)
def asarray(a, dtype=None, order=None): pass
def iota(start, stop, step=1, dtype=None): pass
def broadcasted_iota(shape, start, stop, step=1, dtype=None): pass
def zeros_like_shaped_array(a, dtype=None, order='K', subok=True, shape=None): pass