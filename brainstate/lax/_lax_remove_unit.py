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
    # math funcs remove unit (unary)
    'population_count', 'clz', 'sign',

    # math funcs remove unit (binary)
    'binary_func_placeholder',

    # logic funcs (unary)
    'unary_logic_func_placeholder',

    # logic funcs (binary)
    'eq', 'ne', 'ge', 'gt', 'le', 'lt',

    # indexing
    'argmax', 'argmin',

    # broadcasting
    'broadcast_shapes',
]

# math funcs remove unit (unary)
def population_count(x): pass
def clz(x): pass
def sign(x): pass

# math funcs remove unit (binary)
def binary_func_placeholder(x, y): pass

# logic funcs (unary)
def unary_logic_func_placeholder(x): pass

# logic funcs (binary)
def eq(x, y): pass
def ne(x, y): pass
def ge(x, y): pass
def gt(x, y): pass
def le(x, y): pass
def lt(x, y): pass

# indexing
def argmax(x): pass
def argmin(x): pass

# broadcasting
def broadcast_shapes(*shapes): pass