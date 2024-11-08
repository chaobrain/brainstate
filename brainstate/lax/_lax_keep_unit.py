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
    # sequence inputs
    'concatenate',

    # array manipulation
    'reshape', 'reverse', 'transpose', 'slice', 'dynamic_slice', 'dynamic_update_slice', 'gather',
    'index_take', 'slice_in_dim', 'index_in_dim', 'dynamic_slice_ind_dim', 'dynamic_index_in_dim',
    'dynamic_update_slice_in_dim', 'dynamic_update_index_in_dim',
    'sort', 'sort_key_val', 'squeeze', 'expand_dims',
    
    # math funcs keep unit (unary)
    'neg', 'real', 'imag', 'complex', 'conj', 'abs',
    'cummax', 'cummin', 'cumsum',
    'scatter', 'scatter_add', 'scatter_sub', 'scatter_mul', 'scatter_min', 'scatter_max', 'scatter_apply',
    'pad',

    # type conversion
    'convert_element_type', 'bitcast_convert_type',

    # math funcs keep unit (binary)
    'add', 'sub', 'nextafter', 'max', 'min',

    # math funcs keep unit (n-ary)
    'clamp',

    # math funcs keep unit (return Quantity and index)
    'approx_max_k', 'approx_min_k', 'top_k',

    # math funcs only accept unitless (unary) can return Quantity
    'ceil', 'floor', 'round',

    # broadcasting arrays
    'broadcast', 'broadcast_in_dim', 'broadcast_to_randk',

    # selection
    'select', 'select_n'
]

# sequence inputs
def concatenate(*args): pass

# array manipulation
def reshape(x, shape): pass
def reverse(x): pass
def transpose(x, axes=None): pass
def slice(x, start, stop, step=None): pass
def dynamic_slice(x, start_indices, slice_sizes): pass
def dynamic_update_slice(x, update, start_indices): pass
def gather(x, indices, axis=None): pass
def index_take(x, indices, axis=None): pass
def slice_in_dim(x, start, stop, axis=0): pass
def index_in_dim(x, index, axis=0): pass
def dynamic_slice_ind_dim(x, start_indices, slice_sizes, axis=0): pass
def dynamic_index_in_dim(x, index, axis=0): pass
def dynamic_update_slice_in_dim(x, update, start_indices, axis=0): pass
def dynamic_update_index_in_dim(x, update, index, axis=0): pass
def sort(x, axis=-1, kind='quicksort', order=None): pass
def sort_key_val(keys, values, axis=-1): pass
def squeeze(x, axis=None): pass
def expand_dims(x, axis): pass

# math funcs keep unit (unary)
def neg(x): pass
def real(x): pass
def imag(x): pass
def complex(x, y): pass
def conj(x): pass
def abs(x): pass
def cummax(x, axis=None): pass
def cummin(x, axis=None): pass
def cumsum(x, axis=None): pass
def scatter(x, indices, updates): pass
def scatter_add(x, indices, updates): pass
def scatter_sub(x, indices, updates): pass
def scatter_mul(x, indices, updates): pass
def scatter_min(x, indices, updates): pass
def scatter_max(x, indices, updates): pass
def scatter_apply(x, indices, updates, op): pass
def pad(x, pad_width, mode='constant', constant_values=0): pass

# type conversion
def convert_element_type(x, dtype): pass
def bitcast_convert_type(x, dtype): pass

# math funcs keep unit (binary)
def add(x, y): pass
def sub(x, y): pass
def nextafter(x, y): pass
def max(x, y): pass
def min(x, y): pass

# math funcs keep unit (n-ary)
def clamp(x, min, max): pass

# math funcs keep unit (return Quantity and index)
def approx_max_k(x, k): pass
def approx_min_k(x, k): pass
def top_k(x, k): pass

# math funcs only accept unitless (unary) can return Quantity
def ceil(x): pass
def floor(x): pass
def round(x): pass

# broadcasting arrays
def broadcast(x, shape): pass
def broadcast_in_dim(x, shape, broadcast_dimensions): pass
def broadcast_to_randk(x, shape): pass

# selection
def select(pred, on_true, on_false): pass
def select_n(preds, on_trues, on_falses): pass