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
    'after_all', 'reduce', 'stop_gradient', 'reduce_precision',

    # getting attribute funcs
    'broadcast_shapes', 'is_finite',

    # convolution
    'conv_dimension_numbers', 'conv_general_dilated', 'conv_general_dilated_local', 'conv_general_dilated_patches',
    'conv_with_general_padding',
]

def after_all(*args): pass
def reduce(x): pass
def stop_gradient(x): pass
def reduce_precision(x, precision): pass

def broadcast_shapes(*shapes): pass
def is_finite(x): pass

def conv_dimension_numbers(x): pass
def conv_general_dilated(x, y, window_strides, padding): pass
def conv_general_dilated_local(x, y, window_strides, padding): pass
def conv_general_dilated_patches(x, y, window_strides, padding): pass
def conv_with_general_padding(x, y, window_strides, padding): pass