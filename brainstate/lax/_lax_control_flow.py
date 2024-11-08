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
  'associative_scan', 'cond', 'fori_loop', 'map', 'scan',
  'select', 'select_n', 'switch', 'while_loop',
]

def associative_scan(x): pass
def cond(pred, true_fn, false_fn): pass
def fori_loop(lower, upper, body_fn, init_val): pass
def map(fn, xs): pass
def scan(fn, init, xs): pass
def select(pred, on_true, on_false): pass
def select_n(preds, on_true, on_false): pass
def switch(index, branches): pass
def while_loop(cond_fn, body_fn, init_val): pass