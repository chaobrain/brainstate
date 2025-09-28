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

"""
This module contains the functions for the compilation of JAX code.
"""

from brainstate.transform._ad_checkpoint import checkpoint, remat
from brainstate.transform._conditions import cond, switch, ifelse
from brainstate.transform._error_if import jit_error_if
from brainstate.transform._jit import jit
from brainstate.transform._loop_collect_return import scan, checkpointed_scan, for_loop, checkpointed_for_loop
from brainstate.transform._loop_no_collection import while_loop, bounded_while_loop
from brainstate.transform._make_jaxpr import StatefulFunction, make_jaxpr
from brainstate.transform._progress_bar import ProgressBar

__all__ = [
    'checkpoint', 'remat',
    'cond', 'switch', 'ifelse',
    'jit_error_if',
    'jit',
    'scan', 'checkpointed_scan', 'for_loop', 'checkpointed_for_loop',
    'while_loop', 'bounded_while_loop',
    'StatefulFunction', 'make_jaxpr',
    'ProgressBar',
]
