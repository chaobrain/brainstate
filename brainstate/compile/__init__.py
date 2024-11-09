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

from ._ad_checkpoint import *
from ._ad_checkpoint import __all__ as _ad_checkpoint_all
from ._conditions import *
from ._conditions import __all__ as _conditions_all
from ._error_if import *
from ._error_if import __all__ as _jit_error_all
from ._jit import *
from ._jit import __all__ as _jit_all
from ._loop_collect_return import *
from ._loop_collect_return import __all__ as _loops_collection
from ._loop_no_collection import *
from ._loop_no_collection import __all__ as _loops_no_collection
from ._make_jaxpr import *
from ._make_jaxpr import __all__ as _make_jaxpr_all
from ._progress_bar import *
from ._progress_bar import __all__ as _progress_bar_all

__all__ = (
    _jit_error_all
    + _conditions_all
    + _make_jaxpr_all
    + _jit_all
    + _progress_bar_all
    + _loops_collection
    + _loops_no_collection
    + _ad_checkpoint_all
)

del (
    _jit_error_all,
    _conditions_all,
    _loops_collection,
    _make_jaxpr_all,
    _jit_all,
    _progress_bar_all,
    _loops_no_collection,
    _ad_checkpoint_all
)
