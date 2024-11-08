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
A ``State``-based Transformation System for Program Compilation and Augmentation
"""

__version__ = "0.1.0"

from . import augment
from . import compile
from . import environ
from . import event
from . import functional
from . import graph
from . import init
from . import lax
from . import mixin
from . import nn
from . import optim
from . import random
from . import surrogate
from . import transform
from . import typing
from . import util
from ._state import *
from ._state import __all__ as _state_all

__all__ = (
    [
        'augment', 'compile', 'environ', 'event', 'functional',
        'graph', 'init', 'lax', 'mixin', 'nn', 'optim', 'random',
        'surrogate', 'typing', 'util',
        # deprecated
        'transform',
    ] +
    _state_all
)

# ----------------------- #
# deprecations
# ----------------------- #

from ._utils import deprecation_getattr

transform._deprecations = dict()
for key in compile.__all__:
    transform._deprecations[key] = (f'brainstate.transform.{key}', f'brainstate.compile.{key}', getattr(compile, key))
for key in augment.__all__:
    transform._deprecations[key] = (f'brainstate.transform.{key}', f'brainstate.augment.{key}', getattr(augment, key))
transform.__getattr__ = deprecation_getattr('brainstate.transform', transform._deprecations)
del deprecation_getattr

# ----------------------- #
del _state_all
