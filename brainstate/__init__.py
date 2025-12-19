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

"""
A ``State``-based Transformation System for Program Compilation and Augmentation
"""

__version__ = "0.2.8"
__versio_info__ = tuple(map(int, __version__.split('.')))

from . import environ
from . import graph
from . import mixin
from . import nn
from . import random
from . import transform
from . import typing
from . import util
# Create deprecated module proxies with scoped APIs
from ._deprecation import create_deprecated_module_proxy
from ._error import *
from ._error import __all__ as _error_all
from ._state import *
from ._state import __all__ as _state_all

# Augment module scope
_augment_apis = {
    'GradientTransform': 'brainstate.transform._autograd',
    'grad': 'brainstate.transform._autograd',
    'vector_grad': 'brainstate.transform._autograd',
    'hessian': 'brainstate.transform._autograd',
    'jacobian': 'brainstate.transform._autograd',
    'jacrev': 'brainstate.transform._autograd',
    'jacfwd': 'brainstate.transform._autograd',
    'vmap': 'brainstate.transform._mapping',
    'pmap': 'brainstate.transform._mapping',
    'map': 'brainstate.transform._mapping',
    'vmap_new_states': 'brainstate.transform._mapping',
}

augment = create_deprecated_module_proxy(
    deprecated_name='brainstate.augment',
    replacement_module=transform,
    replacement_name='brainstate.transform',
    scoped_apis=_augment_apis
)

# Compile module scope
_compile_apis = {
    'checkpoint': 'brainstate.transform._ad_checkpoint',
    'remat': 'brainstate.transform._ad_checkpoint',
    'cond': 'brainstate.transform._conditions',
    'switch': 'brainstate.transform._conditions',
    'ifelse': 'brainstate.transform._conditions',
    'jit_error_if': 'brainstate.transform._error_if',
    'jit': 'brainstate.transform._jit',
    'scan': 'brainstate.transform._loop_collect_return',
    'checkpointed_scan': 'brainstate.transform._loop_collect_return',
    'for_loop': 'brainstate.transform._loop_collect_return',
    'checkpointed_for_loop': 'brainstate.transform._loop_collect_return',
    'while_loop': 'brainstate.transform._loop_no_collection',
    'bounded_while_loop': 'brainstate.transform._loop_no_collection',
    'StatefulFunction': 'brainstate.transform._make_jaxpr',
    'make_jaxpr': 'brainstate.transform._make_jaxpr',
    'ProgressBar': 'brainstate.transform._progress_bar',
}

compile = create_deprecated_module_proxy(
    deprecated_name='brainstate.compile',
    replacement_module=transform,
    replacement_name='brainstate.transform',
    scoped_apis=_compile_apis
)

# Functional module scope - use direct attribute access from nn module
_functional_apis = {
    'weight_standardization': 'brainstate.nn._normalizations',
    'clip_grad_norm': 'brainstate.nn._others',
    'tanh': 'brainstate.nn._activations',
    'relu': 'brainstate.nn._activations',
    'squareplus': 'brainstate.nn._activations',
    'softplus': 'brainstate.nn._activations',
    'soft_sign': 'brainstate.nn._activations',
    'sigmoid': 'brainstate.nn._activations',
    'silu': 'brainstate.nn._activations',
    'swish': 'brainstate.nn._activations',
    'log_sigmoid': 'brainstate.nn._activations',
    'elu': 'brainstate.nn._activations',
    'leaky_relu': 'brainstate.nn._activations',
    'hard_tanh': 'brainstate.nn._activations',
    'celu': 'brainstate.nn._activations',
    'selu': 'brainstate.nn._activations',
    'gelu': 'brainstate.nn._activations',
    'glu': 'brainstate.nn._activations',
    'logsumexp': 'brainstate.nn._activations',
    'log_softmax': 'brainstate.nn._activations',
    'softmax': 'brainstate.nn._activations',
    'standardize': 'brainstate.nn._activations',
    'relu6': 'brainstate.nn._activations',
    'hard_sigmoid': 'brainstate.nn._activations',
    'sparse_plus': 'brainstate.nn._activations',
    'hard_silu': 'brainstate.nn._activations',
    'hard_swish': 'brainstate.nn._activations',
    'hard_shrink': 'brainstate.nn._activations',
    'rrelu': 'brainstate.nn._activations',
    'mish': 'brainstate.nn._activations',
    'soft_shrink': 'brainstate.nn._activations',
    'prelu': 'brainstate.nn._activations',
    'softmin': 'brainstate.nn._activations',
    'one_hot': 'brainstate.nn._activations',
    'sparse_sigmoid': 'brainstate.nn._activations',
}

functional = create_deprecated_module_proxy(
    deprecated_name='brainstate.functional',
    replacement_module=nn,
    replacement_name='brainstate.nn',
    scoped_apis=_functional_apis
)


def __getattr__(name):
    if name in ['surrogate', 'init', 'optim']:
        import warnings
        warnings.warn(
            f"brainstate.{name} module is deprecated and will be removed in a future version. "
            f"Please use braintools.{name} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        import braintools
        return getattr(braintools, name)
    raise AttributeError(
        f'module {__name__!r} has no attribute {name!r}'
    )


__all__ = [
    'environ',
    'graph',
    'mixin',
    'nn',
    'random',
    'transform',
    'typing',
    'util',
    # Deprecated modules
    'augment',
    'compile',
    'functional',
]
__all__ = __all__ + _state_all + _error_all
del _state_all, create_deprecated_module_proxy, _augment_apis, _compile_apis, _functional_apis
del _error_all
