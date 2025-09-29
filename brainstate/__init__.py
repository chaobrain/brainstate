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

__version__ = "0.1.11"
__versio_info__ = (0, 1, 11)

from . import environ
from . import graph
from . import mixin
from . import nn
from . import optim
from . import random
from . import surrogate
from . import transform
from . import typing
from . import util
from ._deprecation import create_deprecated_module_proxy

# Create deprecated module proxies with scoped APIs

# Augment module scope
_augment_apis = {
    'GradientTransform': 'brainstate.transform._autograd',
    'grad': 'brainstate.transform._autograd',
    'vector_grad': 'brainstate.transform._autograd',
    'hessian': 'brainstate.transform._autograd',
    'jacobian': 'brainstate.transform._autograd',
    'jacrev': 'brainstate.transform._autograd',
    'jacfwd': 'brainstate.transform._autograd',
    'abstract_init': 'brainstate.transform._eval_shape',
    'vmap': 'brainstate.transform._mapping',
    'pmap': 'brainstate.transform._mapping',
    'map': 'brainstate.transform._mapping',
    'vmap_new_states': 'brainstate.transform._mapping',
    'restore_rngs': 'brainstate.transform._random',
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
    'standardize': 'brainstate.nn._activations'
}

functional = create_deprecated_module_proxy(
    deprecated_name='brainstate.functional',
    replacement_module=nn,
    replacement_name='brainstate.nn',
    scoped_apis=_functional_apis
)

_init_apis = {
    'param': 'brainstate.nn.param',
    'ZeroInit': 'brainstate.nn.ZeroInitInit',
    'ConstantInit': 'brainstate.nn.ConstantInit',
    'Identity': 'brainstate.nn.IdentityInit',
    'Normal': 'brainstate.nn.NormalInit',
    'TruncatedNormal': 'brainstate.nn.TruncatedNormalInit',
    'Uniform': 'brainstate.nn.UniformInit',
    'VarianceScaling': 'brainstate.nn.VarianceScalingInit',
    'KaimingUniform': 'brainstate.nn.KaimingUniformInit',
    'KaimingNormal': 'brainstate.nn.KaimingNormalInit',
    'XavierUniform': 'brainstate.nn.XavierUniformInit',
    'XavierNormal': 'brainstate.nn.XavierNormalInit',
    'LecunUniform': 'brainstate.nn.LecunUniformInit',
    'LecunNormal': 'brainstate.nn.LecunNormalInit',
    'Orthogonal': 'brainstate.nn.OrthogonalInit',
    'DeltaOrthogonal': 'brainstate.nn.DeltaOrthogonalInit',
}

init = create_deprecated_module_proxy(
    deprecated_name='brainstate.init',
    replacement_module=nn,
    replacement_name='brainstate.nn',
    scoped_apis=_init_apis
)

from ._state import *
from ._state import __all__ as _state_all

__all__ = [
    'environ',
    'graph',
    'init',
    'mixin',
    'nn',
    'optim',
    'random',
    'surrogate',
    'transform',
    'typing',
    'util',
    # Deprecated modules
    'augment',
    'compile',
    'functional',
]
__all__ = __all__ + _state_all
del _state_all, create_deprecated_module_proxy, _augment_apis, _compile_apis, _functional_apis, _init_apis
