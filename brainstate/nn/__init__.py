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

from . import init
from ._activations import (
    tanh, relu, squareplus, softplus, soft_sign, sigmoid, silu, swish, log_sigmoid, elu,
    leaky_relu, hard_tanh, celu, selu, gelu, glu, logsumexp, log_softmax, softmax,
    standardize, one_hot, relu6, hard_sigmoid, hard_silu, hard_swish, hard_shrink, rrelu,
    mish, soft_shrink, prelu, tanh_shrink, softmin, sparse_plus, sparse_sigmoid,
)
from ._collective_ops import (
    call_order, call_all_fns, vmap_call_all_fns, init_all_states, vmap_init_all_states,
    reset_all_states, vmap_reset_all_states, assign_state_values,
)
from ._common import (
    EnvironContext, Vmap, ModuleMapper,
)
from ._conv import (
    Conv1d, Conv2d, Conv3d, ScaledWSConv1d, ScaledWSConv2d, ScaledWSConv3d,
    ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
)
from ._delay import (
    Delay, DelayAccess, StateWithDelay, InterpolationRegistry,
)
from ._dropout import (
    Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout, FeatureAlphaDropout, DropoutFixed,
)
from ._dynamics import (
    Dynamics, receive_update_output, not_receive_update_output, receive_update_input,
    not_receive_update_input, Prefetch, PrefetchDelay, PrefetchDelayAt, OutputDelayAt,
    init_maybe_prefetch,
)
from ._elementwise import (
    Threshold, ReLU, RReLU, Hardtanh, ReLU6, Sigmoid, Hardsigmoid, Tanh, SiLU, Mish,
    Hardswish, ELU, CELU, SELU, GLU, GELU, Hardshrink, LeakyReLU, LogSigmoid, Softplus,
    Softshrink, PReLU, Softsign, Tanhshrink, Softmin, Softmax, Softmax2d, LogSoftmax,
    Identity, SpikeBitwise,
)
from ._embedding import (
    Embedding,
)
from ._event_fixedprob import (
    FixedNumConn, EventFixedNumConn, EventFixedProb,
)
from ._event_linear import (
    EventLinear,
)
from ._exp_euler import (
    exp_euler_step,
)
from ._hidata import (
    HiData,
)
from ._linear import (
    Linear, ScaledWSLinear, SignedWLinear, SparseLinear, AllToAll, OneToOne, LoRA,
)
from ._metrics import (
    MetricState, Metric, AverageMetric, WelfordMetric, AccuracyMetric, MultiMetric,
    PrecisionMetric, RecallMetric, F1ScoreMetric, ConfusionMatrix,
)
from ._module import (
    Module, ElementWiseBlock, Sequential,
)
from ._normalizations import (
    weight_standardization, BatchNorm0d, BatchNorm1d, BatchNorm2d, BatchNorm3d,
    LayerNorm, RMSNorm, GroupNorm,
)
from ._paddings import (
    ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d, ReplicationPad2d,
    ReplicationPad3d, ZeroPad1d, ZeroPad2d, ZeroPad3d, ConstantPad1d, ConstantPad2d,
    ConstantPad3d, CircularPad1d, CircularPad2d, CircularPad3d,
)
from ._param import (
    Param, Const,
)
from ._poolings import (
    Flatten, Unflatten, AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d,
    MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, LPPool1d, LPPool2d, LPPool3d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d,
    AdaptiveMaxPool2d, AdaptiveMaxPool3d,
)
from ._regularization import (
    Regularization, ChainedReg, GaussianReg, L1Reg, L2Reg, ElasticNetReg, HuberReg,
    GroupLassoReg, TotalVariationReg, MaxNormReg, EntropyReg, OrthogonalReg,
    SpectralNormReg, StudentTReg, CauchyReg, UniformReg, LogNormalReg,
    ExponentialReg, GammaReg, BetaReg, HorseshoeReg, InverseGammaReg,
    LogUniformReg, SpikeAndSlabReg, DirichletReg,
)
from ._rnns import (
    RNNCell, ValinaRNNCell, GRUCell, MGUCell, LSTMCell, URLSTMCell,
)
from ._transform import (
    Transform, IdentityT, SigmoidT, SoftplusT, NegSoftplusT, LogT, ExpT,
    TanhT, SoftsignT, AffineT, ChainT, MaskedT, ClipT, ReluT, PositiveT,
    NegativeT, ScaledSigmoidT, PowerT, OrderedT, SimplexT, UnitVectorT,
)
from ._utils import (
    count_parameters, clip_grad_norm,
)

__all__ = [
    'init',
    'tanh',
    'relu',
    'squareplus',
    'softplus',
    'soft_sign',
    'sigmoid',
    'silu',
    'swish',
    'log_sigmoid',
    'elu',
    'leaky_relu',
    'hard_tanh',
    'celu',
    'selu',
    'gelu',
    'glu',
    'logsumexp',
    'log_softmax',
    'softmax',
    'standardize',
    'one_hot',
    'relu6',
    'hard_sigmoid',
    'hard_silu',
    'hard_swish',
    'hard_shrink',
    'rrelu',
    'mish',
    'soft_shrink',
    'prelu',
    'tanh_shrink',
    'softmin',
    'sparse_plus',
    'sparse_sigmoid',
    'call_order',
    'call_all_fns',
    'vmap_call_all_fns',
    'init_all_states',
    'vmap_init_all_states',
    'reset_all_states',
    'vmap_reset_all_states',
    'assign_state_values',
    'EnvironContext',
    'Vmap',
    'ModuleMapper',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ScaledWSConv1d',
    'ScaledWSConv2d',
    'ScaledWSConv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
    'Delay',
    'DelayAccess',
    'StateWithDelay',
    'InterpolationRegistry',
    'Dropout',
    'Dropout1d',
    'Dropout2d',
    'Dropout3d',
    'AlphaDropout',
    'FeatureAlphaDropout',
    'DropoutFixed',
    'Dynamics',
    'receive_update_output',
    'not_receive_update_output',
    'receive_update_input',
    'not_receive_update_input',
    'Prefetch',
    'PrefetchDelay',
    'PrefetchDelayAt',
    'init_maybe_prefetch',
    'OutputDelayAt',
    'Threshold',
    'ReLU',
    'RReLU',
    'Hardtanh',
    'ReLU6',
    'Sigmoid',
    'Hardsigmoid',
    'Tanh',
    'SiLU',
    'Mish',
    'Hardswish',
    'ELU',
    'CELU',
    'SELU',
    'GLU',
    'GELU',
    'Hardshrink',
    'LeakyReLU',
    'LogSigmoid',
    'Softplus',
    'Softshrink',
    'PReLU',
    'Softsign',
    'Tanhshrink',
    'Softmin',
    'Softmax',
    'Softmax2d',
    'LogSoftmax',
    'Identity',
    'SpikeBitwise',
    'Embedding',
    'FixedNumConn',
    'EventFixedNumConn',
    'EventFixedProb',
    'EventLinear',
    'exp_euler_step',
    'Linear',
    'ScaledWSLinear',
    'SignedWLinear',
    'SparseLinear',
    'AllToAll',
    'OneToOne',
    'LoRA',
    'MetricState',
    'Metric',
    'AverageMetric',
    'WelfordMetric',
    'AccuracyMetric',
    'MultiMetric',
    'PrecisionMetric',
    'RecallMetric',
    'F1ScoreMetric',
    'ConfusionMatrix',
    'Module',
    'ElementWiseBlock',
    'Sequential',
    'weight_standardization',
    'BatchNorm0d',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'RMSNorm',
    'GroupNorm',
    'ReflectionPad1d',
    'ReflectionPad2d',
    'ReflectionPad3d',
    'ReplicationPad1d',
    'ReplicationPad2d',
    'ReplicationPad3d',
    'ZeroPad1d',
    'ZeroPad2d',
    'ZeroPad3d',
    'ConstantPad1d',
    'ConstantPad2d',
    'ConstantPad3d',
    'CircularPad1d',
    'CircularPad2d',
    'CircularPad3d',
    'Param',
    'Const',
    'HiData',
    'Flatten',
    'Unflatten',
    'AvgPool1d',
    'AvgPool2d',
    'AvgPool3d',
    'MaxPool1d',
    'MaxPool2d',
    'MaxPool3d',
    'MaxUnpool1d',
    'MaxUnpool2d',
    'MaxUnpool3d',
    'LPPool1d',
    'LPPool2d',
    'LPPool3d',
    'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d',
    'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d',
    'RNNCell',
    'ValinaRNNCell',
    'GRUCell',
    'MGUCell',
    'LSTMCell',
    'URLSTMCell',
    'count_parameters',
    'clip_grad_norm',
    'Regularization',
    'ChainedReg',
    'GaussianReg',
    'L1Reg',
    'L2Reg',
    'ElasticNetReg',
    'HuberReg',
    'GroupLassoReg',
    'TotalVariationReg',
    'MaxNormReg',
    'EntropyReg',
    'OrthogonalReg',
    'SpectralNormReg',
    'StudentTReg',
    'CauchyReg',
    'UniformReg',
    'LogNormalReg',
    'ExponentialReg',
    'GammaReg',
    'BetaReg',
    'HorseshoeReg',
    'InverseGammaReg',
    'LogUniformReg',
    'SpikeAndSlabReg',
    'DirichletReg',
    'Transform',
    'IdentityT',
    'SigmoidT',
    'SoftplusT',
    'NegSoftplusT',
    'LogT',
    'ExpT',
    'TanhT',
    'SoftsignT',
    'AffineT',
    'ChainT',
    'MaskedT',
    'ClipT',
    'ReluT',
    'PositiveT',
    'NegativeT',
    'ScaledSigmoidT',
    'PowerT',
    'OrderedT',
    'SimplexT',
    'UnitVectorT',
]

# Deprecated names that redirect to brainpy
_DEPRECATED_NAMES = {
    'SpikeTime': 'brainpy.state.SpikeTime',
    'PoissonSpike': 'brainpy.state.PoissonSpike',
    'PoissonEncoder': 'brainpy.state.PoissonEncoder',
    'PoissonInput': 'brainpy.state.PoissonInput',
    'poisson_input': 'brainpy.state.poisson_input',
    'Neuron': 'brainpy.state.Neuron',
    'IF': 'brainpy.state.IF',
    'LIF': 'brainpy.state.LIF',
    'LIFRef': 'brainpy.state.LIFRef',
    'ALIF': 'brainpy.state.ALIF',
    'LeakyRateReadout': 'brainpy.state.LeakyRateReadout',
    'LeakySpikeReadout': 'brainpy.state.LeakySpikeReadout',
    'STP': 'brainpy.state.STP',
    'STD': 'brainpy.state.STD',
    'Synapse': 'brainpy.state.Synapse',
    'Expon': 'brainpy.state.Expon',
    'DualExpon': 'brainpy.state.DualExpon',
    'Alpha': 'brainpy.state.Alpha',
    'AMPA': 'brainpy.state.AMPA',
    'GABAa': 'brainpy.state.GABAa',
    'COBA': 'brainpy.state.COBA',
    'CUBA': 'brainpy.state.CUBA',
    'MgBlock': 'brainpy.state.MgBlock',
    'SynOut': 'brainpy.state.SynOut',
    'AlignPostProj': 'brainpy.state.AlignPostProj',
    'DeltaProj': 'brainpy.state.DeltaProj',
    'CurrentProj': 'brainpy.state.CurrentProj',
    'align_pre_projection': 'brainpy.state.align_pre_projection',
    'Projection': 'brainpy.state.Projection',
    'SymmetryGapJunction': 'brainpy.state.SymmetryGapJunction',
    'AsymmetryGapJunction': 'brainpy.state.AsymmetryGapJunction',
}


def __getattr__(name: str):
    import warnings
    if name == 'DynamicsGroup':
        warnings.warn(
            f"'brainstate.nn.{name}' is deprecated and will be removed in a future version. "
            f"Please use 'brainstate.nn.Module' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return Module

    if name in _DEPRECATED_NAMES:
        new_name = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"'brainstate.nn.{name}' is deprecated and will be removed in a future version. "
            f"Please use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Import and return the actual brainpy object
        import brainpy
        return getattr(brainpy.state, name)
    raise AttributeError(f"module 'brainstate.nn' has no attribute '{name}'")
