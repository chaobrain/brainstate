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
from ._activations import *
from ._activations import __all__ as activation_all
from ._collective_ops import *
from ._collective_ops import __all__ as collective_ops_all
from ._common import *
from ._common import __all__ as common_all
from ._conv import *
from ._conv import __all__ as conv_all
from ._delay import *
from ._delay import __all__ as state_delay_all
from ._dropout import *
from ._dropout import __all__ as dropout_all
from ._dynamics import *
from ._dynamics import __all__ as dyn_all
from ._elementwise import *
from ._elementwise import __all__ as elementwise_all
from ._embedding import *
from ._embedding import __all__ as embed_all
from ._event_fixedprob import *
from ._event_fixedprob import __all__ as fixedprob_all
from ._event_linear import *
from ._event_linear import __all__ as linear_mv_all
from ._exp_euler import *
from ._exp_euler import __all__ as exp_euler_all
from ._linear import *
from ._linear import __all__ as linear_all
from ._metrics import *
from ._metrics import __all__ as metrics_all
from ._module import *
from ._module import __all__ as module_all
from ._normalizations import *
from ._normalizations import __all__ as normalizations_all
from ._paddings import *
from ._paddings import __all__ as paddings_all
from ._poolings import *
from ._poolings import __all__ as poolings_all
from ._rnns import *
from ._rnns import __all__ as rate_rnns
from ._utils import *
from ._utils import __all__ as utils_all

__all__ = ['init'] + activation_all + metrics_all
__all__ = __all__ + collective_ops_all + common_all + elementwise_all + module_all + exp_euler_all
__all__ = __all__ + utils_all + dyn_all + state_delay_all + conv_all
__all__ = __all__ + linear_all + normalizations_all + paddings_all + poolings_all + fixedprob_all + linear_mv_all
__all__ = __all__ + embed_all + dropout_all + elementwise_all
__all__ = __all__ + rate_rnns

del (
    metrics_all,
    activation_all,
    collective_ops_all,
    common_all,
    module_all,
    exp_euler_all,
    utils_all,
    dyn_all,
    state_delay_all,
    conv_all,
    linear_all,
    normalizations_all,
    paddings_all,
    poolings_all,
    embed_all,
    fixedprob_all,
    linear_mv_all,
    dropout_all,
    elementwise_all,
    rate_rnns,
)

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
