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


from . import metrics
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
from ._exp_euler import *
from ._exp_euler import __all__ as exp_euler_all
from ._fixedprob import *
from ._fixedprob import __all__ as fixedprob_all
from ._inputs import *
from ._inputs import __all__ as inputs_all
from ._linear import *
from ._linear import __all__ as linear_all
from ._linear_mv import *
from ._linear_mv import __all__ as linear_mv_all
from ._ltp import *
from ._ltp import __all__ as ltp_all
from ._module import *
from ._module import __all__ as module_all
from ._neuron import *
from ._neuron import __all__ as dyn_neuron_all
from ._normalizations import *
from ._normalizations import __all__ as normalizations_all
from ._others import *
from ._others import __all__ as _others_all
from ._poolings import *
from ._poolings import __all__ as poolings_all
from ._projection import *
from ._projection import __all__ as projection_all
from ._rate_rnns import *
from ._rate_rnns import __all__ as rate_rnns
from ._readout import *
from ._readout import __all__ as readout_all
from ._stp import *
from ._stp import __all__ as stp_all
from ._synapse import *
from ._synapse import __all__ as dyn_synapse_all
from ._synaptic_projection import *
from ._synaptic_projection import __all__ as _syn_proj_all
from ._synouts import *
from ._synouts import __all__ as synouts_all
from ._utils import *
from ._utils import __all__ as utils_all

__all__ = (
    [
        'metrics',
    ]
    + collective_ops_all
    + common_all
    + elementwise_all
    + module_all
    + exp_euler_all
    + utils_all
    + dyn_all
    + projection_all
    + state_delay_all
    + synouts_all
    + conv_all
    + linear_all
    + normalizations_all
    + poolings_all
    + fixedprob_all
    + linear_mv_all
    + embed_all
    + dropout_all
    + elementwise_all
    + dyn_neuron_all
    + dyn_synapse_all
    + inputs_all
    + rate_rnns
    + readout_all
    + stp_all
    + ltp_all
    + _syn_proj_all
    + _others_all
)

del (
    collective_ops_all,
    common_all,
    module_all,
    exp_euler_all,
    utils_all,
    dyn_all,
    projection_all,
    state_delay_all,
    synouts_all,
    conv_all,
    linear_all,
    normalizations_all,
    poolings_all,
    embed_all,
    fixedprob_all,
    linear_mv_all,
    dropout_all,
    elementwise_all,
    dyn_neuron_all,
    dyn_synapse_all,
    inputs_all,
    readout_all,
    rate_rnns,
    stp_all,
    ltp_all,
    _syn_proj_all,
    _others_all,
)
