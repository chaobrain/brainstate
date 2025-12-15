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


import brainunit as u

import brainstate
from brainstate.experimental._gdiist_bpu import GdiistBPUParser
from brainstate.experimental.neuroir._model_to_test import Single_Pop_EI_COBA_Net

brainstate.environ.set(dt=0.1 * u.ms)

import pytest

pytest.skip('Test is not implemented yet.', allow_module_level=True)


def test_parse():
    net = Single_Pop_EI_COBA_Net()
    brainstate.nn.init_all_states(net)

    t = 0. * u.ms
    inp = 20. * u.mA

    def run_step(t, inp):
        with brainstate.environ.context(t=t):
            spikes = net.update(t, inp)
            return spikes

    parser = GdiistBPUParser(run_step, debug=True)
    r = parser(t, inp)

