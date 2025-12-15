# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
import matplotlib.pyplot as plt

import brainstate
from brainstate.experimental.neuroir import compile_fn
from brainstate.experimental.neuroir._model_to_test import TwoPopNet

import pytest

pytest.skip('Test is not implemented yet.', allow_module_level=True)


class TestDisplayer:
    def test_visualize(self):
        net = TwoPopNet()
        brainstate.nn.init_all_states(net)

        def update(t, inp_exc, inp_inh):
            return net.update(t, inp_exc, inp_inh)

        t = 0. * u.ms
        inp_exc = 5. * u.mA
        inp_inh = 3. * u.mA

        parser = compile_fn(update)
        compiled = parser(t, inp_exc, inp_inh)

        fig = compiled.graph.visualize(layout='tb')
        plt.show()
        plt.close()


class TestTextDisplayer:
    def test_text(self):
        net = TwoPopNet()
        brainstate.nn.init_all_states(net)

        def update(t, inp_exc, inp_inh):
            return net.update(t, inp_exc, inp_inh)

        t = 0. * u.ms
        inp_exc = 5. * u.mA
        inp_inh = 3. * u.mA

        parser = compile_fn(update)
        compiled = parser(t, inp_exc, inp_inh)

        # Test 1: Basic text display (default)
        print("=== Test 1: Basic Display ===")
        print(compiled.graph)
        print()

        # Test 2: Verbose display
        print("=== Test 2: Verbose Display ===")
        from brainstate.experimental.neuroir import TextDisplayer
        displayer = TextDisplayer(compiled.graph)
        print(displayer.display(verbose=True))
        print()

        # Test 3: With JAXPR (first 3 lines only to keep output manageable)
        print("=== Test 3: Display with JAXPR (showing first 50 lines) ===")
        full_output = displayer.display(verbose=False, show_jaxpr=True)
        print(full_output)
