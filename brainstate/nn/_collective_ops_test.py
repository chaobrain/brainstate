# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import brainstate


class Test_vmap_init_all_states:

    def test_vmap_init_all_states(self):
        gru = brainstate.nn.GRUCell(1, 2)
        brainstate.nn.vmap_init_all_states(gru, axis_size=10)
        print(gru)

    def test_vmap_init_all_states_v2(self):
        @brainstate.compile.jit
        def init():
            gru = brainstate.nn.GRUCell(1, 2)
            brainstate.nn.vmap_init_all_states(gru, axis_size=10)
            print(gru)

        init()


class Test_init_all_states:
    def test_init_all_states(self):
        gru = brainstate.nn.GRUCell(1, 2)
        brainstate.nn.init_all_states(gru, batch_size=10)
        print(gru)
