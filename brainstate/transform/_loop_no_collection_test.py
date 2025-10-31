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


from unittest import TestCase

import brainstate


class TestWhileLoop(TestCase):
    def test1(self):
        a = brainstate.State(1.)
        b = brainstate.State(20.)

        def cond(_):
            return a.value < b.value

        def body(_):
            a.value += 1.

        brainstate.transform.while_loop(cond, body, None)

        print(a.value, b.value)

    def test2(self):
        a = brainstate.State(1.)
        b = brainstate.State(20.)

        def cond(x):
            return a.value < b.value

        def body(x):
            a.value += x
            return x

        r = brainstate.transform.while_loop(cond, body, 1.)

        print(a.value, b.value, r)
