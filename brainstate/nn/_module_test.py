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

import unittest

import brainstate


class TestModule(unittest.TestCase):
    def test_states(self):
        class A(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.State(brainstate.random.random(10, 20))
                self.b = brainstate.State(brainstate.random.random(10, 20))

        class B(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()
                self.b = brainstate.State(brainstate.random.random(10, 20))

        b = B()
        print()
        print(b.states())
        print(b.states())
        print(b.states(level=0))
        print(b.states(level=0))
