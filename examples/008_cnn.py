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

import brainstate


class CNNNet(brainstate.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.layer = brainstate.nn.Sequential(
            brainstate.nn.Conv2d(in_size, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
            brainstate.nn.ReLU(),
            brainstate.nn.MaxPool2d.desc(kernel_size=(2, 2), stride=(2, 2), channel_axis=-1),
            brainstate.nn.Conv2d.desc(out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
            brainstate.nn.ReLU(),
            brainstate.nn.MaxPool2d.desc(kernel_size=(2, 2), stride=(2, 2), channel_axis=-1),
            brainstate.nn.Flatten.desc(),
            brainstate.nn.Linear.desc(out_size=1024),
            brainstate.nn.ReLU(),
            brainstate.nn.Linear.desc(out_size=512),
            brainstate.nn.ReLU(),
            brainstate.nn.Linear.desc(out_size=10)
        )

    def update(self, x):
        x = self.layer(x)
        return x


example_imag = brainstate.random.normal(size=(28, 28, 3))
cnn = CNNNet(example_imag.shape)



