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

from typing import Optional, Tuple

import numpy as np

from brainstate.util import PrettyRepr, PrettyType, PrettyAttr

__all__ = ['Initializer', 'to_size']


class Initializer(PrettyRepr):
    """
    Base class for initializers.
    """
    __module__ = 'brainstate.init'

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __pretty_repr__(self):
        """
        Pretty repr for the object.
        """
        yield PrettyType(type=type(self))
        for name, value in vars(self).items():
            if name.startswith('_'):
                continue
            yield PrettyAttr(name, repr(value))


def to_size(x) -> Optional[Tuple[int]]:
    if isinstance(x, (tuple, list)):
        return tuple(x)
    if isinstance(x, (int, np.integer)):
        return (x,)
    if x is None:
        return x
    raise ValueError(f'Cannot make a size for {x}')
