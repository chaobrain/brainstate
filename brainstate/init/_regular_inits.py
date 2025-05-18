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

# -*- coding: utf-8 -*-

import brainunit as u

from brainstate import environ
from ._base import Initializer, to_size

__all__ = [
    'ZeroInit',
    'Constant',
    'Identity',
]


class ZeroInit(Initializer):
    """Zero initializer.

    Initialize the weights with zeros.
    """
    __module__ = 'brainstate.init'

    def __init__(self, unit: u.Unit = u.UNITLESS):
        super(ZeroInit, self).__init__()
        self.unit = unit

    def __call__(self, shape, dtype=None):
        dtype = dtype or environ.dftype()
        shape = to_size(shape)
        return u.maybe_decimal(u.math.zeros(shape, dtype=dtype, unit=self.unit))


class Constant(Initializer):
    """Constant initializer.

    Initialize the weights with the given values.

    Parameters
    ----------
    value : float, int, bm.ndarray
      The value to specify.
    """
    __module__ = 'brainstate.init'

    def __init__(self, value=1., ):
        super(Constant, self).__init__()
        self.value = value

    def __call__(self, shape, dtype=None):
        dtype = dtype or environ.dftype()
        shape = to_size(shape)
        return u.maybe_decimal(u.math.full(shape, self.value, dtype=dtype))


class Identity(Initializer):
    """Returns the identity matrix.

    This initializer was proposed in (Le, et al., 2015) [1]_.

    Parameters
    ----------
    value : float
      The optional scaling factor.

    Returns
    -------
    shape: tuple of int
      The weight shape/size.

    References
    ----------
    .. [1] Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. "A simple way to
           initialize recurrent networks of rectified linear units." arXiv preprint
           arXiv:1504.00941 (2015).
    """
    __module__ = 'brainstate.init'

    def __init__(self, value=1., unit: u.Unit = u.UNITLESS):
        super(Identity, self).__init__()
        self.value = value
        self.unit = unit

    def __call__(self, shape, dtype=None):
        dtype = dtype or environ.dftype()
        shape = to_size(shape)
        if isinstance(shape, (tuple, list)):
            if len(shape) > 2:
                raise ValueError(f'Only support initialize 2D weights for {self.__class__.__name__}.')
        r = u.math.eye(*shape, dtype=dtype)
        r = u.math.fill_diagonal(r, self.value)
        return u.maybe_decimal(u.Quantity(r, unit=self.unit))
