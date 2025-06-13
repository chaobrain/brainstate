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

from typing import Union, Sequence

__all__ = [
    'MemScaling',
    'IdMemScaling',
]


class MemScaling(object):
    """
    The scaling object for membrane potential.

    The scaling object is used to transform the membrane potential range to a
    standard range. The scaling object can be used to transform the membrane
    potential to a standard range, and transform the standard range to the
    membrane potential.

    """
    __module__ = 'brainstate.util'

    def __init__(self, scale, bias):
        self._scale = scale
        self._bias = bias

    @classmethod
    def transform(
        cls,
        oring_range: Sequence[Union[float, int]],
        target_range: Sequence[Union[float, int]] = (0., 1.)
    ) -> 'MemScaling':
        """Transform the membrane potential range to a ``Scaling`` instance.

        Args:
          oring_range:   [V_min, V_max]
          target_range:  [scaled_V_min, scaled_V_max]

        Returns:
          The instanced scaling object.
        """
        V_min, V_max = oring_range
        scaled_V_min, scaled_V_max = target_range
        scale = (V_max - V_min) / (scaled_V_max - scaled_V_min)
        bias = scaled_V_min * scale - V_min
        return cls(scale=scale, bias=bias)

    def scale_offset(self, x, bias=None, scale=None):
        """
        Transform the membrane potential to the standard range.

        Parameters
        ----------
        x : array_like
          The membrane potential.
        bias : float, optional
          The bias of the scaling object. If None, the default bias will be used.
        scale : float, optional
          The scale of the scaling object. If None, the default scale will be used.

        Returns
        -------
        x : array_like
          The standard range of the membrane potential.
        """
        if bias is None:
            bias = self._bias
        if scale is None:
            scale = self._scale
        return (x + bias) / scale

    def scale(self, x, scale=None):
        """
        Transform the membrane potential to the standard range.

        Parameters
        ----------
        x : array_like
          The membrane potential.
        scale : float, optional
          The scale of the scaling object. If None, the default scale will be used.

        Returns
        -------
        x : array_like
          The standard range of the membrane potential.
        """
        if scale is None:
            scale = self._scale
        return x / scale

    def offset(self, x, bias=None):
        """
        Transform the membrane potential to the standard range.

        Parameters
        ----------
        x : array_like
          The membrane potential.
        bias : float, optional
          The bias of the scaling object. If None, the default bias will be used.

        Returns
        -------
        x : array_like
          The standard range of the membrane potential.
        """
        if bias is None:
            bias = self._bias
        return x + bias

    def rev_scale(self, x, scale=None):
        """
        Reversely transform the standard range to the original membrane potential.

        Parameters
        ----------
        x : array_like
          The standard range of the membrane potential.
        scale : float, optional
          The scale of the scaling object. If None, the default scale will be used.

        Returns
        -------
        x : array_like
          The original membrane potential.
        """
        if scale is None:
            scale = self._scale
        return x * scale

    def rev_offset(self, x, bias=None):
        """
        Reversely transform the standard range to the original membrane potential.

        Parameters
        ----------
        x : array_like
          The standard range of the membrane potential.
        bias : float, optional
          The bias of the scaling object. If None, the default bias will be used.

        Returns
        -------
        x : array_like
          The original membrane potential.
        """
        if bias is None:
            bias = self._bias
        return x - bias

    def rev_scale_offset(self, x, bias=None, scale=None):
        """
        Reversely transform the standard range to the original membrane potential.

        Parameters
        ----------
        x : array_like
          The standard range of the membrane potential.
        bias : float, optional
          The bias of the scaling object. If None, the default bias will be used.
        scale : float, optional
          The scale of the scaling object. If None, the default scale will be used.

        Returns
        -------
        x : array_like
          The original membrane potential.
        """
        if bias is None:
            bias = self._bias
        if scale is None:
            scale = self._scale
        return x * scale - bias

    def clone(self):
        """
        Clone the scaling object.

        Returns
        -------
        scaling : MemScaling
          The cloned scaling object.
        """
        return MemScaling(bias=self._bias, scale=self._scale)


class IdMemScaling(MemScaling):
    """
    The identity scaling object.

    The identity scaling object is used to transform the membrane potential to
    the standard range, and reversely transform the standard range to the
    membrane potential.

    """
    __module__ = 'brainstate.util'

    def __init__(self):
        super().__init__(scale=1., bias=0.)

    def scale_offset(self, x, bias=None, scale=None):
        """
        Transform the membrane potential to the standard range.
        """
        return x

    def scale(self, x, scale=None):
        """
        Transform the membrane potential to the standard range.
        """
        return x

    def offset(self, x, bias=None):
        """
        Transform the membrane potential to the standard range.
        """
        return x

    def rev_scale(self, x, scale=None):
        """
        Reversely transform the standard range to the original membrane potential.

        """
        return x

    def rev_offset(self, x, bias=None):
        """
        Reversely transform the standard range to the original membrane potential.


        """
        return x

    def rev_scale_offset(self, x, bias=None, scale=None):
        """
        Reversely transform the standard range to the original membrane potential.
        """
        return x

    def clone(self):
        """
        Clone the scaling object.
        """
        return IdMemScaling()
