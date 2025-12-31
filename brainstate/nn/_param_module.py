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

"""
Neural network parameter modules with transform and regularization support.

This module provides parameter container classes that integrate with brainstate's
module system, supporting bijective transformations and regularization for
constrained optimization.
"""

from typing import Optional

import brainstate
import brainunit as u

from ._module import Module
from ._regularization import Regularization
from ._transform import IdentityT, Transform

__all__ = [
    'ParaM',
    'ConstM',
]

Data = brainstate.typing.ArrayLike


class ParaM(Module):
    """
    Neural network parameter with optional transform and regularization.

    A flexible parameter container that supports:

    - Bijective transformations for constrained optimization
    - Regularization (L1, L2, Gaussian, etc.)
    - Trainable or fixed parameter modes

    Parameters
    ----------
    value : array_like
        Initial parameter value in the constrained space.
    t : Transform, optional
        Bijective transformation to apply. Default is ``IdentityT()``.
    reg : Regularization, optional
        Regularization to apply. Default is ``None``.
    fit_par : bool, optional
        Whether the parameter is trainable. Default is ``True``.

    Attributes
    ----------
    fit_par : bool
        Whether the parameter is trainable.
    t : Transform
        The bijective transformation.
    reg : Regularization or None
        The regularization, if any.
    val : array_like or ParamState
        The internal parameter storage.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import ParaM, SoftplusT, L2Reg
    >>> # Trainable positive parameter with L2 regularization
    >>> param = ParaM(
    ...     jnp.array([1.0, 2.0]),
    ...     t=SoftplusT(0.0),
    ...     reg=L2Reg(weight=0.01)
    ... )
    >>> param.value()  # Get constrained value
    >>> param.reg_loss()  # Get regularization loss

    Notes
    -----
    The internal value is stored in the unconstrained space when a transform
    is provided. The ``value()`` method returns the constrained value after
    applying the forward transformation.
    """

    def __init__(
        self,
        value: Data,
        t: Transform = IdentityT(),
        reg: Optional[Regularization] = None,
        fit_par: bool = True,
    ):
        super().__init__()

        self.fit_par = fit_par
        self.t = t
        self.reg = reg

        # Convert value to tensor
        val_tensor = u.math.asarray(value, dtype=brainstate.environ.dftype())

        # Register reg as submodule if provided
        if not (reg is None or isinstance(reg, Regularization)):
            raise ValueError(
                'Regularization must be None or instance of '
                'Regularization.'
            )
        if not isinstance(t, Transform):
            raise TypeError(f't must be an instance of Transform. But got {type(t)}.')
        val_tensor = t.inverse(val_tensor)

        if fit_par:
            val_tensor = brainstate.ParamState(val_tensor)

        self.val = val_tensor

    def value(self) -> Data:
        """
        Get current parameter value after applying transform.

        Returns
        -------
        array_like
            Parameter value in the constrained space.
        """
        if isinstance(self.val, brainstate.State):
            val = self.val.value
        else:
            val = self.val
        return self.t.forward(val)

    def set_value(self, value: Data):
        """
        Set parameter value from constrained space.

        The value is transformed to unconstrained space for internal storage.

        Parameters
        ----------
        value : array_like
            New value in the constrained space.
        """
        value = self.t.inverse(value)
        if isinstance(self.val, brainstate.State):
            self.val.value = value
        else:
            self.val = value

    def reg_loss(self) -> Data:
        """
        Calculate regularization loss.

        Returns
        -------
        array_like
            Regularization loss. Returns 0.0 for fixed parameters
            or parameters without regularization.
        """
        if not self.fit_par:
            return 0.0

        if self.reg is None:
            return 0.0

        return self.reg.loss(self.value())

    def reset_to_prior(self):
        """
        Reset parameter value to regularization prior value.

        Only has effect if regularization is defined.
        """
        if self.reg is not None:
            self.set_value(self.reg.reset_value())

    def clip(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        """
        Clamp parameter value in-place.

        Parameters
        ----------
        min_val : float, optional
            Minimum value for clipping. Default is ``None`` (no lower bound).
        max_val : float, optional
            Maximum value for clipping. Default is ``None`` (no upper bound).
        """
        clipped_val = u.math.clip(self.value(), a_min=min_val, a_max=max_val)
        self.set_value(clipped_val)


class ConstM(ParaM):
    """
    Non-trainable constant parameter.

    A convenience class that creates a fixed (non-trainable) parameter.
    Equivalent to ``ParaM(value, fit_par=False)``.

    Parameters
    ----------
    value : array_like
        The constant value.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import ConstM
    >>> const = ConstM(jnp.array([1.0, 2.0]))
    >>> const.value()
    """

    def __init__(self, value: Data):
        super().__init__(value, fit_par=False)
