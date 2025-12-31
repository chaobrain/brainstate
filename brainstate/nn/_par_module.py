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

import logging
import threading
from typing import Optional

import brainunit as u

import brainstate
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
    A module has neural network parameters for optional transform and regularization.

    A flexible parameter container that supports:

    - Bijective transformations for constrained optimization
    - Regularization (L1, L2, Gaussian, etc.)
    - Trainable or fixed parameter modes
    - Automatic caching of transformed values for performance

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
    enable_cache_logging : bool, optional
        Whether to enable INFO-level logging for cache events. Default is ``False``.
        Logs cache hits, misses, invalidations, and errors for debugging.

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

    >>> # Caching is automatic for all parameters
    >>> param = ParaM(
    ...     jnp.array([1.0, 2.0]),
    ...     t=SoftplusT()
    ... )
    >>> val1 = param.value()  # Computes and caches
    >>> val2 = param.value()  # Returns cached value (fast)
    >>> param.set_value(jnp.array([3.0, 4.0]))  # Invalidates cache
    >>> val3 = param.value()  # Recomputes and caches

    Notes
    -----
    The internal value is stored in the unconstrained space when a transform
    is provided. The ``value()`` method returns the constrained value after
    applying the forward transformation.

    **Caching behavior**: The transformed value is cached on first access
    and automatically invalidated when the parameter is updated (via ``set_value()``
    or direct state writes). Use ``clear_cache()`` for manual invalidation.
    The caching mechanism is thread-safe using RLock.
    """

    def __init__(
        self,
        value: Data,
        t: Transform = IdentityT(),
        reg: Optional[Regularization] = None,
        fit_par: bool = True,
        enable_cache_logging: bool = False,
    ):
        super().__init__()

        self.fit_par = fit_par
        self.t = t
        self.reg = reg

        # Initialize cache infrastructure (always enabled)
        self._enable_cache_logging = enable_cache_logging
        self._cache_lock = threading.RLock()
        self._cached_value: Optional[Data] = None
        self._cache_valid = False
        self._cache_logger: Optional[logging.Logger] = None
        self._cache_invalidation_hook_handle = None

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

        # Register hooks for automatic cache invalidation
        if fit_par and isinstance(self.val, brainstate.State):
            self._cache_invalidation_hook_handle = self.val.register_hook(
                'write_after',
                self._on_param_state_write,
                priority=100,
                name='param_cache_invalidator'
            )

    def cache(self) -> None:
        """
        Manually cache the transformed value.

        This method forces immediate computation and caching of the transformed
        value, even if the cache is already valid. Useful for warming up the
        cache before performance-critical sections.

        Note
        ----
        The cache is automatically populated on first access to ``value()``.
        This method is only needed for explicit cache warming.

        Example
        -------
        >>> import jax.numpy as jnp
        >>> from brainstate.nn import ParaM, SoftplusT
        >>> param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT())
        >>> param.cache()  # Warm up cache before performance-critical code
        >>> val = param.value()  # Fast - returns cached value
        """
        # Get unconstrained value
        if isinstance(self.val, brainstate.State):
            val = self.val.value
        else:
            val = self.val
        with self._cache_lock:
            transformed = self.t.forward(val)
            self._cached_value = transformed
            self._cache_valid = True
            self._log_cache_event('manual_cache')
            return transformed

    def clear_cache(self) -> None:
        """
        Explicitly clear the parameter transformation cache.

        This method invalidates any cached transformed value, forcing the next
        call to ``value()`` to recompute the transformation. Thread-safe.

        Note
        ----
        Cache is automatically invalidated when the parameter is updated.
        This method is primarily useful for manual cache management or debugging.

        Example
        -------
        >>> import jax.numpy as jnp
        >>> from brainstate.nn import ParaM, SoftplusT
        >>> param = ParaM(jnp.array([1.0, 2.0]), t=SoftplusT())
        >>> _ = param.value()  # Computes and caches
        >>> param.clear_cache()  # Manual invalidation
        >>> _ = param.value()  # Recomputes
        """
        with self._cache_lock:
            if self._cache_valid:
                self._cache_valid = False
                self._cached_value = None
                self._log_cache_event('invalidate', reason='manual_clear')

    def value(self) -> Data:
        """
        Get current parameter value after applying transform.

        Returns cached value when valid. Otherwise, computes ``t.forward(val)``,
        caches it, and returns the result.

        Returns
        -------
        array_like
            Parameter value in the constrained space.
        """
        # Get unconstrained value
        if isinstance(self.val, brainstate.State):
            val = self.val.value
        else:
            val = self.val

        # Check cache
        with self._cache_lock:
            if self._cache_valid:
                self._log_cache_event('hit')
                return self._cached_value

            transformed = self.t.forward(val)
            return transformed

    def set_value(self, value: Data):
        """
        Set parameter value from constrained space.

        The value is transformed to unconstrained space for internal storage.
        Automatically invalidates cache.

        Parameters
        ----------
        value : array_like
            New value in the constrained space.
        """
        value = self.t.inverse(value)

        # Invalidate cache BEFORE writing
        with self._cache_lock:
            self._cache_valid = False
            self._cached_value = None
            self._log_cache_event('invalidate', reason='set_value')

        if isinstance(self.val, brainstate.State):
            self.val.value = value  # This will also trigger write_after hook
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

    @property
    def cache_stats(self) -> dict:
        """
        Get cache statistics (for debugging/monitoring).

        Returns
        -------
        dict
            Dictionary with keys: ``valid``, ``has_cached_value``

        Example
        -------
        >>> import jax.numpy as jnp
        >>> from brainstate.nn import ParaM, SoftplusT
        >>> param = ParaM(jnp.array([1.0]), t=SoftplusT())
        >>> param.cache_stats
        {'valid': False, 'has_cached_value': False}
        >>> _ = param.value()  # Compute and cache
        >>> param.cache_stats
        {'valid': True, 'has_cached_value': True}
        """
        with self._cache_lock:
            return {
                'valid': self._cache_valid,
                'has_cached_value': self._cached_value is not None
            }

    def _get_logger(self) -> logging.Logger:
        """Lazy logger initialization using ParaM name or ID."""
        if self._cache_logger is None:
            name = f'brainstate.nn.ParaM.{self._name or id(self)}'
            self._cache_logger = logging.getLogger(name)
        return self._cache_logger

    def _log_cache_event(self, event: str, **kwargs):
        """Log cache events (hit/miss/invalidate/error) if logging enabled."""
        if not self._enable_cache_logging:
            return

        logger = self._get_logger()

        if event == 'hit':
            logger.info(f"Cache HIT for ParaM '{self._name or id(self)}'")
        elif event == 'miss':
            logger.info(f"Cache MISS for ParaM '{self._name or id(self)}' - computing")
        elif event == 'invalidate':
            reason = kwargs.get('reason', 'unknown')
            logger.info(f"Cache INVALIDATED for ParaM '{self._name or id(self)}' (reason: {reason})")
        elif event == 'error':
            error = kwargs.get('error')
            logger.error(f"Cache ERROR for ParaM '{self._name or id(self)}': {error}", exc_info=True)

    def _on_param_state_write(self, ctx):
        """Invalidate cache when underlying ParamState is written."""
        with self._cache_lock:
            if self._cache_valid:
                self._cache_valid = False
                self._cached_value = None
                self._log_cache_event('invalidate', reason='state_write')

    def __pretty_repr_item__(self, name, value):
        if name in ('_enable_cache_logging',
                    '_cache_lock',
                    '_cached_value',
                    '_cache_valid',
                    '_cache_logger',
                    '_cache_invalidation_hook_handle'):
            return None
        if name.startswith('_'):
            return None if value is None else (name[1:], value)  # skip the first `_`
        return name, value


class ConstM(ParaM):
    """
    A module has non-trainable constant parameter.

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
