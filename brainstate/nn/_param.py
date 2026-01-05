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
from typing import Optional, Union, Callable, Sequence, Tuple

import brainunit as u
import jax
import numpy as np

from brainstate import environ
from brainstate._state import ParamState, State, maybe_state
from brainstate.typing import ArrayLike
from ._module import Module
from ._regularization import Regularization
from ._transform import IdentityT, Transform

__all__ = [
    'Param',
    'Const',
]


class Param(Module):
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
    fit : bool, optional
        Whether the parameter is trainable. Default is ``True``.
    enable_cache_logging : bool, optional
        Whether to enable INFO-level logging for cache events. Default is ``False``.
        Logs cache hits, misses, invalidations, and errors for debugging.

    Attributes
    ----------
    fit : bool
        Whether the parameter is trainable.
    t : Transform
        The bijective transformation.
    reg : Regularization or None
        The regularization, if any.
    precompute : Callable or None
        Optional precompute function applied after transformation.
    val : array_like or ParamState
        The internal parameter storage.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import Param, SoftplusT, L2Reg
    >>> # Trainable positive parameter with L2 regularization
    >>> param = Param(
    ...     jnp.array([1.0, 2.0]),
    ...     t=SoftplusT(0.0),
    ...     reg=L2Reg(weight=0.01)
    ... )
    >>> param.value()  # Get constrained value
    >>> param.reg_loss()  # Get regularization loss

    >>> # Caching is automatic for all parameters
    >>> param = Param(
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
        value: ArrayLike,
        t: Transform = IdentityT(),
        reg: Optional[Regularization] = None,
        precompute: Optional[Callable] = None,
        fit: bool = True,
        enable_cache_logging: bool = False,
    ):
        super().__init__()

        self.fit = fit
        self.t = t
        self.reg = reg
        assert precompute is None or callable(precompute), 'precompute must be a callable function or None.'
        self.precompute = precompute

        # Initialize cache infrastructure (always enabled)
        self._enable_cache_logging = enable_cache_logging
        self._cache_lock = threading.RLock()
        self._cached_value: Optional[ArrayLike] = None
        self._cache_valid = False
        self._cache_logger: Optional[logging.Logger] = None
        self._cache_invalidation_hook_handle = None

        # Convert value to tensor
        val_tensor = u.math.asarray(value, dtype=environ.dftype())

        # Register reg as submodule if provided
        if not (reg is None or isinstance(reg, Regularization)):
            raise ValueError(
                'Regularization must be None or instance of '
                'Regularization.'
            )
        if not isinstance(t, Transform):
            raise TypeError(f't must be an instance of Transform. But got {type(t)}.')
        val_tensor = t.inverse(val_tensor)
        if fit:
            val_tensor = ParamState(val_tensor)
        self.val = val_tensor

        # Register hooks for automatic cache invalidation
        if fit and isinstance(self.val, State):
            self._cache_invalidation_hook_handle = self.val.register_hook(
                'write_after',
                self._on_param_state_write,
                priority=100,
                name='param_cache_invalidator'
            )

    def cache(self) -> ArrayLike:
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
        >>> from brainstate.nn import Param, SoftplusT
        >>> param = Param(jnp.array([1.0, 2.0]), t=SoftplusT())
        >>> param.cache()  # Warm up cache before performance-critical code
        >>> val = param.value()  # Fast - returns cached value
        """
        # Get unconstrained value
        if isinstance(self.val, State):
            val = self.val.value
        else:
            val = self.val
        with self._cache_lock:
            transformed = self.t.forward(val)
            if self.precompute is not None:
                transformed = self.precompute(transformed)
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
        >>> from brainstate.nn import Param, SoftplusT
        >>> param = Param(jnp.array([1.0, 2.0]), t=SoftplusT())
        >>> _ = param.value()  # Computes and caches
        >>> param.clear_cache()  # Manual invalidation
        >>> _ = param.value()  # Recomputes
        """
        with self._cache_lock:
            if self._cache_valid:
                self._cache_valid = False
                self._cached_value = None
                self._log_cache_event('invalidate', reason='manual_clear')

    def value(self) -> ArrayLike:
        """
        Get current parameter value after applying transform.

        Returns cached value when valid. Otherwise, computes ``t.forward(val)``,
        caches it, and returns the result.

        Returns
        -------
        array_like
            Parameter value in the constrained space.
        """

        # Check cache
        with self._cache_lock:
            if self._cache_valid:
                self._log_cache_event('hit')
                return self._cached_value

        # Get unconstrained value
        val = maybe_state(self.val)
        transformed = self.t.forward(val)
        if self.precompute is not None:
            transformed = self.precompute(transformed)
        return transformed

    def set_value(self, value: ArrayLike):
        """
        Set parameter value from constrained space.

        The value is transformed to unconstrained space for internal storage.
        Automatically invalidates cache.

        Parameters
        ----------
        value : array_like
            New value in the constrained space.
        """

        # Invalidate cache BEFORE writing
        with self._cache_lock:
            self._cache_valid = False
            self._cached_value = None
            self._log_cache_event('invalidate', reason='set_value')

        value = self.t.inverse(value)
        if isinstance(self.val, State):
            self.val.value = value  # This will also trigger write_after hook
        else:
            self.val = value

    def reg_loss(self) -> ArrayLike:
        """
        Calculate regularization loss.

        Returns
        -------
        array_like
            Regularization loss. Returns 0.0 for fixed parameters
            or parameters without regularization.
        """
        if not self.fit:
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
        >>> from brainstate.nn import Param, SoftplusT
        >>> param = Param(jnp.array([1.0]), t=SoftplusT())
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
        """Lazy logger initialization using Param name or ID."""
        if self._cache_logger is None:
            name = f'brainstate.nn.Param.{self._name or id(self)}'
            self._cache_logger = logging.getLogger(name)
        return self._cache_logger

    def _log_cache_event(self, event: str, **kwargs):
        """Log cache events (hit/miss/invalidate/error) if logging enabled."""
        if not self._enable_cache_logging:
            return

        logger = self._get_logger()

        if event == 'hit':
            logger.info(f"Cache HIT for Param '{self._name or id(self)}'")
        elif event == 'miss':
            logger.info(f"Cache MISS for Param '{self._name or id(self)}' - computing")
        elif event == 'invalidate':
            reason = kwargs.get('reason', 'unknown')
            logger.info(f"Cache INVALIDATED for Param '{self._name or id(self)}' (reason: {reason})")
        elif event == 'error':
            error = kwargs.get('error')
            logger.error(f"Cache ERROR for Param '{self._name or id(self)}': {error}", exc_info=True)

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
                    'precompute',
                    '_cache_invalidation_hook_handle'):
            return None
        if name.startswith('_'):
            return None if value is None else (name[1:], value)  # skip the first `_`
        return name, value

    @classmethod
    def init(
        cls,
        data: Union[Callable, ArrayLike, 'Param'],
        sizes: Union[int, Sequence[int]] = None,
        allow_none: bool = True,
        **param_kwargs,
    ) -> Union['Param', 'Const']:

        """
        Initialize parameters.

        Parameters
        ----------
        data: callable, ArrayLike, State
            The initialization of the parameter.

            - If it is None, the created parameter will be None.
            - If it is a callable function :math:`f`, the ``f(size)`` will be returned.
            - If it is an instance of :py:class:`init.Initializer``, the ``f(size)`` will be returned.
            - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
        sizes: int, sequence of int
            The shape of the parameter.
        allow_none: bool
            Whether allow the parameter is None.
        **param_kwargs
            Additional keyword arguments passed to the initialization.

        """
        # Check if the parameter is None
        if data is None:
            if allow_none:
                return None
            else:
                raise ValueError(
                    f'Expect a parameter with type of float, ArrayType, Initializer, or '
                    f'Callable function, but we got None. '
                )

        # Convert sizes to a tuple
        if sizes is not None:
            sizes = tuple(_to_size(sizes))

        if not isinstance(data, Param):
            # Check if the parameter is a callable function
            if callable(data):
                assert sizes is not None, (
                    'When the parameter is a callable function, the size must be provided.'
                )
                data = data(sizes, **param_kwargs)
            if u.math.isscalar(data):
                pass
            elif isinstance(data, (np.ndarray, jax.Array, u.Quantity, Param)):
                pass
            else:
                raise TypeError(f'Unknown parameter type: {type(data)}')
            data = Const(data)

        if sizes is not None:
            _check_shape(data.value(), sizes)
        return data


class Const(Param):
    """
    A module has non-trainable constant parameter.

    A convenience class that creates a fixed (non-trainable) parameter.
    Equivalent to ``ParamM(value, fit=False)``.

    Parameters
    ----------
    value : array_like
        The constant value.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainstate.nn import Const
    >>> const = Const(jnp.array([1.0, 2.0]))
    >>> const.value()
    """

    def __init__(self, value: ArrayLike, **param_kwargs):
        fit = param_kwargs.pop('fit', False)
        if fit:
            raise ValueError('Const parameters must be non-trainable (fit=False).')
        super().__init__(value, fit=False, **param_kwargs)


def _check_shape(init, sizes):
    # Check if the shape of the parameter matches the given size
    if not _are_broadcastable_shapes(u.math.shape(init), sizes):
        raise ValueError(
            f'The shape of the parameter {u.math.shape(init)} '
            f'does not match with the given size {sizes}'
        )


def _to_size(x) -> Optional[Tuple[int]]:
    if isinstance(x, (tuple, list)):
        return tuple(x)
    if isinstance(x, (int, np.integer)):
        return (x,)
    if x is None:
        return x
    raise ValueError(f'Cannot make a size for {x}')


def _are_broadcastable_shapes(shape1, shape2):
    """
    Check if two shapes are broadcastable.

    Parameters:
    - shape1: Tuple[int], the shape of the first array.
    - shape2: Tuple[int], the shape of the second array.

    Returns:
    - bool: True if shapes are broadcastable, False otherwise.
    """
    # Reverse the shapes to compare from the last dimension
    shape1_reversed = shape1[::-1]
    shape2_reversed = shape2[::-1]

    # Iterate over the dimensions of the shorter shape
    for dim1, dim2 in zip(shape1_reversed, shape2_reversed):
        # Check if the dimensions are not equal and neither is 1
        if dim1 != dim2 and 1 not in (dim1, dim2):
            return False

    # If all dimensions are compatible, the shapes are broadcastable
    return True


def _expand_params_to_match_sizes(params, sizes):
    """
    Expand the dimensions of params to match the dimensions of sizes.

    Parameters:
    - params: jax.Array or np.ndarray, the parameter array to be expanded.
    - sizes: tuple[int] or list[int], the target shape dimensions.

    Returns:
    - Expanded params with dimensions matching sizes.
    """
    params_dim = params.ndim
    sizes_dim = len(sizes)
    dim_diff = sizes_dim - params_dim

    # Add new axes to params if it has fewer dimensions than sizes
    for _ in range(dim_diff):
        params = u.math.expand_dims(params, axis=0)  # Add new axis at the last dimension
    return params
