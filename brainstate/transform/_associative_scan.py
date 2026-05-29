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

from typing import Any, Callable

import jax

from brainstate._utils import set_module_as

__all__ = ['associative_scan', 'linear_recurrence']


@set_module_as("brainstate.transform")
def associative_scan(fn: Callable, elems: Any, reverse: bool = False, axis: int = 0):
    """Perform a parallel-prefix (associative) scan along an axis.

    A thin, unit- and pytree-aware surface over :func:`jax.lax.associative_scan`.
    The scan runs in ``O(log n)`` parallel depth (versus ``O(n)`` for a
    sequential :func:`~brainstate.transform.scan`), provided ``fn`` is
    associative. brainunit :class:`~brainunit.Quantity` leaves pass through
    transparently and retain their units.

    Parameters
    ----------
    fn : callable
        A binary associative combine function ``fn(a, b)`` where ``a`` and ``b``
        are pytrees matching the structure of one slice of ``elems`` along
        ``axis``. Must be associative for the result to be well-defined.
    elems : PyTree
        The input array(s) to scan. May be a single array, a pytree of arrays,
        or contain ``Quantity`` leaves. All leaves are scanned along ``axis``.
    reverse : bool, default False
        If ``True``, perform the scan from the end of ``axis`` to the start.
    axis : int, default 0
        The axis along which to scan.

    Returns
    -------
    PyTree
        The scanned result, with the same structure and leading shape as
        ``elems``.

    See Also
    --------
    linear_recurrence, scan

    Notes
    -----
    ``fn`` must be associative — ``fn(a, fn(b, c)) == fn(fn(a, b), c)`` — but
    need not be commutative. Addition, multiplication, and ``max`` are common
    choices.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> xs = jnp.arange(1.0, 6.0)
        >>> brainstate.transform.associative_scan(lambda a, b: a + b, xs)
        Array([ 1.,  3.,  6., 10., 15.], dtype=float32)
    """
    return jax.lax.associative_scan(fn, elems, reverse=reverse, axis=axis)


@set_module_as("brainstate.transform")
def linear_recurrence(decay: Any, drive: Any, *, reverse: bool = False, axis: int = 0):
    """Solve a first-order linear recurrence in parallel (log-depth).

    Computes ``h_t = decay_t * h_{t-1} + drive_t`` with ``h_0 = 0`` for every
    ``t`` along ``axis`` using an associative scan, giving ``O(log n)`` parallel
    depth instead of the ``O(n)`` of a sequential loop. This is the parallel
    engine behind linear state-space models (S4/Mamba-style), linear RNNs, and
    the linear/subthreshold part of neuron dynamics.

    Parameters
    ----------
    decay : array or Quantity
        Per-step multiplicative coefficients ``decay_t``, indexed along
        ``axis``. Must be dimensionless (it multiplies the running state).
    drive : array or Quantity
        Per-step additive input ``drive_t``, indexed along ``axis`` and
        broadcasting against ``decay``. May carry units; the output inherits
        them.
    reverse : bool, default False
        If ``True``, run the recurrence backwards along ``axis``.
    axis : int, default 0
        The time axis of ``decay`` and ``drive``.

    Returns
    -------
    array or Quantity
        The sequence ``h_t`` with the same shape as ``drive`` (and ``drive``'s
        units, if any).

    See Also
    --------
    associative_scan, scan

    Notes
    -----
    The associative combine operator is
    ``(a_l, b_l) • (a_r, b_r) = (a_l * a_r, b_r + a_r * b_l)``; the second
    component of the inclusive scan equals ``h_t``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> decay = jnp.array([0.9, 0.8, 0.7])
        >>> drive = jnp.array([1.0, 2.0, 3.0])
        >>> brainstate.transform.linear_recurrence(decay, drive)
        Array([1.  , 2.8 , 4.96], dtype=float32)
    """
    def _op(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, b_r + a_r * b_l

    _, h = jax.lax.associative_scan(_op, (decay, drive), reverse=reverse, axis=axis)
    return h
