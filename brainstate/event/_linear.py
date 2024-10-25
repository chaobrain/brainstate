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

from typing import Union, Callable, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate._state import ParamState, State
from brainstate._utils import set_module_as
from brainstate.init import param
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike
from ._misc import IntScalar

__all__ = [
  'Linear',
]


class Linear(Module):
  """
  The FixedProb module implements a fixed probability connection with CSR sparse data structure.

  Parameters
  ----------
  n_pre : int
      Number of pre-synaptic neurons.
  n_post : int
      Number of post-synaptic neurons.
  weight : float or callable or jax.Array or brainunit.Quantity
      Maximum synaptic conductance.
  name : str, optional
      Name of the module.
  """

  __module__ = 'brainstate.event'

  def __init__(
      self,
      n_pre: IntScalar,
      n_post: IntScalar,
      weight: Union[Callable, ArrayLike],
      name: Optional[str] = None,
      grad_mode: str = 'vjp'
  ):
    super().__init__(name=name)
    self.n_pre = n_pre
    self.n_post = n_post
    self.in_size = n_pre
    self.out_size = n_post

    assert grad_mode in ['vjp', 'jvp'], f"Unsupported grad_mode: {grad_mode}"
    self.grad_mode = grad_mode

    # maximum synaptic conductance
    weight = param(weight, (self.n_pre, self.n_post), allow_none=False)
    self.weight = ParamState(weight)

  def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
    weight = self.weight.value if isinstance(self.weight, State) else self.weight
    if u.math.size(weight) == 1:
      return u.math.ones(self.n_post) * (u.math.sum(spk) * weight)

    device_kind = jax.devices()[0].platform  # spk.device.device_kind
    if device_kind == 'cpu':
      return cpu_event_linear(u.math.asarray(weight),
                              u.math.asarray(spk),
                              n_post=self.n_post,
                              grad_mode=self.grad_mode)
    elif device_kind in ['gpu', 'tpu']:
      raise NotImplementedError()
    else:
      raise ValueError(f"Unsupported device: {device_kind}")


@set_module_as('brainstate.event')
def cpu_event_linear(
    g_max: Union[u.Quantity, jax.Array],
    spk: jax.Array,
    *,
    n_post: int = None,
    grad_mode: str = 'vjp'
) -> Union[u.Quantity, jax.Array]:
  """
  The FixedProb module implements a fixed probability connection with CSR sparse data structure.

  Parameters
  ----------
  n_post : int
      Number of post-synaptic neurons.
  g_max : brainunit.Quantity or jax.Array
      Maximum synaptic conductance.
  spk : jax.Array
      Spike events.
  grad_mode : str, optional
      Gradient mode. Default is 'vjp'. Can be 'vjp' or 'jvp'.

  Returns
  -------
  post_data : brainunit.Quantity or jax.Array
      Post synaptic data.
  """
  unit = u.get_unit(g_max)
  g_max = u.get_mantissa(g_max)
  spk = jnp.asarray(spk)

  def mv(spk_vector):
    assert spk_vector.ndim == 1, f"spk must be 1D. Got: {spk.ndim}"
    if jnp.size(g_max) == 1:
      assert isinstance(n_post, int), f"n_post must be an integer when weight is homogenous. Got: {n_post}"
      # return jnp.full((n_post,), fill_value=jnp.sum(spk_vector) * weight)
      return jnp.ones((n_post,), dtype=g_max.dtype) * (jnp.sum(spk_vector) * g_max)

    if grad_mode == 'vjp':
      post = _cpu_event_linear_mv_vjp(g_max, spk_vector)
    elif grad_mode == 'jvp':
      post = _cpu_event_linear_mv_jvp(g_max, spk_vector)
    else:
      raise ValueError(f"Unsupported grad_mode: {grad_mode}")
    return post

  assert spk.ndim >= 1, f"spk must be at least 1D. Got: {spk.ndim}"
  assert g_max.ndim in [2, 0], f"weight must be 2D or 0D. Got: {g_max.ndim}"

  if spk.ndim == 1:
    post_data = mv(spk)
  else:
    shape = spk.shape[:-1]
    post_data = jax.vmap(mv)(u.math.reshape(spk, (-1, spk.shape[-1])))
    post_data = u.math.reshape(post_data, shape + post_data.shape[-1:])
  return u.maybe_decimal(u.Quantity(post_data, unit=unit))


# --------------
# Implementation
# --------------


def _cpu_event_linear_mv(g_max, spk) -> jax.Array:
  def scan_fn(post, i):
    sp = spk[i]
    if spk.dtype == jnp.bool_:
      post = jax.lax.cond(sp, lambda: post + g_max[i], lambda: post)
    else:
      post = jax.lax.cond(sp == 0., lambda: post, lambda: post + g_max[i] * sp)
    return post, None

  return jax.lax.scan(scan_fn, jnp.zeros(g_max.shape[1], dtype=g_max.dtype), np.arange(len(spk)))[0]


# --------------
# VJP
# --------------

def _cpu_event_linear_mv_fwd(g_max, spk):
  return _cpu_event_linear_mv(g_max, spk), (g_max, spk)


def _cpu_event_linear_mv_bwd(res, ct):
  g_max, spk = res

  # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
  ct_spk = jnp.matmul(g_max, ct)

  # ∂L/∂w = ∂L/∂y * ∂y/∂w
  def map_fn(sp):
    if spk.dtype == jnp.bool_:
      d_gmax = jax.lax.cond(sp, lambda: ct, lambda: jnp.zeros_like(ct))
    else:
      d_gmax = jax.lax.cond(sp == 0., lambda: jnp.zeros_like(ct), lambda: ct * sp)
    return d_gmax

  ct_gmax = jax.vmap(map_fn)(spk)
  return ct_gmax, ct_spk


_cpu_event_linear_mv_vjp = jax.custom_vjp(_cpu_event_linear_mv)
_cpu_event_linear_mv_vjp.defvjp(_cpu_event_linear_mv_fwd, _cpu_event_linear_mv_bwd)


# --------------
# JVP
# --------------


def _cpu_event_linear_mv_jvp_rule(primals, tangents):
  # forward pass
  g_max, spk = primals
  y = _cpu_event_linear_mv(g_max, spk)

  # forward gradients
  gmax_dot, spk_dot = tangents

  # ∂y/∂gmax
  dgmax = _cpu_event_linear_mv(gmax_dot, spk)

  # ∂y/∂gspk
  dspk = spk_dot @ g_max
  return y, dgmax + dspk


_cpu_event_linear_mv_jvp = jax.custom_jvp(_cpu_event_linear_mv)
_cpu_event_linear_mv_jvp.defjvp(_cpu_event_linear_mv_jvp_rule)
