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

# Modified from https://github.com/hennequin-lab/SOFO

from typing import Callable, Any, Tuple, Union, Sequence, Dict, Optional

import brainunit as u
import jax
import jax.numpy as jnp

from brainstate._state import State
from brainstate._utils import set_module_as
from brainstate.typing import SeedOrKey, Missing
from ._grad_transform import GradientTransform
from ._util import warp_grad_fn, tree_random_split

__all__ = [
    'sofo_grad',
]


def _batch_jvp(f, W, M, has_aux=False):
    _jvp = lambda s: jax.jvp(f, (W,), (s,), has_aux=has_aux)
    return jax.vmap(_jvp)(M)


def _batch_jvp_pair(f, W, M, has_aux=False):
    M_1, M_2 = M
    _jvp = lambda M_1, M_2: jax.jvp(f, W, (M_1, M_2), has_aux=has_aux)
    return jax.vmap(_jvp)(M_1, M_2)


def _ggn_ce(tangents, h):
    """
    Generalised Gauss-Newton (GGN) matrices for cross-entropy loss.

    Args:
        tangents (jnp.ndarray): Tangents associated with network output. size (k, batch_size, dim).
        h (jnp.ndarray): Predictions, usually probabilities of classes. size (dim,).

    Returns:
        jnp.ndarray: GGN matrix. size (k, k).
    """
    Jgh = (tangents @ h)[:, None]
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T  # (k, k)


def _ggn_mse(tangents):
    """
    Generalised Gauss-Newton (GGN) matrices for mean-squared loss.

    Args:
        tangents (jnp.ndarray): Tangents associated with network output. size (k, batch_size, dim).

    Returns:
        jnp.ndarray: GGN matrix. size (k, k).
    """
    return tangents @ tangents.T


def _sample_v(tangent_size, params, rng):
    """
    Samples a batch of random, normalized tangent vectors matching the structure of `params`.

    Each tangent vector is drawn from a standard normal distribution and normalized across
    the entire pytree (global L2 norm). The output is a pytree where each leaf has shape
    `(tangent_size, *x.shape)`.

    Args:
        tangent_size (int): The number of tangents/subspace dimension.
        params (PyTree): A pytree of parameters whose structure and shapes are used to sample tangents.
        rng (jax.Array): A JAX PRNG key.

    Returns:
        PyTree: A pytree with the same structure as `params`, where each leaf is a tensor of
                shape `(tangent_size, *leaf.shape)` representing a batch of normalized tangent vectors.
    """
    v = jax.tree.map(
        lambda x, k: jax.random.normal(k, (tangent_size,) + x.shape, x.dtype),
        params,
        tree_random_split(rng, params)
    )
    # Normalize, tangent-wise
    l2 = jnp.sqrt(sum(jax.tree.leaves(jax.vmap(lambda v: jax.tree.map(lambda x: jnp.sum(jnp.square(x)), v))(v))))
    v = jax.tree.map(lambda x: jax.vmap(lambda a, b: a / b)(x, l2), v)
    return v


def _sofo_grad_impl(
    fn: Callable,
    loss_fn: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    return_loss: bool = False,
    tangent_size: int = 100,
    damping: float = 1E-5,
    loss: str = 'mse',
    key: SeedOrKey = None,
) -> Callable:
    """
    SOFO forward pass to compute loss and gradient.

    Args:
        fn (Callable): Forward pass of the network. ``fn`` s answer should be concatenation
            of function on a batch of samples with mean function over the same batch.
        tangent_size (int, optional): Number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): Dampling parameter on ggn. Defaults to 1e-5.
        loss (str, optional): Loss function. Defaults to 'mse'. Options are 'mse' and 'ce'.
        key (SeedOrKey, optional): Random key. Defaults to None.
        argnums (int, sequence of int, optional): Argument numbers to differentiate with respect to. Defaults to 0.
        has_aux (bool, optional): Whether the function ``fn`` returns auxiliary data. Defaults to False.
    """
    from brainstate.random._seed import split_key

    def wrapper(*args, **kwargs):
        f_partial, params = warp_grad_fn(fn, argnums, args, kwargs)
        v = _sample_v(tangent_size, params, split_key() if key is None else key)

        # tangents_out shape: t_size, b_size, out_size
        res = _batch_jvp(f_partial, params, v, has_aux=has_aux)
        if has_aux:
            outs, tangents_out, aux = res
            aux = jax.tree.map(lambda x: x[0], aux)
        else:
            outs, tangents_out = res
        losses, vg = _batch_jvp(loss_fn, outs[0], tangents_out)

        if loss == 'mse':
            vg_gv = u.math.mean(jax.vmap(_ggn_mse, in_axes=1)(tangents_out), axis=0)
        elif loss == 'ce':
            vg_gv = u.math.mean(
                jax.vmap(_ggn_ce, in_axes=(1, 0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0
            )
        else:
            raise ValueError(f'Unknown loss function: {loss}.')

        u_, s_, _ = jnp.linalg.svd(vg_gv)
        damped_s = s_ + damping * jnp.max(s_)

        vggv_vg = (u_ / damped_s) @ (u_.T @ vg)
        h = jax.tree.map(lambda v_: jnp.einsum('i,i...->...', vggv_vg, v_), v)
        if return_loss:
            return ((h, losses[0]), aux) if has_aux else (h, losses[0])
        else:
            return (h, aux) if has_aux else h

    return wrapper


@set_module_as("brainstate.transform")
def sofo_grad(
    fun: Callable,
    loss_fn: Callable,
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
    check_states: bool = True,
    loss: str = 'mse',
    tangent_size: int = 100,
    damping: float = 1E-5,
    key: SeedOrKey = None,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """
    Second-order forward-mode optimization to compute loss and gradient.

    1. When ``grad_states`` is None

        - ``has_aux=False`` + ``return_loss=False`` => ``arg_grads``.
        - ``has_aux=True`` + ``return_loss=False`` => ``(arg_grads, aux_data)``.
        - ``has_aux=False`` + ``return_loss=True`` => ``(arg_grads, fn_value)``.
        - ``has_aux=True`` + ``return_loss=True`` => ``(arg_grads, fn_value, aux_data)``.
    2. When ``grad_states`` is not None and ``argnums`` is None

        - ``has_aux=False`` + ``return_loss=False`` => ``var_grads``.
        - ``has_aux=True`` + ``return_loss=False`` => ``(var_grads, aux_data)``.
        - ``has_aux=False`` + ``return_loss=True`` => ``(var_grads, fn_value)``.
        - ``has_aux=True`` + ``return_loss=True`` => ``(var_grads, fn_value, aux_data)``.
    3. When ``grad_states`` is not None and ``argnums`` is not None

        - ``has_aux=False`` + ``return_loss=False`` => ``(var_grads, arg_grads)``.
        - ``has_aux=True`` + ``return_loss=False`` => ``((var_grads, arg_grads), aux_data)``.
        - ``has_aux=False`` + ``return_loss=True`` => ``((var_grads, arg_grads), fn_value)``.
        - ``has_aux=True`` + ``return_loss=True`` => ``((var_grads, arg_grads), fn_value, aux_data)``.


    Parameters
    ----------
    fun : callable, optional
        The scalar-valued function to be differentiated.
    grad_states : State, sequence of State, or dict of State, optional
        The variables in fun to take their gradients.
    argnums : int or sequence of int, optional
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux : bool, optional
        Indicates whether fun returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.
    return_value : bool, default False
        Indicates whether to return the value of the
        function along with the gradient.
    check_states : bool, default True
        Whether to check that all grad_states are found in the function.
    loss: str, default 'mse'
        Loss function to use. Supported values are 'mse' and 'ce'.

    Returns
    -------
    GradientTransform or callable
        A function which computes the gradient of fun. The function takes the same
        arguments as `fun`, but returns the gradient instead. If `has_aux` is True,
        the function returns a pair where the first element is the gradient and the
        second element is the auxiliary data. If `return_loss` is True, the function
        returns a pair where the first element is the gradient and the second element
        is the value of the function.

    """
    return GradientTransform(
        target=fun,
        transform=_sofo_grad_impl,
        grad_states=grad_states,
        argnums=argnums,
        return_value=return_value,
        has_aux=(False if has_aux is None else has_aux),
        check_states=check_states,
        transform_params=dict(
            loss=loss,
            tangent_size=tangent_size,
            damping=damping,
            loss_fn=loss_fn,
            key=key,
        )
    )


def _sofo_grad_scan_impl(
    fn: Callable,
    loss_fn: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    return_value: bool = False,
    tangent_size: int = 100,
    damping: float = 1E-5,
    loss: str = 'mse',
    key: SeedOrKey = None
) -> Callable[..., Tuple[Any, Any]]:
    """
    SOFO forward pass to compute loss and gradient.

    Args:
        rnn (Callable): One-step update of the recurrent network. ``rnn`` s answer should be concatenation
            of function on a batch of samples with mean function over the same batch.
        tangent_size (int, optional): Number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): Dampling parameter on ggn. Defaults to 1e-5.
        loss (str, optional): Loss function. Defaults to 'mse'. Options are 'mse' and 'ce'.
        key (SeedOrKey, optional): Random key. Defaults to None.
    """
    key = split_key() if key is None else key
    argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    def wrapper(params, z_init, batch):
        v = _sample_v(tangent_size, params, key)

        def fn(carry, xs):
            latent, latent_tangents, losses, vg, vggv = carry
            inputs, labels = xs

            fn2jvp = lambda params, latent: rnn(params, latent, inputs)
            latent_new, latent_tangents_out, outs = _batch_jvp_pair(
                fn2jvp,
                (params, latent),
                (v, latent_tangents),
                has_aux=True,
            )
            [latent_primal, primal_out] = latent_new
            [new_latent_tangents_out, tangents_out] = latent_tangents_out

            if loss == 'mse':
                loss_fn = lambda logits: jnp.mean(jnp.square(logits), axis=0)
            elif loss == 'ce':
                loss_fn = lambda logits: jnp.mean(
                    jax.nn.softmax(logits, axis=-1) * jnp.log(jax.nn.softmax(logits, axis=-1)), axis=0
                )
            else:
                raise ValueError(f'Unknown loss function: {loss}.')
            losses_new, vg_new = _batch_jvp(loss_fn, primal_out[0], tangents_out)
            losses += losses_new[0]
            vg += vg_new

            if loss == 'mse':
                vggv_new = jnp.mean(jax.vmap(_ggn_mse, in_axes=1)(tangents_out), axis=0)
            elif loss == 'ce':
                vggv_new = jnp.mean(
                    jax.vmap(_ggn_ce, in_axes=(1, 0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0
                )
            else:
                raise ValueError(f'Unknown loss function: {loss}.')
            vggv += vggv_new
            return (latent_primal[0], new_latent_tangents_out, losses, vg, vggv), outs[0]

        (_, _, losses, vg, vggv), preds = jax.lax.scan(
            fn,
            init=(
                z_init,
                jnp.zeros((tangent_size, *z_init.shape)),
                0.,
                jnp.zeros((tangent_size,)),
                jnp.zeros((tangent_size, tangent_size)),
            ),
            xs=batch
        )

        u_, s_, _ = jnp.linalg.svd(vggv)
        damped_s = s_ + damping * jnp.max(s_)

        vggv_vg = (u_ / damped_s) @ (u_.T @ vg)
        h = jax.tree.map(lambda v_: jnp.einsum('i,i...->...', vggv_vg, v_), v)
        # return losses, h, preds
        if return_value:
            return (h, losses[0]) if has_aux else (h, losses[0])
        else:
            return h if has_aux else h

    return wrapper


@set_module_as("brainstate.transform")
def sofo_grad_scan(
    fun: Callable = Missing(),
    grad_states: Optional[Union[State, Sequence[State], Dict[str, State]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
    check_states: bool = True,
) -> GradientTransform | Callable[[Callable], GradientTransform]:
    """
    Compute the gradient of a scalar-valued function with respect to its arguments.

    1. When ``grad_states`` is None

        - ``has_aux=False`` + ``return_value=False`` => ``arg_grads``.
        - ``has_aux=True`` + ``return_value=False`` => ``(arg_grads, aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``(arg_grads, fn_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``(arg_grads, fn_value, aux_data)``.
    2. When ``grad_states`` is not None and ``argnums`` is None

        - ``has_aux=False`` + ``return_value=False`` => ``var_grads``.
        - ``has_aux=True`` + ``return_value=False`` => ``(var_grads, aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``(var_grads, fn_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``(var_grads, fn_value, aux_data)``.
    3. When ``grad_states`` is not None and ``argnums`` is not None

        - ``has_aux=False`` + ``return_value=False`` => ``(var_grads, arg_grads)``.
        - ``has_aux=True`` + ``return_value=False`` => ``((var_grads, arg_grads), aux_data)``.
        - ``has_aux=False`` + ``return_value=True`` => ``((var_grads, arg_grads), fn_value)``.
        - ``has_aux=True`` + ``return_value=True`` => ``((var_grads, arg_grads), fn_value, aux_data)``.


    Parameters
    ----------
    fun : callable, optional
        The scalar-valued function to be differentiated.
    grad_states : State, sequence of State, or dict of State, optional
        The variables in fun to take their gradients.
    argnums : int or sequence of int, optional
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux : bool, optional
        Indicates whether fun returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.
    return_value : bool, default False
        Indicates whether to return the value of the
        function along with the gradient.
    check_states : bool, default True
        Whether to check that all grad_states are found in the function.

    Returns
    -------
    GradientTransform or callable
        A function which computes the gradient of fun. The function takes the same
        arguments as `fun`, but returns the gradient instead. If `has_aux` is True,
        the function returns a pair where the first element is the gradient and the
        second element is the auxiliary data. If `return_value` is True, the function
        returns a pair where the first element is the gradient and the second element
        is the value of the function.

    """
    return GradientTransform(
        target=fun,
        transform=_sofo_grad_scan_impl,
        grad_states=grad_states,
        argnums=argnums,
        return_value=return_value,
        has_aux=False if has_aux is None else has_aux,
        check_states=check_states
    )
