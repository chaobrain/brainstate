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


from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp


def batch_jvp(f, W, M, has_aux=False):
    _jvp = lambda s: jax.jvp(f, (W,), (s,), has_aux=has_aux)
    return jax.vmap(_jvp)(M)


def batch_jvp_pair(f, W, M, has_aux=False):
    M_1, M_2 = M
    _jvp = lambda M_1, M_2: jax.jvp(f, W, (M_1, M_2), has_aux=has_aux)
    return jax.vmap(_jvp)(M_1, M_2)


def ggn_ce(tangents, h):
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


def ggn_mse(tangents):
    """
    Generalised Gauss-Newton (GGN) matrices for mean-squared loss.

    Args:
        tangents (jnp.ndarray): Tangents associated with network output. size (k, batch_size, dim).

    Returns:
        jnp.ndarray: GGN matrix. size (k, k).
    """
    return tangents @ tangents.T


def random_split_like_tree(rng_key, target=None, treedef=None):
    """
    Split key for a key for every leaf.

    Args:
        rng_key (jax.Array): A JAX PRNG key.
        target (PyTree, optional): A pytree to infer the tree structure from.
                                   Required if `treedef` is not provided.
        treedef (TreeDef, optional): An explicit tree structure. If provided, `target` is ignored.

    Returns:
        PyTree: A pytree of PRNG keys with the same structure as `target` or `treedef`.
    """
    if treedef is None:
        treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def sample_v(tangent_size, params, rng):
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
        random_split_like_tree(rng, params)
    )
    # Normalize, tangent-wise
    l2 = jnp.sqrt(sum(jax.tree.leaves(jax.vmap(lambda v: jax.tree.map(lambda x: jnp.sum(jnp.square(x)), v))(v))))
    v = jax.tree.map(lambda x: jax.vmap(lambda a, b: a / b)(x, l2), v)
    return v


def sofo_grad(
    fn: Callable,
    tangent_size: int = 100,
    damping: float = 1E-5,
    loss: str = 'mse',
) -> Callable[..., Tuple[Any, Any]]:
    """
    SOFO forward pass to compute loss and gradient.

    Args:
        fn (Callable): Forward pass of the network. ``fun`` s answer should be concatenation
            of function on a batch of samples with mean function over the same batch.
        tangent_size (int, optional): Number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): Dampling parameter on ggn. Defaults to 1e-5.
        loss (str, optional): Loss function. Defaults to 'mse'. Options are 'mse' and 'ce'.
    """

    def value_and_fish_grad_f(rng, params):
        v = sample_v(tangent_size, params, jax.random.split(rng)[1])

        outs, tangents_out = batch_jvp(fn, params, v)  # tangents_out shape: t_size, b_size, out_size
        if loss == 'mse':
            loss_fn = lambda logits: jnp.mean(jnp.square(logits), axis=0)
        elif loss == 'ce':
            loss_fn = lambda logits: jnp.mean(
                jax.nn.softmax(logits, axis=-1) * jnp.log(jax.nn.softmax(logits, axis=-1)),
                axis=0
            )
        else:
            raise ValueError(f'Unknown loss function: {loss}.')
        losses, vg = batch_jvp(loss_fn, outs[0], tangents_out)

        if loss == 'mse':
            vg_gv = jnp.mean(jax.vmap(ggn_mse, in_axes=1)(tangents_out), axis=0)
        elif loss == 'ce':
            vg_gv = jnp.mean(jax.vmap(ggn_ce, in_axes=(1, 0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0)
        else:
            raise ValueError(f'Unknown loss function: {loss}.')

        u, s, _ = jnp.linalg.svd(vg_gv)
        damped_s = s + damping * jnp.max(s)

        vggv_vg = (u / damped_s) @ (u.T @ vg)
        h = jax.tree.map(lambda v_: jnp.dot(jnp.moveaxis(v_, 0, -1), vggv_vg), v)
        return losses[0], h, jnp.max(s)

    return value_and_fish_grad_f


def sofo_grad_scan(
    rnn: Callable,
    tangent_size: int = 100,
    damping: float = 1E-5,
    loss: str = 'mse',
) -> Callable[..., Tuple[Any, Any]]:
    """
    SOFO forward pass to compute loss and gradient.

    Args:
        rnn (Callable): One-step update of the recurrent network. ``rnn`` s answer should be concatenation
            of function on a batch of samples with mean function over the same batch.
        tangent_size (int, optional): Number of tangets/subspace dimension. Defaults to 100.
        damping (float, optional): Dampling parameter on ggn. Defaults to 1e-5.
        loss (str, optional): Loss function. Defaults to 'mse'. Options are 'mse' and 'ce'.
    """

    def wrapper(rng, params, z_init, batch):
        v = sample_v(tangent_size, params, jax.random.split(rng)[1])

        def fn(carry, xs):
            latent, latent_tangents, losses, vg, vggv = carry
            inputs, labels = xs

            fn2jvp = lambda params, latent: rnn(params, latent, inputs)
            latent_new, latent_tangents_out, outs = batch_jvp_pair(
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
            losses_new, vg_new = batch_jvp(loss_fn, primal_out[0], tangents_out)
            losses += losses_new[0]
            vg += vg_new

            if loss == 'mse':
                vggv_new = jnp.mean(jax.vmap(ggn_mse, in_axes=1)(tangents_out), axis=0)
            elif loss == 'ce':
                vggv_new = jnp.mean(
                    jax.vmap(ggn_ce, in_axes=(1, 0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0
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

        u, s, _ = jnp.linalg.svd(vggv)
        damped_s = s + damping * jnp.max(s)

        vggv_vg = (u / damped_s) @ (u.T @ vg)
        h = jax.tree.map(lambda vs: jnp.dot(jnp.moveaxis(vs, 0, -1), vggv_vg), v)
        return losses, h, preds

    return wrapper
