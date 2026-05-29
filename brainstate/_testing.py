# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Shared, importable test helpers for the brainstate test suite.

This module is private (underscore-prefixed) and excluded from coverage.  It is
importable by the co-located ``*_test.py`` files, which are ``unittest.TestCase``
subclasses and therefore cannot consume pytest fixtures.

See ``CONTRIBUTING.md`` (Testing conventions) for usage guidance.
"""

from __future__ import annotations

import brainunit as u
import jax
import jax.numpy as jnp

import brainstate

__all__ = [
    "SMALL_BATCH",
    "SMALL_DIM",
    "SMALL_SEQ",
    "DEFAULT_RTOL",
    "DEFAULT_ATOL",
    "assert_allclose",
    "assert_jit_equal",
    "assert_grad_finite",
    "assert_vmap_equal",
    "assert_transform_compatible",
    "assert_pytree_roundtrip",
    "seeded",
]

# Keep test fixtures tiny (see CONTRIBUTING.md > Testing conventions > Performance).
SMALL_BATCH = 4
SMALL_DIM = 16
SMALL_SEQ = 5

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6

# Re-export brainstate's seed context manager under a discoverable name.
# Usage: ``with seeded(0): ...`` — temporarily seeds and restores RNG state.
seeded = brainstate.random.seed_context


def assert_allclose(actual, expected, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, check_dtype=False):
    """Assert two (possibly unit-carrying) arrays match in shape and value.

    Parameters
    ----------
    actual, expected : array_like or brainunit.Quantity
        Values to compare.  Units are compared via ``brainunit``.
    rtol, atol : float
        Relative and absolute tolerances.
    check_dtype : bool
        When ``True``, also assert the underlying magnitudes share a dtype.
    """
    actual_shape = u.math.shape(actual)
    expected_shape = u.math.shape(expected)
    assert actual_shape == expected_shape, (
        f"shape mismatch: {actual_shape} != {expected_shape}"
    )
    # Dimensions must match (mV vs V is fine — same dimension; mV vs mA is not).
    actual_dim = u.get_dim(actual)
    expected_dim = u.get_dim(expected)
    assert actual_dim == expected_dim, (
        f"unit mismatch: {u.get_unit(actual)} vs {u.get_unit(expected)}"
    )
    if check_dtype:
        a_dtype = u.get_magnitude(actual).dtype
        e_dtype = u.get_magnitude(expected).dtype
        assert a_dtype == e_dtype, f"dtype mismatch: {a_dtype} != {e_dtype}"
    # Express both magnitudes in expected's unit so the dimensionless rtol/atol
    # apply to plain numbers (saiunit rejects a dimensionless atol compared
    # against a unit-carrying quantity).
    ref_unit = u.get_unit(expected)
    actual_mag = u.Quantity(actual).to_decimal(ref_unit)
    expected_mag = u.Quantity(expected).to_decimal(ref_unit)
    assert bool(jnp.allclose(actual_mag, expected_mag, rtol=rtol, atol=atol)), (
        f"value mismatch beyond rtol={rtol}, atol={atol}"
    )


def _assert_tree_allclose(tree_a, tree_b, *, rtol, atol):
    """Leaf-wise ``assert_allclose`` over two matching pytrees."""
    jax.tree.map(
        lambda a, b: assert_allclose(a, b, rtol=rtol, atol=atol),
        tree_a,
        tree_b,
    )


def assert_jit_equal(fn, *args, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, **kwargs):
    """Assert ``brainstate.transform.jit(fn)`` matches eager ``fn`` leaf-wise."""
    eager = fn(*args, **kwargs)
    jitted = brainstate.transform.jit(fn)(*args, **kwargs)
    _assert_tree_allclose(eager, jitted, rtol=rtol, atol=atol)
    return jitted


def assert_grad_finite(fn, *args, argnums=0, **kwargs):
    """Assert ``brainstate.transform.grad(fn)`` yields all-finite gradients.

    ``fn`` must return a scalar.
    """
    grads = brainstate.transform.grad(fn, argnums=argnums)(*args, **kwargs)
    leaves = jax.tree.leaves(grads)
    assert leaves, "no gradient leaves were produced"
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(u.get_magnitude(g)))), "non-finite gradient leaf"
    return grads


def assert_vmap_equal(fn, *batched_args, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Assert ``vmap(fn)`` over axis 0 matches an explicit Python loop.

    Every positional argument is mapped over its leading axis (``in_axes=0``).
    """
    vmapped = brainstate.transform.vmap(fn, in_axes=0)(*batched_args)
    n = u.math.shape(batched_args[0])[0]
    looped = [fn(*[jnp.take(a, i, axis=0) for a in batched_args]) for i in range(n)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *looped)
    _assert_tree_allclose(vmapped, stacked, rtol=rtol, atol=atol)
    return vmapped


def assert_transform_compatible(
    fn,
    *args,
    transforms=("jit",),
    grad_argnums=0,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    **kwargs,
):
    """Run ``fn`` under each requested JAX transform and assert agreement.

    Parameters
    ----------
    transforms : sequence of {"jit", "grad", "vmap"}
        Which transforms to check.  ``"jit"`` only by default.  ``"grad"``
        requires ``fn`` to return a scalar; ``"vmap"`` maps every arg over axis 0.
    """
    if "jit" in transforms:
        assert_jit_equal(fn, *args, rtol=rtol, atol=atol, **kwargs)
    if "grad" in transforms:
        assert_grad_finite(fn, *args, argnums=grad_argnums, **kwargs)
    if "vmap" in transforms:
        assert_vmap_equal(fn, *args, rtol=rtol, atol=atol)
    return True


def assert_pytree_roundtrip(obj, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Assert ``obj`` survives a flatten/unflatten roundtrip structurally and by value."""
    leaves, treedef = jax.tree.flatten(obj)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert jax.tree.structure(obj) == jax.tree.structure(rebuilt), "pytree structure changed"
    _assert_tree_allclose(obj, rebuilt, rtol=rtol, atol=atol)
    return rebuilt
