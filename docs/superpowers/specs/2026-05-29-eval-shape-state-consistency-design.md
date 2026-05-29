# Design: Make `brainstate.transform.eval_shape` consistent with state-based transformations

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Branch:** `eval-shape-state-consistency`

## Problem

`brainstate.transform.eval_shape` is the only transformation in `brainstate/transform/`
that is **not** built on `StatefulFunction`. Instead it rolls its own state handling
with a raw `StateTraceStack(check_read=...)` callback plus the
`graph_to_tree` / `tree_to_graph` graph machinery.

The hand-rolled `check_read` callback rejects **any** pre-existing global `State` that
the traced function reads, raising a bare `ValueError('')`. As a result, the standard
stateful pattern that every other transform supports breaks:

```python
st = brainstate.State(jnp.zeros(3))

def f(x):
    st.value = st.value + x      # reads + writes an existing global State
    return st.value * 2.0

brainstate.transform.eval_shape(f, jnp.ones(3))   # -> ValueError('')
```

Meanwhile, the equivalent canonical path works fine:

```python
brainstate.transform.jit(f).eval_shape(jnp.ones(3))   # OK
```

### Root cause

The `check_read` callback conflates two distinct cases:

- **New** states/Nodes *constructed inside* `f` (the intended `lambda: LSTMCell(...)`
  abstract-init use case), and
- **Existing** global states that `f` merely *reads or writes* (the standard stateful
  pattern).

It rejects the second case, which is exactly what makes `eval_shape` inconsistent with
`jit`, `grad`, `vmap`, `scan`, and the rest — all of which trace existing states
transparently via `StatefulFunction`.

### What already works and must be preserved

`eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))` already returns a real `LSTMCell`
whose `ParamState` leaves are abstract (no memory allocated). This is the brainstate
equivalent of Flax NNX's `nnx.eval_shape` — lazy/abstract model construction, useful for
parameter inspection, memory budgeting, and sharding-aware initialization. This
capability is the main reason a brainstate-specific `eval_shape` exists (vs. plain
`jax.eval_shape`) and must keep working.

## Goals

1. Trace **existing** global states read/written by `f` without error — consistent with
   all other state-based transforms.
2. Preserve **new-state / Node** construction inside `f` (`lambda: LSTMCell(...)`),
   returning a reconstructed abstract Node.
3. Optionally return the abstract shapes of touched states.
4. Build on `StatefulFunction` + `catch_new_states` (the same primitives the other
   transforms use), eliminating the bespoke `StateTraceStack` callback.
5. Never execute `f` for real and never allocate concrete arrays or mutate state values.

## Non-goals

- No changes to `jax.eval_shape` semantics for plain array outputs.
- No new sharding API (sharding-aware init is a downstream *use case*, not built here).
- No changes to other transforms.

## Public API

```python
def eval_shape(
    f: Callable[..., A],
    *args: Any,
    return_state_shapes: bool = False,
    **kwargs: Any,
) -> A | tuple[dict, A]:
    ...
```

Computes abstract output shapes via JAX abstract evaluation, **without executing `f`,
allocating arrays, or mutating any `State.value`**.

### Return contract

- **Default (`return_state_shapes=False`)** — returns the abstract shape of `f`'s
  output only, matching `jax.eval_shape`'s single-value contract:
  - If `f` returns arrays/pytrees → a pytree with `jax.ShapeDtypeStruct` leaves.
  - If `f` returns a brainstate **Node** (e.g. `LSTMCell`) → a reconstructed Node of the
    same type whose `State` leaves hold `jax.ShapeDtypeStruct` values (abstract lazy
    init, no memory allocated).

- **`return_state_shapes=True`** — returns `(state_shapes, out_shapes)` where:
  - `state_shapes` is a `dict` mapping each touched `State` to the `ShapeDtypeStruct` of
    its value, and
  - `out_shapes` is the same value returned in the default case.

Abstract leaves are `jax.ShapeDtypeStruct` (not the lower-level `ShapedArray` the old
implementation produced), consistent with `jax.eval_shape` and `jit().eval_shape`.

## Mechanism (Approach B)

One abstract trace drives all three behaviors:

1. Wrap `f` in `StatefulFunction` for canonical read/write state tracking — identical to
   `jit` / `grad` / `vmap`. This removes the bespoke `StateTraceStack(check_read=...)`
   rejection and is what fixes the bug at its root.
2. Wrap the traced body in `catch_new_states` (the same primitive `vmap_new_states`
   uses) to capture states/Nodes *created inside* `f`.
3. Reconstruct a returned Node via `graph_to_tree` → abstract → `tree_to_graph`, so the
   result is a real (abstract) Node rather than a raw pytree.
4. Collect output shapes from the stateful function's out-shapes; collect touched-state
   shapes from its state trace for the optional `return_state_shapes` return.

### Considered alternatives

- **A — thin wrapper over `jit().eval_shape`.** Rejected: `jit` requires concrete state
  values up front and routes through `jaxpr_call`; it cannot construct new states/Nodes
  inside `f`, breaking the `lambda: LSTMCell(...)` case.
- **C — keep the hand-rolled `StateTraceStack`, just stop erroring.** Rejected: leaves
  `eval_shape` as the lone non-`StatefulFunction` transform, gives no state-shape
  return, and keeps a parallel code path that will drift again. Fixes the symptom, not
  the inconsistency.

## Edge cases (to be covered by tests)

1. Stateless function returning plain arrays → `ShapeDtypeStruct` pytree.
2. Function reading an existing global `State` → no error (the reported bug).
3. Function writing an existing global `State` → no error; `state_shapes` reflects it.
4. Function returning a Node (`LSTMCell`) → reconstructed abstract Node with
   `ShapeDtypeStruct` leaves.
5. `return_state_shapes=True` → correct shapes **and** dtypes for touched states.
6. Passing a `State` as a direct positional/keyword arg → same clear error other
   transforms raise (delegated to `StatefulFunction`'s input check), not a bare
   `ValueError('')`.
7. Nested pytree args and kwargs.
8. Non-array / static-style kwargs.
9. `brainstate.random` / `RandomState` usage inside `f`.
10. `State.value` is unchanged after the call (no mutation, no allocation).
11. Multiple states of mixed types (`ParamState`, `ShortTermState`, etc.).

## Backward compatibility

- The Node-return case keeps working; leaves change from `ShapedArray` to the
  higher-level `ShapeDtypeStruct`.
- The previously-broken existing-global-state case now works.
- The default return type is unchanged for stateless functions.
- No internal callers of `transform.eval_shape` exist; only `docs/apis/transform.rst`
  references it, which will be updated.

## Files affected

- `brainstate/transform/_eval_shape.py` — reimplementation.
- `brainstate/transform/_eval_shape_test.py` — new test module (does not exist today).
- `docs/apis/transform.rst` — documentation refresh if needed.
