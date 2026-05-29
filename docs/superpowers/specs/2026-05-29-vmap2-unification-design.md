# Design Spec — Unified State-Aware Mapping Core (`vmap2` upgrade)

- **Date:** 2026-05-29
- **Branch:** `worktree-vmap2-unification`
- **Status:** Approved design; ready for implementation planning
- **Scope owner:** `brainstate.transform` mapping transforms

---

## 1. Background & motivation

`brainstate.transform` currently ships **two parallel, divergent** state-aware
mapping APIs:

- **`_mapping1.py`** — `vmap`, `vmap_new_states`. Built on the
  `StatefulFunction` / `make_jaxpr` engine (the same engine that powers
  `jit`, `grad`, `scan`, `cond`). States are selected **by instance** via
  explicit `in_states={axis: {name: State}}` / `out_states` dicts and threaded
  through `jax.vmap` as explicit value pytrees.
- **`_mapping2.py`** — `vmap2`, `pmap2`, `StatefulMapping`, `map`,
  `vmap2_new_states`, `pmap2_new_states`. Built on a bespoke
  **batch-trace-introspection** engine that manufactures `BatchTracer`s for
  states via JAX-private APIs (`make_iota`, `to_elt`, `trace_ctx`,
  `source_info_util`) and selects states **by filter/predicate**
  (`state_in_axes` / `state_out_axes`).

This duplication is confusing, and a deep review (below) found multiple
confirmed bugs and robustness risks in `_mapping2.py`.

### 1.1 Deep-review findings

**Confirmed bugs**

1. **Trailing-comma tuple bug** (`_mapping2.py:195-198`):
   `self.static_argnums = static_argnums,` (and `static_argnames`, `axis_env`,
   `return_only_write`) silently become 1-tuples, so the cache key at `:548`
   never honors static args/names.
2. **Empty error message** (`:416-418`): `raise BatchAxisError(f'')`.
3. **Cache key omits closed-over state shape** (`:548`): the key is built from
   positional `args` only. Two calls with the same arg shapes but a reshaped
   state reuse a stale trace → wrong results or a confusing crash. (The old
   `vmap` correctly folds state values into its key.)

**Robustness / correctness risks**

4. **No `kwargs` support** (`:381-384`, `:542-545`): raises `NotImplementedError`.
5. **Fragile stack-level coupling** in `vmap2_new_states` (`:1042-1043`): two
   hardcoded `decrease_stack_level()` calls keyed to literal trace names
   `'vmap2_eval'` / `'vmap2'`. `vmap_new_states` calls it **once** — the
   divergence is itself a smell and breaks if trace nesting changes.
6. **`None`-axis input states** mixed into `in_axes` (`:564`) is an
   under-tested path.
7. **Double tracing**: `vmap2` runs a full discovery `jax.vmap` trace *plus*
   the real mapping trace on a cold call.

**API / usability concerns**

8. **Two divergent state-selection models**: `vmap` is by-instance,
   `vmap2` is by-filter. `vmap2` cannot easily target a single `State`;
   `vmap` cannot say "all `ParamState`s."
9. **`pmap2` signature drift**: `axis_name` is 2nd-positional in `pmap2` but
   keyword-only in `vmap2`; `StatefulMapping` surfaces the broken JIT params.
10. **Docstring example for `vmap2` is wrong** (`:644-668`): a *scalar*
    `counter` batched on axis 0 with claimed output `Array([0.,1.,3.])` does
    not match the working test (which uses `jnp.zeros(3)`).

**Test coverage**: `_mapping2_test.py` has ~5 real tests (one is a bare
`print`); no coverage for `axis_size` inference, multi-axis states,
`unexpected_out_state_mapping` policies, `state_out_axes`, nested vmap,
`pmap2_new_states`, or the caching path.

---

## 2. Goals & non-goals

### Goals

- One canonical, hardened state-aware mapping engine.
- Fix all confirmed bugs (#1–#3) **structurally**, not by patching.
- Keep `vmap2` / `pmap2` as the primary documented API (filter-based).
- Re-implement `vmap` / `vmap_new_states` as **signature-preserving shims**
  over the unified core so all existing internal callers keep working.
- Comprehensive docs (NumPy-style, doctest-accurate) and tests (unit +
  integration, including real multi-device `pmap2`).

### Non-goals (this round)

- No renames; no deprecation warnings on the old `vmap` / `vmap_new_states`.
- No stateless fast-path (explicitly de-scoped).
- No per-`kwarg` axis mapping (kwargs are broadcast-only this round).

---

## 3. Decisions (locked)

| # | Decision |
|---|----------|
| D1 | **Converge** onto a single engine; old `vmap`/`vmap_new_states` become shims. |
| D2 | **Share core, keep both names working.** `vmap2`/`pmap2` stay primary; `vmap`/`vmap_new_states` keep exact signatures; no deprecation. |
| D3 | **Scope:** hardened `vmap2`/`StatefulMapping` core; `pmap2` + `pmap2_new_states`; `vmap2_new_states`; `map`; `vmap`/`vmap_new_states` shims. No stateless fast-path. |
| D4 | **Engine B core + auto-dim inference.** Re-found internals on `StatefulFunction`/`make_jaxpr` (explicit-value execution), and port automatic output-axis detection via read-only `BatchTracer.batch_dim` introspection. `StatefulMapping`/`vmap2`/`pmap2` remain public façades; the filter API is layered on top. |
| D5 | **State selection accepts both** `Filter`s and explicit `State` instances. |
| D6 | **`pmap2` tested for real** using `XLA_FLAGS='--xla_force_host_platform_device_count=8'`. |
| D7 | `unexpected_out_state_mapping` default is **`'auto'`** for `vmap2`/`pmap2`; the `vmap` shim forces **`'raise'`** for legacy parity. |
| D8 | **`kwargs` supported, broadcast-only** (unmapped) this round. |
| D9 | Engine split into a new `_mapping_core.py`; `StatefulMapping` stays thin. |

---

## 4. Architecture

### 4.1 Module layout

```
brainstate/transform/
  _mapping_core.py   (NEW)  engine: axis resolution, discovery pass,
                            execution pass, RNG handling, cache helpers
  _mapping2.py              public façades: StatefulMapping, vmap2, pmap2,
                            vmap2_new_states, pmap2_new_states, map
  _mapping1.py              vmap, vmap_new_states  -> thin shims into the core
  _mapping_core_test.py (NEW) engine unit tests
  _mapping2_test.py         rewritten public-API tests
  _mapping1_test.py         retained as the backward-compat regression gate
```

All JAX-private symbols are accessed **read-only** and only through
`brainstate._compatible_import`, which raises a clear, actionable error if a
future JAX version removes/renames them. The only private surface retained is
`BatchTracer` + `.batch_dim` (read-only, for auto-dim detection and error
messages) — the engine no longer *manufactures* tracers.

### 4.2 Execution model

`StatefulMapping.__call__(*args, **kwargs)` runs three steps:

**Step 1 — Cache key.**
Built via `StatefulFunction.get_arg_cache_key(state_vals, args, kwargs)`, which
folds the **abstract shapes of touched states** together with args/kwargs.
This makes the cache state-shape-aware by construction (fixes #3) and removes
the trailing-comma static-arg bug (#1).

**Step 2 — Discovery pass (cold-call only; cached).**

1. Enumerate touched states with a single `make_jaxpr` probe (no batching) →
   the `StateTraceStack` of read/written states.
2. Classify each state via `state_in_axes` (instance match → filter match →
   broadcast). Determine each batched-input state's input axis.
3. Run **one** `jax.vmap` over **explicit** state-value pytrees (the actual
   batched values; no tracer manufacturing). Inside the traced function, after
   calling `fn`, read each *written* state's leaf `.batch_dim` to record its
   **output** axis. This is where **auto-dim inference** lives: a written state
   that is batched but matched by no `state_out_axes` entry is auto-scattered
   at its detected dim (subject to `unexpected_out_state_mapping`, §4.5).
4. Cache `{state → input_axis}`, `{state → output_axis}`, batch size, and the
   RNG state list, keyed by the Step-1 cache key.

**Step 3 — Execution pass.**
`jax.vmap` (or `jax.pmap`) over
`(rng_keys, in_state_vmap_vals, in_state_oth_vals, args, kwargs)` with
`in_axes`/`out_axes` derived from the cached maps. The inner call goes through
`StatefulFunction` (consistent with `jit`/`grad`/`scan`). Written values come
back as explicit mapped outputs and are scattered back to the `State` objects;
"other" (broadcast) states are restored unmapped; RNG keys are restored
advanced (§4.4).

**Tracing cost.** Cold call = discovery + execution traces (the accepted cost
of keeping auto-dim inference). Warm call (same cache key) = execution only;
XLA compilation is cached by JAX. The discovery pass **always uses `jax.vmap`**
even for `pmap2`, because batch-dim detection is mapping-agnostic; only the
execution pass swaps in `jax.pmap`.

### 4.3 State selection (fixes #8)

`state_in_axes` / `state_out_axes` uniformly accept:

- a `Filter` / predicate (e.g. `OfType(ParamState)`),
- a single `State`, or a collection of `State`s (identity match),
- a `dict[axis → (Filter | State | collection)]`,
- a bare value → shorthand for `{0: value}`.

Per-state resolution order: **explicit instance match → filter match →**
(input) broadcast / (output) auto-dim-or-policy. A state assigned conflicting
axes (e.g. axis 0 on input, axis 1 on output without justification) raises a
`BatchAxisError` carrying the state's source info.

### 4.4 RNG semantics (consistent across all entry points)

Every `RandomState` touched during tracing is split into `batch_size` keys via
`split_key(batch_size)`, passed as an axis-0 mapped input, and the global key
is advanced exactly once per call. Behavior is identical across `vmap2`,
`pmap2`, and the `vmap` shim, and is documented. It remains **always-on / not
disableable** (unchanged contract). Use `brainstate.random`, never
`jax.random`, for any randomness inside the engine and tests.

### 4.5 `unexpected_out_state_mapping` policy

Values: `{'auto', 'raise', 'warn', 'ignore'}`. Applies to a written state that
is **batched** but matched by **no** `state_out_axes` entry and is **not** a
batched input:

- `'auto'` — **new default for `vmap2`/`pmap2`** — scatter at the detected
  `.batch_dim` (this is the auto-dim inference behavior).
- `'raise'` — `BatchAxisError` with a **non-empty, actionable** message (fixes
  #2). **Forced by the `vmap` shim** to preserve legacy declare-or-error
  semantics (keeps `test_vmap_batched_state_not_in_out_states` green).
- `'warn'` — `UserWarning`, then scatter at the detected dim.
- `'ignore'` — scatter at the detected dim silently.

### 4.6 `kwargs` (fixes #4)

`kwargs` are threaded through `StatefulFunction` (which already supports
`kwargs` + `static_argnames`). Default treatment is **broadcast** (in_axes
`None`). Per-kwarg axis mapping is out of scope this round and is documented as
a known limitation. This is a strict improvement over today's
`NotImplementedError`.

### 4.7 `vmap2_new_states` / `pmap2_new_states` robust unwinding (fixes #5)

Replace hardcoded `decrease_stack_level()` counts with a delta-based mechanism:

1. Capture the outer trace level `base = TRACE_CONTEXT.get_trace_stack_level()`
   **before** entering the transform.
2. After execution and value restoration, set each newly-created state's
   `stack_level` back to `base` (i.e. unwind exactly `state._level - base`
   levels) using the existing `stack_level` setter / `decrease_stack_level`.

A single shared helper backs both `vmap2_new_states` and the `vmap_new_states`
shim, eliminating the "1 vs 2" divergence. `state_out_axes` and the
`INIT_NO_BATCHING` / `NonBatchState` routing are preserved.

### 4.8 `map` hardening

- Build the `vmap2(f)` wrapper **once**, not per `scan` iteration.
- Keep the `batch_size` + remainder path; both the batched and the sequential
  fallback run through the unified state-aware core (via
  `brainstate.transform.scan`).
- Validate equal leading lengths across inputs with a clear `ValueError`.
- Document that `batch_size` trades peak memory for throughput.

### 4.9 `pmap2` cleanup (fixes #9)

- `axis_name` becomes keyword-only, consistent with `vmap2`.
- `StatefulMapping` no longer surfaces the broken JIT params
  (`static_argnums` / `static_argnames` / `axis_env` / `return_only_write`) as
  constructor noise; they are handled internally and correctly.
- `axis_size` defaults to the local device count for `pmap2`; a value greater
  than the available device count raises a clear error.

### 4.10 Backward-compat shims (D2)

`vmap(in_states=…, out_states=…)` and `vmap_new_states(...)` keep their exact
current signatures. They translate `in_states` / `out_states` dicts into the
core's instance-based `state_in_axes` / `state_out_axes` and force
`unexpected_out_state_mapping='raise'`. Internal callers that must keep working
unchanged:

- `brainstate/random/_impl.py` (`vmap(...)`)
- `brainstate/nn/_event_fixedprob.py` (`@transform.vmap(axis_size=n_pre)`)
- `brainstate/nn/_collective_ops.py` (`vmap_new_states(...)`)
- `brainstate/nn/_common.py` (`vmap2`, `vmap2_new_states`, `pmap2`,
  `pmap2_new_states`)

`_mapping1_test.py` (40 tests) is the compat regression gate and must pass
unchanged.

---

## 5. Edge cases to cover

- Negative / out-of-range axes (clear `IndexError`).
- `in_axes` as `int` / `tuple` / `list` / `None` / pytree-prefix.
- `axis_size`: inferred vs. explicit vs. conflicting (clear error).
- Empty positional args with `axis_size`-only.
- States whose `.value` is a pytree / dict.
- Mixed batched + broadcast states in one call.
- `None`-axis input states.
- Nested `vmap2(vmap2(...))` and `vmap` ∘ `vmap2`.
- `vmap2` over a function that touches **no** states.
- RNG advance correctness across repeated calls (no accidental key reuse).
- Cache reuse when a closed-over state changes shape between calls → must
  re-trace, never corrupt.
- All four `unexpected_out_state_mapping` values.
- `pmap2` with `axis_size` > device count → clear error.

---

## 6. Testing plan

- **`_mapping_core_test.py` (new):** axis resolution (filters + instances),
  discovery pass, cache key shape-awareness, RNG split/advance, robust
  stack-level unwinding.
- **`_mapping2_test.py` (rewritten):** remove the `print`-only test; cover each
  `unexpected_out_state_mapping` value, `state_out_axes`, multi-axis states,
  `axis_size` inference, `map` batch/remainder, `vmap2_new_states`.
- **Integration:** batch a small stateful `nn.Module` via `vmap2_new_states`
  and run it; assert `vmap` shim ≡ `vmap2` on the same workload; assert `vmap`
  ≡ `jax.vmap` numerically on stateless functions.
- **Multi-device (`pmap2` / `pmap2_new_states`):** set
  `os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'` at the
  **top of the test module, before any JAX import**, to run real multi-device
  execution on a single host.
- **Compat gate:** `_mapping1_test.py` unchanged + new nested-vmap and
  cache-invalidation tests.

All randomness in tests uses `brainstate.random`.

---

## 7. Documentation plan

- NumPy-style docstrings (canonical section order per project `CLAUDE.md`) for
  every public symbol: `StatefulMapping`, `vmap2`, `pmap2`,
  `vmap2_new_states`, `pmap2_new_states`, `map`, `vmap`, `vmap_new_states`.
- All `Examples` blocks are **doctest-accurate** and self-contained; the
  incorrect scalar-counter `vmap2` example (#10) is corrected.
- A short narrative section (module docstring / docs page) covering: the
  discovery → execution model, filter-vs-instance selection, RNG behavior, the
  `unexpected_out_state_mapping` policy, and a `vmap` → `vmap2` migration note.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Auto-dim inference still requires a discovery + execution trace on cold calls. | Accepted (explicit decision D4). Discovery is cached; warm calls are single-pass. |
| Read-only `BatchTracer.batch_dim` is still a JAX-private surface. | Centralize behind `_compatible_import` with a version-guard error; far smaller surface than the current `make_iota`/`to_elt`/tracer-manufacturing. |
| Shims must reproduce subtle legacy `vmap` semantics. | `_mapping1_test.py` (40 tests) is the unchanged regression gate; shim forces `'raise'`. |
| `pmap2` behavior differs across backends. | Real multi-device tests via `XLA_FLAGS`; shape/trace assertions as a floor. |
| Changing `_mapping2.py` internals could affect `nn._common` (`Map`). | `nn._common` exercised in integration tests; public signatures unchanged. |

---

## 9. Definition of done

- All confirmed bugs (#1–#3) fixed structurally; #4–#10 addressed.
- `vmap2`/`pmap2`/`StatefulMapping`/`map`/`*_new_states` hardened on the unified
  Engine-B core with auto-dim inference.
- `vmap`/`vmap_new_states` shims pass `_mapping1_test.py` unchanged.
- New + rewritten test suites pass, including real multi-device `pmap2`.
- All public symbols carry doctest-accurate NumPy-style docstrings.
- No direct `jax.random` usage introduced; all randomness via
  `brainstate.random`.
