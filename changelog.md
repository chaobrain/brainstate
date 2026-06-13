# Release Notes


## Version 0.5.0 (2026-06-14)

A repository-wide correctness release. Following the `brainstate.transform` audit that shipped in 0.4.x, this cycle extended the same expert-audit discipline to nearly every remaining module — `random`, `graph`, `interop`, `nn`, `util`, the `vmap` / `pmap` / `shard_map` mapping engine, and the `exp_euler` integrator — and then closed out a single consolidated cross-module audit (`dev/issues.md`) covering one critical, twenty-one high, forty-six medium, and twenty-nine low findings, plus a long appendix of unverified items. Every fix ships with a behavioral regression test (several previously-skipped "known bug" tests are now un-skipped and passing), and the suite is green across the full CI JAX matrix (0.7.0, 0.8.0, 0.9.0, and latest). The release also lands a graph-layer performance pass. No public APIs are removed or renamed; the only behavioral changes are previously-silent wrong-result or invalid-input paths that now fail loudly with descriptive errors.

### Performance

- **Graph flatten/unflatten fast paths** (#218): value classification is now memoized in a type-keyed cache that backs the node predicates, the encoder dispatch, and the flattening kernel (which classifies once and iterates node items directly); the decoder uses exact-type dispatch; all-static hashable pytrees collapse to a single `StaticEdge`; and `graph_to_tree` reads its `State`s directly from the `RefMap`. Shared `State`s are de-duplicated in `iter_leaf` / `states` to match `treefy_states`.

### Bug Fixes

#### `brainstate.random` (#211)

Six reachable distribution bugs, each contradicting the function's own NumPy-style docstring:

- **`standard_t`** with an array `df` and `size=None` returned shape `()` (and raised `ValueError`) via a dead `shape(size)` branch; it now infers the shape from `df`, matching the sibling `t`.
- **`weibull_min`** divided by `scale` instead of multiplying; it is now `r * scale`, matching `scipy.stats.weibull_min` and the `weibull` scale convention.
- **`triangular`** was a Rademacher `2*bernoulli-1` (±1) draw with a size-only signature, so the documented `triangular(-3, 0, 8, N)` raised `TypeError`. It is reimplemented as the true `triangular(left, mode, right, size)` via inverse-CDF, with shared-unit support like `uniform`.
- **`geometric`** was off-by-one (support `{0,1,...}` instead of `{1,2,...}`) and returned a float; it now returns an integer dtype with `P(k==1) == p`.
- **`randint_like`** computed its default `high = max(input)` with the Python builtin, raising on templates with more than one dimension; it now uses `u.math.max`.
- **`chisquare`** summed `df` squared normals, rejecting non-integer scalar `df` and array `df` with `size=None`; it now uses the `2 · Gamma(df/2)` relation, valid for any positive real or array `df`.

#### `brainstate.graph` (#212)

- **`merge_context`** yielded `dict(index_ref)` — an empty snapshot disconnected from the table that `treefy_merge` populates. It now yields the live dict, symmetric with `split_context`.
- **`Node.check_valid_context`** read `self._trace_state`, which graph nodes never carry, raising `AttributeError` for every graph node (e.g. `nn.Linear`). A node's validity is now computed as the conjunction of the trace validity of the `State`s reachable from it.
- **`pop_states`** deduplicated a matched `State` by identity and popped only its first reference, leaving later shared/tied aliases dangling on the node. Every alias of a popped state is now detached while the state is still recorded once.

#### `brainstate.interop` (#213)

- Added the missing `input_dilation` guard to the `nnx` `Conv` import (previously silent data corruption).
- Norm-channel extraction now reads framework metadata instead of affine parameters that may be `None`, fixing crashes on `LayerNorm` / `RMSNorm` / `GroupNorm` configured without affine.
- `bst_set_norm` early-returns when both `scale` and `offset` are `None`; `bst_set_batchnorm` omits `None`-valued keys from the weight dict.
- `lookup_export` no longer rebuilds an O(N) dict on every call.

#### `brainstate.nn` (#215)

A systematic audit of the neural-network module:

- **Dropout & activations**: corrected the self-normalizing affine constants in `AlphaDropout` / `FeatureAlphaDropout`; fixed unbatched minimal-dim detection in `Dropout2d` / `Dropout3d` (per-element mask independence); defaulted `Softmin` / `Softmax` / `LogSoftmax` to the last axis; and fixed unit/integer handling in `rrelu` and the `soft_shrink` zero branch.
- **Linear & init**: fixed `ScaledWSLinear` mask/weight/bias shapes and `AllToAll` `out > in` padding; corrected `TruncatedNormal` default bounds and the `clip_grad_norm` unitless-gradient note.
- **Metrics**: added the `'weighted'` average (with validation) to `Precision` / `Recall` and fixed the `Welford` integer counter.
- **Bijective transforms**: `Softplus` / `NegSoftplus` / `Negative` / `Ordered` now use a saturation-free forward and a unit-safe stable inverse; `Sigmoid` / `Affine` `log_abs_det_jacobian` handle units and per-batch shape; `Affine` checks for a zero scale; `HiData.clone` / `add` / `pop` / `replace` preserve the name.
- **Module & collective ops**: `assign_state_values` accepts pytree / `Quantity` values via `tree.map` and dotted-string or tuple keys; `vmap_call_all_fns` is rebuilt on `vmap_new_states` (fixing a `BatchTracer` leak); `Map.update` no longer forwards `spmd_axis_name` to `pmap2`; an empty-slice of a `Sequential` returns an empty `Sequential`; and the `in_size` / `out_size` setters accept numpy scalars and 0-d arrays uniformly.
- **Delay & dynamics**: `Delay.max_time` grows monotonically across registrations; unit-aware retrieval no longer crashes or double-applies units; `update_every` is now functional via a monotonic per-call write pointer; and `FixedNumConn` respects the seed for its `afferent_ratio` mask and guards the unsupported `efferent_target='pre'` path with a clear `NotImplementedError`.

#### `brainstate.transform` mapping engine (#216)

Eight verified bugs in the `vmap` / `pmap` / `shard_map` engine:

- An `'auto'` undeclared read-modify-write state whose leading dim differs from the batch grew a new leading axis on every warm call; per-lane promotion is now re-decided from the live value each call, so warm and cold results agree.
- `pmap2_new_states` failed when the init used no `RandomState`; a dummy iota is now fed (and ignored) so `pmap` always has a mapped argument.
- `'auto'` could silently flip read-modify-write vs. scatter on a coincidental dimension match; a `_ReadTrackingTrace` now separates genuine reads and a one-time `UserWarning` fires when an undeclared RMW state's dim differs from the batch.
- `axis_size` is validated, raising a clear `ValueError` on conflict with the inferred batch size; `map` over a 0-d input raises a clear `ValueError` instead of a cryptic `IndexError`; and the legacy `vmap` undeclared-write error now speaks the `out_states` vocabulary.
- `shard_map`'s undeclared (replicated) per-shard write against sharded data now points at `state_in_specs` / `state_out_specs` instead of failing with an opaque broadcast error.
- `StatefulMapping(static_argnums=...)` no longer maps the static argument: static positional args are closed over, matching `jax.jit`.

#### `brainstate.nn.exp_euler` (#210)

- Corrected the Jacobian unit conversion in the drift calculation of the exponential-Euler integrator, and clarified the diagonal-Jacobian docstring.

#### `brainstate.util` & cross-module hardening (#217)

The consolidated `dev/issues.md` audit resolved every catalogued finding across `nn`, `random`, `transform`, `util`, `graph`, `interop`, and the core, plus utility edge cases surfaced separately. Each fix is paired with a behavioral regression test; genuinely ambiguous contracts were resolved by documenting the existing behavior rather than silently changing it.

### Hardening (stricter validation)

Runtime validation that previously relied on `assert` — and was therefore stripped under `python -O`, allowing invalid input to flow through to silently wrong results — now raises descriptive `TypeError` / `ValueError` across `nn`, `random`, `transform`, `util`, `graph`, `interop`, and the core (#217). All such checks target stable, public JAX APIs.

### Quality

- Full test suite: **5296 passed, 23 skipped**; `mypy` clean; patch coverage 100% (lines) for the cross-module audit and 98% for the mapping-engine fixes (#216, #217).
- Verified green on the complete CI JAX matrix: 0.7.0, 0.8.0, 0.9.0, and latest.


## Version 0.4.2 (2026-06-10)

A correctness-hardening patch release for `brainstate.transform`. A JAX-expert audit of the state-based transformation layer — `jit`, `grad` / `vector_grad` / `jacobian` / `hessian`, `cond` / `switch` / `ifelse`, the bounded and collecting loops, the state-aware mapping engine, `shard_map`, `checkify`, `named_scope`, and `checkpoint` — surfaced a family of stale-cache, tracer-leak, and silent-misbehavior bugs. This release fixes every reproduced issue and tightens argument validation so that previously silent wrong-result paths now fail loudly. The minimum supported JAX is raised to 0.7.0. Each fix ships with a regression test verified to fail before and pass after the change (#207, #208).

### Bug Fixes

- **Stale compiled trace after an out-of-band state change**: when a captured `State`'s shape or dtype changes between calls, `StatefulFunction` no longer replays a stale cached jaxpr (which silently produced wrong results). A state-aval mismatch is now treated as a cache miss, triggering recompilation across `get_arg_cache_key`, `make_jaxpr`, and `__call__` (#207).
- **`cond` / `switch` / `ifelse` with asymmetric branch state access**: fixed a crash when a state is written in one branch but only read in others, and fixed a state-value misalignment between the merged trace order and each branch's own trace order in the multi-branch wrappers (#207).
- **`bounded_while_loop` correctness**: fixed wrong results caused by the checkpointed-scan counter bump leaking into user carries, by `max_steps=1` ignoring the loop condition, by missing per-lane masking under `vmap`, and by iteration-cap overshoot (#207).
- **Tracer leaks on the failure path**: `make_jaxpr`, the state-aware mapping engine, `shard_map`, `checkify`, `vmap_new_states`, `map`, and `eval_shape` now snapshot and restore original state values (including RNG backups) when the wrapped execution raises, so a failed trace no longer leaves dead tracers in global states. The mapping engine additionally detects a stale cached plan via a write-set watcher and rebuilds it once before failing (#207).
- **States created inside a trace no longer leak a dead tracer**: such a `State` is poisoned after tracing with an `_InvalidatedTraceValue` sentinel — reading it raises a descriptive `TraceContextError`, and assigning a concrete value clears the poison (#207).
- **Cached compilations no longer retain enclosing-trace tracers**: original-value snapshots are replaced with their avals before a trace is cached, so `grad`-under-`jit` now passes `jax.checking_leaks()` (#207).
- **`grad(..., debug_nan=True)`**: fixed an `AttributeError` when the transformed callable is a `functools.partial` (which has no `__name__`); under an enclosing trace, the NaN flag is now routed through `lax.cond` plus an ordered callback instead of being concretized (which raised `TracerBoolConversionError` under `jit`) (#207).
- **`hessian` block structure**: results are now returned structured like `grad_states` rather than exposing internal id-keyed dictionaries (#207).
- **Ahead-of-time `jit` paths** (`eval_shape` / `lower` / `trace` / `compile`) no longer perform a spurious state writeback that marked read-only states as written in an enclosing trace (#207).
- **`States` passed via keyword arguments** are no longer silently flattened: the in-`kwargs` state check now runs before abstractification in `get_arg_cache_key` (#207).
- **`named_scope`**: jit-compiled functions are now cached per static configuration; a `conda:false` trace-name typo in `cond`, an incorrect `ifelse` docstring example, and documentation for nonexistent `non_static_*` parameters were all corrected (#207).
- **`NewStateCatcher.get_by_tag`** now matches against the catcher's tag set instead of failing to find tagged states (#207).

### Behavior Changes (stricter validation)

The following paths previously produced silently wrong results or accepted invalid input; they now raise descriptive errors:

- **Writing a tracer into a pre-existing `State` outside a `brainstate` trace** (for example under raw `jax.jit` / `vmap` / `grad` / `scan`) now raises a `TraceContextError` instead of silently storing the tracer. States created inside the current JAX trace remain exempt, since they die with that trace (#207).
- **`grad` / `vector_grad` / `jacobian` / `hessian`** reject negative and non-integer `argnums` up front instead of differentiating the wrong argument; `hessian` additionally rejects the `grad_states` + `argnums` combination (#207).
- **`jit`** aligns user-supplied `in_shardings` / `out_shardings` with the internally prepended state argument and rejects negative `static_argnums` / `donate_argnums`; `checkpoint` / `remat` likewise reject negative `static_argnums` (#207).
- **Unhashable static arguments** raise an actionable `TypeError` (#207).
- **`checkpointed_scan`** raises a clear `ValueError` for `length < 1` instead of a math-domain error, and `ProgressBar` frequency validation raises `ValueError` rather than failing an `assert` (#207).

### Build

- **Minimum JAX raised to `>=0.7.0`** (previously `>=0.6.0`) across all `pyproject.toml` extras (`cpu`, `cuda12`, `cuda13`, `tpu`, `testing`) and `requirements.txt` (#208).


## Version 0.4.1 (2026-06-09)

A focused patch release that hardens the shared state-aware mapping engine behind `vmap` / `pmap` / `map` (and their module-level `*2` variants) against a set of correctness edge cases surfaced by a JAX-expert audit, alongside a routine CI and developer-dependency refresh. No public APIs change.

### Bug Fixes

- **Read–modify–write states no longer accumulate a spurious axis under mapping**: an undeclared state that a mapped function reads and writes in place, and whose shape already matches the mapped axes, is now auto-promoted to a per-lane input *and* output. Previously each call grew an extra leading axis on the state's value (#203).
- **`pmap2` now rejects positional argument indices it cannot honor**: `static_broadcasted_argnums` and `donate_argnums` are no longer silently accepted, because those indices addressed the wrapper's internally bundled arguments rather than the user's. Passing them now raises an explicit error (#203).
- **Stale plan cache after state garbage collection**: the mapping engine's plan cache is now weakref-backed. When any state captured by a cached plan has been garbage-collected — for example after a module is re-initialized — the plan is rebuilt instead of scattering writes onto orphaned `State` objects (#203).
- **Random sampling inside batched `map`**: drawing random numbers within `map(..., batch_size=...)` is now supported (#203).
- **Consistent replication of non-batched states in the legacy `vmap_new_states`**: `NonBatchState` / `INIT_NO_BATCHING` states created inside `vmap_new_states` are now replicated rather than batched along axis 0, matching the behavior of `vmap2_new_states` (#203).

### Internal Changes

- Consolidated the new-state resolver and the `INIT_NO_BATCHING` sentinel into the shared `_mapping_core` module, re-exported from `_mapping2` to preserve backward compatibility (#203).
- Documented and hardened the zero-placeholder shape probe and value-dependent control flow, multi-pass (Python-level) side effects, the double `init_all_states` pass, and the engine's thread-safety guarantees (audit items B4, B7–B10) (#203).
- Merged the standalone composition and nested-leak test suites into the primary `_mapping1` / `_mapping2` / `_mapping_core` test modules; the full suite reports 4645 passed, 24 skipped (#203).

### CI/CD

- Bumped `codecov/codecov-action` from v5 to v7 (#199, #202).
- Bumped `actions/cache` from v4 to v5 (#200).
- Refreshed development dependencies (`braintools`, `mypy`) in `requirements-dev.txt` (#201).


## Version 0.4.0 (2026-06-01)

### Breaking Changes

- **Renamed `jit_named_scope` to `named_scope`**: The `brainstate.transform.jit_named_scope` decorator is now exported as `brainstate.transform.named_scope`. Update any usage accordingly.
- **Removed `brainstate.transform.sofo_grad`**: the second-order forward-mode (SOFO) gradient helper has moved to `braintools`. Replace `brainstate.transform.sofo_grad(fn, ...)` with the `braintools.optim.SOFO` optimizer (see `examples/009_sofo_mnist.py` for the updated usage).
- **Removed `brainstate.graph.NodeDef` and `brainstate.graph.NodeRef`**: the graph representation was reworked. A flattened graph is now described by `brainstate.graph.NodeSpec` together with the new edge types (`NodeEdge`, `StateEdge`, `StateLeafEdge`, `PytreeEdge`, `StaticEdge`, `Static`). Code that referenced `NodeDef`/`NodeRef` directly must migrate to these types; users of the high-level `graph.flatten` / `graph.treefy_split` / `graph.treefy_merge` API are unaffected.

#### Typed PRNG Keys in `brainstate.random`

`brainstate.random` now uses JAX's modern **typed PRNG keys** (`jax.random.key`,
dtype `key<fry>`, scalar shape `()`) everywhere a key is produced, replacing the
legacy raw `uint32[2]` representation.

- **`get_key()`, `split_key()`, `split_keys()`, `self_assign_multi_keys()`, and `RandomState.value` now return typed keys.** A single key has shape `()` (was `(2,)`); a batch of `n` keys has shape `(n,)` (was `(n, 2)`). Code that asserted `key.shape == (2,)` or `key.dtype == uint32`, or that indexed the raw words of a key, must be updated.
- **Key inputs accept three forms**: an integer seed, a typed JAX key, or a legacy `uint32[2]` array (the last is auto-wrapped via `jax.random.wrap_key_data`). Passing an integer seed array of size 1 is also accepted. Invalid inputs now raise `TypeError` (previously `ValueError` in some paths).
- **`RandomState` remains transform-compatible**: typed keys `vmap`/`jit`/`grad` cleanly over their leading axis, and state-aware transformations that special-case `RandomState` continue to work unchanged.
- The module-level `DEFAULT` generator still constructs without triggering JAX backend initialization at import time: it holds a lazy `uint32[2]` placeholder that is materialized into a typed key (via `wrap_key_data`, preserving the exact seed) on first use.

**Migration**: to recover the raw `uint32[2]` words from a typed key, use the new
`brainstate.random.get_key_data()` or `jax.random.key_data(key)`.

### New Features

#### Inline Type Information (PEP 561)

- **`py.typed` marker added**: `brainstate` now ships inline type information, so downstream projects' type checkers (mypy, pyright, etc.) pick up brainstate's annotations automatically.
- **Typing correctness gate**: a `mypy` configuration with a per-module "ratchet" enforces type correctness in CI, starting with `brainstate.typing`. Coverage expands module-by-module over time.
- All annotations are evaluated lazily (`from __future__ import annotations`), so they impose no import-time or runtime cost.

#### Physical Unit Support in `brainstate.random`

Random distributions are now **comprehensively and strictly compatible with
`brainunit` physical units**, with a consistent location–scale convention.

- **Location/scale parameters carry the output unit**: `normal`, `laplace`, `logistic`, `gumbel`, `wald`, and `truncated_normal` propagate the unit of their `loc`/`scale` (or `mean`/bounds) into the samples. When only one of `loc`/`scale` carries a unit, the plain value is interpreted in that same unit; a compatible-but-different unit (e.g. `volt` against `mV`) is converted, while an incompatible one raises `UnitMismatchError`.
- **Scale-only distributions carry the scale unit**: `exponential`, `gamma`, `rayleigh`, and `weibull_min` propagate the unit of their `scale` parameter.
- **`multivariate_normal`** carries the unit of `mean` (with `cov` required to be `mean`-unit squared).
- **Shape / rate / count / probability parameters are strictly dimensionless**: parameters such as `df`, `a`/`b`, `lam`, `n`, `p`, `alpha`, `logits`, `kappa`, `concentration`, and friends reject a dimensional `Quantity` with a clear `ValueError`. A genuinely dimensionless `Quantity` (e.g. `3.0 * u.UNITLESS`) is accepted.
- **No units → plain arrays**: every distribution returns a plain array when given plain inputs, so existing unitless code is unaffected.

#### Raw Key Interop Helper

- **`brainstate.random.get_key_data()`** returns the current global key as a raw `uint32[2]` array (via `jax.random.key_data`), for interfacing with code that still expects the legacy representation.

#### Framework Interoperability (`brainstate.interop`)

A new `brainstate.interop` module converts modules to and from other JAX
frameworks, with an extensible layer registry:

- **Flax NNX**: `to_nnx` / `from_nnx`.
- **Flax Linen**: `to_linen` / `from_linen`.
- **Equinox**: `to_equinox` / `from_equinox`.
- **Registry**: `register_layer_mapping`, `supported_layers`, `LayerMapping`.
- **Typed errors**: `InteropError` and its subclasses (`MissingDependencyError`, `UnmappedLayerError`, `UnsupportedLayerError`, `UnsupportedStructureError`, `MissingShapeError`, `ConversionError`).

#### New Transformations

`brainstate.transform` gains several state-aware transformations:

- **`vjp` / `jvp`**: state-aware reverse- and forward-mode differentiation products (companions to `grad`).
- **`shard_map`**: a state-aware wrapper over `jax.shard_map` for SPMD sharding.
- **`named_call`**: attach a name to a sub-computation for clearer jaxprs and profiles.
- **Runtime checks (`checkify` family)**: `checkify`, `check`, `check_error`, and the error-class selectors `nan_checks`, `div_checks`, `index_checks`, `float_checks`, `user_checks`, `automatic_checks`, `all_checks`.
- **`register_prim_handler`**: register custom primitive handlers for the IR/codegen pipeline.

### Bug Fixes

- **`multivariate_normal` now propagates physical units**: previously the output unit was read after the mantissa had already been stripped from `mean`, so units were silently dropped. Samples now correctly carry the unit of `mean`.
- **`truncated_normal` now accepts unit-carrying bounds with default `loc`/`scale`**: the shared output unit is inferred from whichever of `lower`/`upper`/`loc`/`scale` carries one, and plain values are interpreted in that unit (previously a unit on the bounds with the default plain `loc`/`scale` raised `UnitMismatchError`).
- **`brainstate.transform.vjp` now supports state-only differentiation**: calling `vjp(fun, grad_states=...)` with no differentiable positional argument (e.g. a loss that closes over trainable parameters) previously raised `IndexError`. It now returns a pullback yielding just the state cotangents, matching `brainstate.transform.grad` semantics.
- **`brainstate.transform.vjp` accepts `argnums=None`**: like `grad`, `argnums=None` disables positional-argument differentiation so the pullback returns only state cotangents.
- **Clearer `vjp` errors**: out-of-range `argnums` now raises a descriptive `ValueError` instead of a bare `IndexError`, and supplying neither positional primals nor `grad_states` raises an explanatory `ValueError`.
- **No `jax.core.DropVar` deprecation warning on import**: the JAX compatibility layer now sources `DropVar` from `jax.extend.core` on JAX >= 0.10, removing a redundant deprecated import.

### Known Issues

Known defects deferred to a future patch release (each has a skipped regression
test capturing the repro):

- `nn.AdaptiveAvgPool2d/3d` (and Max variants) raise `TypeError` when a target dimension is `None`, despite documenting `None` as "do not pool this dimension".
- `random.truncated_normal` / `nn.init.TruncatedNormal()` crash when `lower`/`upper` are left at their `None` defaults.
- `nn.weight_standardization` raises when given a unit-carrying `Quantity` input.
- The `nn` collective-op `vmap`-call helpers can leak a JAX `BatchTracer` into newly created state values.
- `nn` delay unit retrieval can fail with a pytree-node mismatch (`Quantity` history vs `Unit`).
- `nn` event fixed-probability connectivity with `efferent_target='pre'` can crash (and, with `afferent_ratio < 1`, abort) inside the `brainevent` CSC path.
- State filtering with the documented `{filter: axis}` mapping form raises `TypeError`.


## Version 0.3.0

This release delivers on-device NaN debugging, a unified compilation cache, simplified JAX compatibility, and major internal cleanup — with a net reduction of ~1,800 lines of code. It raises the minimum requirements to Python 3.11 and JAX 0.6.0.

### Breaking Changes

- **Python >= 3.11 required**: Dropped support for Python 3.10. The `requires-python` field and classifiers now start at 3.11.
- **JAX >= 0.6.0 required**: All dependency groups (`cpu`, `cuda12`, `cuda13`, `tpu`, `testing`) now mandate `jax>=0.6.0`.
- **Unified compilation cache in `StatefulFunction`**: The four separate internal caches (`_cached_jaxpr`, `_cached_out_shapes`, `_cached_jaxpr_out_tree`, `_cached_state_trace`) have been consolidated into a single `_compilation_cache` storing `_CachedCompilation` objects. `get_cache_stats()` now returns `{'compilation_cache': {...}}` instead of four individual entries.
- **Immutable `CacheKey` replaces `hashabledict`**: `get_arg_cache_key()` now returns a `CacheKey` (NamedTuple) instead of the mutable `hashabledict`. Code that directly inspected or constructed cache keys must be updated.
- **Removed internal `_make_jaxpr` function**: The custom tracing implementation has been deleted in favor of using `jax.make_jaxpr()` directly (available in JAX >= 0.6.0).
- **Removed `debug_depth` and `debug_context` from `GradientTransform`**: The `depth` and `context` parameters for NaN debugging no longer exist following the debug module rewrite.
- **Removed `breakpoint_if` function**: The conditional breakpoint helper has been removed from `brainstate.transform._debug`.
- **Removed `extend_axis_env_nd` from compatible imports**: This compatibility shim is no longer exported.

### New Features

#### On-Device NaN/Inf Detection

- Complete rewrite of the NaN debugging system (`brainstate.transform._debug`). NaN checking now runs **on-device** via JAX primitives rather than pulling data to the host, providing significantly better performance.
- Uses `jax.debug.callback` with thread-local storage to collect and report NaN findings.
- Error tracebacks now point to the **user's source code** via `source_info_util.user_context`, producing IDE-clickable source locations extracted from jaxpr equations.
- Recursive instrumentation of nested primitives (`jit`, `cond`, `while`, `scan`) for comprehensive NaN detection throughout the computation graph.
- More compact and informative error messages via `_format_nan_message()`.

#### JAX Traceback Filtering

- Registered brainstate with JAX's `traceback_util.register_exclusion()` so internal frames are hidden in user-facing error tracebacks. Follows the same pattern as Flax, Equinox, and other JAX ecosystem libraries.
- Users can still see full tracebacks via `JAX_TRACEBACK_FILTERING=off`.

#### State Validation at Call Time

- New `_validate_state_shapes()` method checks that current state shapes and dtypes match those recorded at compile time.
- `StatefulFunction.__call__()` automatically validates before execution, catching state shape mismatches early with clear error messages.
- Added `static_argnums` bounds validation — `make_jaxpr()` now raises `ValueError` if indices exceed the number of positional arguments.

#### New Compatible Import

- Added `mapped_aval` import with version-based routing: `jax.core.mapped_aval` for JAX < 0.8.2, `jax.extend.core.mapped_aval` for >= 0.8.2.

### Improvements

- **Atomic cache writes**: Compilation results are only stored on success, eliminating partial cache entries on error. Uses a double-checked locking pattern for thread safety during compilation.
- **Better cache key hashing**: Dynamic args/kwargs are now flattened via `jax.tree.flatten()` before hashing, fixing non-deterministic hashing issues with custom pytree nodes (e.g., `Quantity`).
- **Modern Python type annotations**: Migrated from `typing.Tuple`, `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Union` to built-in `tuple`, `list`, `dict`, `X | None`, `X | Y` syntax across the codebase.
- **IR visualization compatibility**: Replaced direct `jax.core.X` references with compatible imports (`Var`, `ClosedJaxpr`, `Jaxpr`, `JaxprEqn`, `Literal`, `DropVar`) in the IR visualizer.
- **Deterministic error reporting**: `jax.debug.callback` in `_error_if.py` now uses `ordered=True` for deterministic error callback ordering.
- **Graph operations cleanup**: Major refactoring of `_operation.py`, `_node.py`, `_convert.py`, and `_context.py` with streamlined docstrings, better thread-safety documentation, and cleaner context managers.

### Bug Fixes

- **Fixed `Delay.__init__` initialization order**: `update_every` is now initialized before `register_entry` is called, preventing attribute errors during entry registration (#135).
- **Fixed `graph_to_tree` private attribute access**: Replaced internal `_mapping` access with public API usage in `_convert.py`.

### Internal Changes

- Massive docstring reduction across the graph module (~1,000+ lines removed), replacing verbose multi-paragraph docstrings with concise descriptions.
- Cleaned up TypeVar usage: removed unused `C` and `Names` aliases, renamed `Node` TypeVar to `N`, removed `Hashable` bound from type variables.
- Removed unused tests (`test_all_exports`, `test_function_imports_availability`) from compatible import tests.
- Rewrote debug and make_jaxpr test suites to match the new APIs.
- IR optimization imports are now lazy-loaded inside `make_jaxpr()` only when `ir_optimizations` is configured.

### CI/CD

- Bumped `actions/upload-artifact` from v6 to v7.
- Bumped `actions/download-artifact` from v7 to v8.


## Version 0.2.10

This release introduces a comprehensive NaN debugging system for gradient computations, refactors the module mapping API for improved clarity, and adds graph context utilities for advanced state management.

### New Features

#### NaN Debugging System

- **JIT-Compatible NaN/Inf Debugging**: New debugging utilities for identifying NaN and Inf values during gradient computations
  - `debug_nan`: Analyze a function for NaN/Inf values with detailed reporting
  - `debug_nan_if`: Conditional NaN debugging with predicate-based activation
  - Full JIT compatibility for seamless integration into compiled workflows
  - Support for debugging NaN in `while` and `scan` primitives
  - Detailed analysis output including variable names, shapes, and affected indices

- **Gradient Function Integration**: Added `debug_nan` parameter to gradient transformation functions
  - `grad`: Enable NaN debugging during gradient computation
  - `vector_grad`: NaN debugging for vectorized gradients
  - `jacobian` and `jacobian_reverse`: NaN debugging for Jacobian computations
  - `hessian`: NaN debugging for Hessian computations

- **Breakpoint Utility**: New `breakpoint` function for conditional debugging
  - Wraps `jax.debug.breakpoint` with predicate support
  - Only triggers when the specified condition is True

### API Changes

#### Module System

- **Renamed `ModuleMapper` to `Map`**: Simplified naming for the vectorized module wrapper
  - `Map` provides vectorized (`vmap2`) and parallel (`pmap2`) mapping over modules
  - `ModuleMapper` retained as a deprecated alias for backward compatibility
  - Internal `_ModuleMapperCalling` renamed to `_MapCaller` for consistency

- **Enhanced `Map.map()` Method**: Now accepts callable functions for flexible mapping operations

### Bug Fixes

- Fixed `get_backend` import for JAX version compatibility across different JAX releases
- Removed `abstractmethod` decorators from `Regularization` class to allow proper instantiation
- Cleaned up unused imports in module initialization files

### Internal Changes

- Added comprehensive test suite for NaN debugging (`_debug_test.py`, 938 lines)
- Removed deprecated `_mapping3.py` module and associated tests
- Streamlined module exports in `__init__.py` files


## Version 0.2.9

This release introduces a powerful state hook system for advanced state management, refactors neural network modules with enhanced parameter handling, and improves delay mechanisms with frequency-controlled updates.

### State Management

#### State Hook System

- **Global Hook Infrastructure**: Comprehensive hook system for intercepting state operations
  - `register_read_hook`: Register hooks that execute when state values are read
  - `register_write_hook`: Register hooks that execute when state values are written
  - `register_restore_hook`: Register hooks that execute when state values are restored
  - `HookManager`: Thread-safe manager for organizing and executing hooks with priority support
  - `HookContext`: Context manager for scoped hook registration and execution
  - Enables advanced use cases: logging, debugging, value transformation, validation

- **Enhanced State Class**: Improved state management with hook integration
  - Automatic hook execution on read/write operations
  - Better cache key handling for improved performance
  - Enhanced thread safety and context management
  - Comprehensive test coverage (346 tests for thread safety, 320 tests for hooks)

### Neural Network Components

#### Parameter Management (`brainstate.nn.Param` and `brainstate.nn.Const`)

- **Renamed Classes**: Simplified naming convention
  - `ParaM` → `Param`: Trainable parameter wrapper
  - `ConstM` → `Const`: Non-trainable constant wrapper

- **Enhanced Caching System**: Improved parameter precomputation and caching
  - `param_precompute` context manager for efficient parameter transformation caching
  - `cache()` method for retrieving cached parameter values
  - Support for custom precompute functions
  - Automatic cache invalidation and management
  - 391 comprehensive tests for caching behavior

- **Hierarchical Parameter Data** (`brainstate.nn.HiData`): New module for structured parameter organization
  - `define_param_data()` method for declaring hierarchical parameter structures
  - Support for nested parameter groups
  - Improved parameter surgery and manipulation
  - Enhanced type hints and documentation

#### Module System Enhancements

- **ModuleMapper**: New helper for vectorized module operations (formerly `Vmap2Module`)
  - Simplified API for applying `vmap2` to module methods
  - Automatic state management for vectorized operations
  - Consistent interface with `Vmap2ModuleCaller`
  - Comprehensive documentation with usage examples

- **Enhanced Module Methods**:
  - `parameters()`: Iterate over all parameters in the module hierarchy
  - `named_parameters()`: Iterate over parameters with their qualified names
  - `children()`: Access direct child modules
  - `named_children()`: Access child modules with names
  - `init_all_states()`: Initialize states with additional keyword arguments
  - Improved `Sequential` with `extend()` and `insert()` methods

#### Delay Mechanisms

- **Frequency-Controlled Updates**: Enhanced `Delay` class with flexible update strategies
  - `update_every` parameter: Control how often delay buffers are updated
  - Support for integer steps (update every N steps)
  - Support for time-based updates with physical units (e.g., `1*ms`)
  - Automatic handling of unit conversions and validation
  - Comprehensive tests covering various update strategies

- **Unified Delay Implementation**: Refactored delay mechanism
  - Ring buffer implementation for efficient historical value storage
  - Support for linear interpolation
  - Better handling of multi-dimensional inputs
  - Improved integration with neural network modules

#### Regularization

- **Comprehensive Regularization Module** (`brainstate.nn._regularization`, 2840 lines):
  - Complete suite of regularization techniques
  - L1, L2, and elastic net regularization
  - Dropout variants
  - Weight decay and other parameter constraints
  - 1261 tests for regularization functionality

- **Transform Module** (`brainstate.nn._transform`, 1661 lines):
  - Advanced parameter transformations
  - Quantization support
  - Normalization techniques
  - Integration with caching system
  - 452 comprehensive tests

### Transformations

#### Vectorization and Parallelization

- **Mapping Function Refactoring**: Reorganized mapping implementations
  - Renamed `_mapping.py` → `_mapping2.py` (primary `vmap2` implementation)
  - Renamed `_mapping_old.py` → `_mapping1.py` (legacy `vmap` implementation)
  - Added `_mapping3.py`: New `pmap2` implementation for parallelization
  - `vmap2_new_states`: Helper for creating new states in vectorized operations
  - Relaxed return type requirements for more flexible mapping functions

- **Enhanced Documentation**: Updated tutorials and API documentation
  - Comprehensive `vmap2` tutorial with practical examples
  - Enhanced parallelization documentation for `pmap2`
  - Updated state management guides
  - Expanded gradient transformation documentation

### Compatibility and Utilities

#### JAX Compatibility

- **Enhanced JAX Integration**: Improved compatibility with newer JAX versions
  - Updated backend import for JAX version detection
  - Enhanced `get_aval` function for JAX version compatibility
  - Standardized `jit_named_scope` arguments
  - Support for JAX 0.8.0+ in CI configuration

#### Utility Functions

- **Dataclass Support**: Added `is_dataclass` utility function in `brainstate.util.struct`
  - Robust dataclass type checking
  - Better handling of dataclass-based structures

- **Tracer Utilities**: New `_tracers.py` module for JAX tracer handling
  - `current_jax_trace()`: Get current JAX trace context with version compatibility
  - Helper functions for working with JAX abstract values

### Graph Operations

- **Context Management** (`brainstate.graph._context`):
  - New context management system for graph operations (119 lines)
  - `TraceContextError`: Specialized error class for tracing issues
  - Enhanced state tracking during graph construction
  - 64 tests for context management

- **Conversion Utilities** (`brainstate.graph._convert`):
  - New conversion utilities for graph operations (278 lines)
  - Better handling of graph transformations
  - Improved node conversion logic

### Random Number Generation

- **Enhanced RandomState**: Improved random number generation
  - Better compatibility with newer JAX versions (98 lines of improvements)
  - Enhanced state management for random keys
  - Improved thread safety
  - Better error messages and validation

### Documentation

- **Comprehensive API Documentation**: Expanded documentation across all modules
  - `brainstate.rst`: Reorganized with improved structure (21 lines removed, refactored into submodules)
  - `environ.rst`: Added 48 lines of documentation for environment state and keys
  - `nn.rst`: Added 222 lines documenting neural network components
  - `transform.rst`: Added 132 lines for gradient transformations and mapping functions

- **Tutorial Updates**:
  - Updated vectorization tutorial to reflect `vmap` → `vmap2` transition
  - Enhanced examples with `ModuleMapper` usage
  - Improved state management examples

### Breaking Changes

- **Renamed Functions and Classes**:
  - `ParaM` → `Param`
  - `ConstM` → `Const`
  - `vmap` → `vmap2` (old `vmap` preserved in `_mapping1.py` for compatibility)
  - `pmap` → `pmap2`
  - `_param_data` → `_hidata`

- **Parameter Naming Standardization**:
  - `fit_par` → `fit` across all modules
  - `brainscale` → `braintrace` in example files

- **Method Signature Changes**:
  - `init_all_states()` now accepts additional keyword arguments
  - `param_precompute()` signature updated to support caching and custom functions
  - Module initialization methods enhanced with keyword argument support

### Bug Fixes

- Fixed cache key handling in state management
- Improved error messages for missing states in gradient transformations
- Enhanced validation for delay update frequency
- Corrected import paths for better module organization
- Fixed compatibility issues with JAX 0.8.0+

### Internal Changes

- Reorganized import statements across all modules for clarity
- Enhanced type hints throughout the codebase
- Improved code documentation with comprehensive docstrings
- Streamlined module exports in `__all__` definitions
- Better separation of concerns in module organization


## Version 0.2.8

This release ensures compatibility with JAX 0.8.2+ and removes the experimental module that was superseded by upstream changes.

### Compatibility

- **JAX 0.8.2+ Support**: Added compatibility with JAX version 0.8.2 and later. The library now uses `jax.make_jaxpr` directly for JAX >= 0.8.2 while maintaining backward compatibility with earlier versions.

### Breaking Changes

- **Removed `abstracted_axes` parameter**: The `abstracted_axes` parameter has been removed from:
  - `StatefulFunction.__init__`
  - `StatefulMapping.__init__`
  - `make_jaxpr` function
  - `_make_jaxpr` internal function

### Improvements

- **Debug mode support**: Added `debug_call` method to `StatefulFunction` for proper execution when `jax.config.jax_disable_jit` is enabled. This improves debugging workflows by allowing stateful functions to execute without JIT compilation.

- **Lazy loading optimization**: `RandomState` import in the `_mapping` module is now lazily loaded via `_import_rand_state()`, improving initial import performance and reducing circular dependency issues.

### Internal Changes

- Removed unused imports (`annotate`, `api_boundary` from `jax._src`) at module level; now imported only where needed
- Removed internal helper functions `_broadcast_prefix` and `_flat_axes_specs`
- Simplified `_abstractify` function by removing abstracted axes handling
- Updated example files to reflect API changes

## Version 0.2.7

BrainState 0.2.7 modernizes the experimental compilation stack, deepens the transformation APIs, and tightens runtime infrastructure across the project.

### Experimental Compiler and Visualization

- Introduced the experimental `neuroir` compiler built on dataclass-based graph IR elements and an explicit `CompilationContext`, improving dependency tracking, hidden-state mapping, and ClosedJaxpr fidelity even for self-connections and delay buffers.
- Added GraphDisplayer and TextDisplayer backends with hierarchical and force-directed layouts, plus richer diagnostics and tests that cover large sample networks and neuro-graph visualizations.

### Transformations and Autodiff

- Added the `jit_named_scope` decorator and supporting utilities so nested transformations emit meaningful names inside traced functions, together with `_make_jaxpr` refinements that separate dynamic/static arguments and improve caching semantics for `StatefulFunction`.
- Expanded the gradient toolkit by exporting the new Jacobian (forward and reverse), Hessian, and SOFO transforms, unifying gradient handling for classes, auxiliary returns, and state-aware updates through the transform module.

### State and Runtime Enhancements

- Replaced the experimental `ArrayParam` with a dedicated `DelayState`, propagating the new state through the compiler, delay modules, and neuro-IR so historical buffers participate in tracing and optimization just like other states.
- Environment helpers can now run against injected `EnvironmentState` instances, enabling sandboxed or per-thread configurations while DelayState-aware unit tests extend coverage of the updated modules.

### Experimental and Infrastructure Updates

- Completed the neuron IR → neuroir rename, aligned the GDiist BPU codebase with the new terminology, and added new sample networks plus placeholder skips to keep the growing compiler/displayer test surface manageable.
- Added `braincell` to the development requirements, refreshed documentation wording, and kept CI dependencies current for the GitHub Actions runners.

### Bug Fixes

- Hardened caching, randomness, and initialization logic by fixing `get_arg_cache_key`, removing stale decorator parameters, validating truncated normal draws, and correcting the exported version metadata.
- Declared Python 3.14 support and cleaned up compiler import ordering to keep linting noise low.


## Version 0.2.6

This release focuses on the experimental export pipeline and device-aware execution adapters.

### Device-Aware Wrappers

- Added registry-driven `ForLoop` and `JIT` adapters that expose decorator-style ergonomics, call counters, and validation, with CPU/GPU/TPU implementations wired through `register_*_impl` so experiments can swap device backends without touching user code.

### GDiist BPU Export

- Replaced the monolithic exporter with `gdiist_bpu.main`, refreshed parser/component/utils modules, and renamed `BpuParser` to `GdiistBpuParser`, yielding clearer analysis output, text display helpers, and far more granular unit tests.
- Introduced the thread-safe `BoundedCache` utility and integrated it with compiler wrappers to safely reuse traced graphs, alongside `_make_jaxpr` updates that enforce argument checks and improve cache key generation.
- Updated tutorials and examples to the streamlined naming scheme and refreshed device implementation docs for the new wrapper entry points.


## Version 0.2.5

Version 0.2.5 concentrates on intermediate-representation (IR) optimization quality.

### IR Optimization

- Added `_ir_optim_v2`, a comprehensive optimizer that ships constant folding, dead-code elimination, common subexpression elimination, copy propagation, and algebraic simplification passes backed by identity-aware set semantics.
- Updated the transform exports and accompanying tests to exercise the new optimizer while pruning unused configuration knobs from the earlier implementation.


## Version 0.2.4

This release introduces the new `ArrayParam` state type for parameter arrays with custom transformations, experimental BPU backend export support, enhanced JAXPR optimization capabilities, and improved module organization.

### New Features

#### ArrayParam State Type

- **ArrayParam Class**: New state type for managing parameter arrays with advanced transformation control
  - Supports custom transformations (e.g., quantization, normalization) that preserve array identity
  - Enables `vmap`, `pmap`, and other JAX transformations to correctly handle stateful parameters
  - Provides `identity()` method that returns the raw array without applying custom transformations
  - Integrates seamlessly with existing State management infrastructure
  - Useful for implementing quantization-aware training and other advanced parameter manipulations
  - Comprehensive documentation with usage examples and best practices

#### Experimental BPU Backend Export (`brainstate.experimental.gdiist_bpu`)

- **BPU Backend Export Support**: Complete infrastructure for exporting models to GDiist BPU hardware backend (727 lines)
  - `export.py`: Main export API with `to_bpu()` function for model conversion
  - `parser.py`: Operation parser that analyzes JAXPR to identify operations and connections (305 lines)
  - `data.py`: Data structures and analysis utilities for operation representation (215 lines)

- **Operation Parser Features**:
  - Automatic detection of operations from JAXPR equations using brainevent primitives
  - Data flow analysis to identify connections between operations
  - Support for various operation types: slice, add, multiply, and more
  - Detailed analysis output showing equations, inputs, outputs, and connections

- **Analysis and Debugging Tools**:
  - `display_analysis_results()`: Comprehensive visualization of parsed operations
  - Shows operation details including equation count, variable mappings, and connections
  - Displays connection information with producer/consumer operations and variable details
  - Example implementation in `examples/400_CUBA_2005_bpu.py`

### Enhancements

#### JAXPR Optimization Improvements

- **Enhanced Constant Folding**:
  - Better handling of literal values in constant folding optimization
  - Improved detection and elimination of redundant literal operations
  - More efficient constant propagation through computation graphs

- **Identity Equation Optimization**:
  - Optimized handling of `Literal` outputs to avoid unnecessary bridging equations
  - Improved identity equation creation for interface preservation
  - Better handling of edge cases in optimization passes

- **Error Handling**:
  - Added fallback source info utility for better error messages
  - Fixed potential NoneType errors in equation handling
  - Improved validation of optimization results

#### State Management

- **Enhanced State Tests**: Comprehensive test refactoring with improved coverage (454 tests)
  - Better organization of state type tests
  - More thorough validation of state behavior
  - Enhanced test readability and maintainability



## Version 0.2.3

This release introduces powerful IR (Intermediate Representation) optimization capabilities for JAX computation graphs, comprehensive state management refactoring for vectorized mapping operations, and extensive testing infrastructure improvements.

### New Features

#### IR Optimization (`brainstate.transform._ir_optim`)

- **Intermediate Representation Optimization Module** (876 lines): Complete suite of compiler-level optimizations for JAX computation graphs
  - `constant_fold`: Evaluates constant expressions at compile time, reducing runtime computation
  - `dead_code_elimination`: Removes equations whose outputs are unused, reducing computation overhead
  - `common_subexpression_elimination`: Identifies and reuses results of identical computations
  - `copy_propagation`: Eliminates unnecessary copy operations by propagating original variables
  - `algebraic_simplification`: Applies algebraic identities (x+0=x, x*1=x, x-x=0, etc.)
  - `optimize_jaxpr`: Orchestrates multiple optimization passes with configurable iteration and verbose mode

- **IdentitySet Class**: Custom set implementation using object identity (`id()`) instead of equality
  - Enables proper handling of JAX variables and Literals in optimization passes
  - Implements `MutableSet` interface with full collection protocol support
  - Essential for tracking variable usage without relying on equality comparisons

#### Optimization Features

- **Interface Preservation**: All optimizations preserve function input/output variables (invars/outvars)
  - Identity equations automatically added when needed to maintain correct interfaces
  - Uses `convert_element_type` primitive with matching dtypes as identity operation
  - Ensures optimized functions remain drop-in replacements

- **Optimization Pipeline**: Configurable multi-pass optimization with convergence detection
  - Customizable optimization sequence via `optimizations` parameter
  - Automatic convergence detection when no more reductions possible
  - Maximum iteration control with `max_iterations` parameter
  - Verbose mode with detailed statistics and progress tracking

- **JAX Integration**: Full support for JAX primitives and special cases
  - Blacklist for primitives that shouldn't be folded (broadcast_in_dim, broadcast)
  - Proper handling of `closed_call` and `scan` primitives
  - Support for both Jaxpr and ClosedJaxpr inputs

#### State Management Refactoring (`brainstate.transform._mapping`)

- **Renamed vmap to vmap2**: Major refactoring of vectorized mapping implementation (647 lines)
  - Enhanced state management with improved axis tracking
  - Better error messages and validation
  - Streamlined state value restoration logic

- **Old vmap Implementation Preserved** (`_mapping_old.py`, 579 lines): Legacy vmap with explicit state management
  - Exports original `vmap` and `vmap_new_states` functions
  - Maintains backward compatibility for existing code
  - Specialized for stateful functions with explicit state parameters

### Documentation

#### API Documentation

- **transform.rst**: Added comprehensive IR Optimization section (24 lines)
  - Detailed module description explaining compiler optimizations
  - All 6 optimization functions documented with autosummary
  - Clear explanation of benefits: reduced computation overhead, improved runtime performance
  - Positioned between Compilation Tools and Gradient Computations sections

- **NumPy-style Docstrings**: All optimization functions include:
  - Comprehensive parameter descriptions with types and defaults
  - Detailed return value documentation
  - Notes sections explaining preservation of function interfaces
  - Multiple practical examples demonstrating usage
  - Algorithm descriptions for complex optimizations
  - Cross-references between related functions

### Enhancements

#### Optimization Pipeline

- **Progress Tracking**: Verbose mode shows equation count changes after each optimization
  - Displays initial, intermediate, and final equation counts
  - Shows reduction statistics with percentages
  - Indicates convergence detection
  - Reports iteration counts

- **Validation**: Runtime checks ensure optimization correctness
  - Verifies input variables unchanged after optimization
  - Validates output variables preserved
  - Raises clear errors if interface violated
  - Checks for valid optimization names

- **Flexibility**: Customizable optimization sequences
  - Apply all optimizations in recommended order (default)
  - Select specific optimizations only
  - Control iteration limits
  - Toggle verbose output

#### JAX Integration

- **JaxprEqn Construction**: Proper handling of required `ctx` parameter
  - Uses `JaxprEqnContext(None, True)` for identity equations
  - Ensures compatibility with JAX internal API
  - Maintains proper equation structure

- **Primitive Handling**: Special cases for JAX primitives
  - Blacklist for primitives that shouldn't be optimized
  - Proper parameter extraction and validation
  - Support for effects and source_info fields

### Bug Fixes

- Fixed JaxprEqn constructor calls to include required `ctx` parameter (7th positional argument)
- Corrected import paths for `vmap2` in test files and tutorials
- Fixed `RandomState.uniform()` calls to use `size` parameter instead of `shape`
- Enhanced test assertions for proper state axis handling
- Improved error messages for batch axis mismatches

### Refactoring

#### Transform Module

- **Renamed Files**:
  - `vmap` → `vmap2` in `_mapping.py`
  - Preserved original `vmap` in `_mapping_old.py` for compatibility

- **Module Exports**: Updated `__init__.py` to export both old and new vmap implementations
  - `vmap` from `_mapping_old.py` (legacy)
  - `vmap2` from `_mapping.py` (new)
  - `vmap_new_states` from both modules

## Version 0.2.2

This release focuses on enhancing hidden state management for recurrent neural networks and eligibility trace-based learning, along with comprehensive testing and documentation improvements.

### New Features

#### Hidden State Classes

- **HiddenGroupState**: New class for managing multiple hidden states within a single array
  - Stores multiple states in the last dimension of a single array
  - Provides `get_value()` and `set_value()` methods for accessing individual states by index or name
  - Optimized for LSTM-style architectures with multiple hidden components (h, c)
  - Includes `name2index` mapping for convenient state access

- **HiddenTreeState**: New class for managing multiple hidden states with different physical units
  - Supports PyTree structure (dict or sequence) of hidden states
  - Preserves physical units (e.g., voltage, current, conductance) via `brainunit` integration
  - Provides `name2unit` and `index2unit` mappings for unit tracking
  - Ideal for neuroscience models with heterogeneous state variables
  - Maintains compatibility with BrainScale online learning

#### State Utilities

- **maybe_state**: New utility function for flexible value extraction
  - Extracts values from State objects automatically
  - Returns non-State values unchanged
  - Simplifies writing functions that accept both states and raw values

### Enhancements

#### State Classes

- **HiddenState**: Enhanced documentation and type checking
  - Restricted to `numpy.ndarray`, `jax.Array`, and `brainunit.Quantity` types only
  - Added comprehensive docstrings with examples
  - Clarified equivalence to `brainstate.HiddenState` for online learning
  - Improved error messages for invalid input types

- **BatchState**: Now properly exported in the public API
  - Available via `brainstate.BatchState`
  - Enhanced documentation for batch data management

#### Documentation

- **API Reference**: Completely reorganized `brainstate.rst` documentation
  - Organized into 6 major sections: Core State Classes, State Management, State Utilities, Error Handling, and Submodules
  - Added detailed descriptions for each section and subsection
  - Included comprehensive bullet-point summaries for all APIs
  - Enhanced deprecation warnings with clear migration paths
  - Added module-level descriptions for all submodules

- **State Classes**: Enhanced documentation for all state types
  - Added detailed use case descriptions
  - Included practical examples for each state type
  - Clarified semantic distinctions between state types
  - Documented integration with JAX transformations

- **JAX Transformations**: Improved documentation for stateful transforms
  - Enhanced docstrings for `jit`, `grad`, `vmap`, `scan`, and other transforms
  - Added examples showing state management patterns
  - Documented state tracing behavior
  - Clarified interaction with `StateTraceStack`

#### Transform System

- **Enhanced State Finding**: New `_find_state.py` module for automatic state discovery
  - Improved state detection in nested structures
  - Better handling of state dependencies
  - Enhanced error messages for state-related issues

- **StatefulFunction**: Major enhancements to `make_jaxpr` functionality
  - Improved Jaxpr generation for stateful computations
  - Better handling of state read/write tracking
  - Enhanced debugging support

- **Mapping Transformations**: Significant refactoring of `vmap` and `pmap`
  - Improved state management across vectorized operations
  - Better handling of state broadcasting
  - Enhanced error reporting for mapping operations

#### Random Number Generation

- **Module Reorganization**: Complete refactoring of random module structure
  - Renamed `_rand_funs.py` to `_fun.py`
  - Renamed `_rand_seed.py` to `_seed.py`
  - Renamed `_rand_state.py` to `_state.py`
  - Extracted distribution implementations to new `_impl.py` module (691 lines)

- **Improved Random State**: Enhanced `RandomState` class with better state management
  - Simplified implementation (reduced from 534 to ~300 lines)
  - Better integration with JAX's random number generation
  - Improved thread safety and state isolation

### Testing

- **Comprehensive Test Suite**: Added 102 tests covering all state functionality
  - **TestBasicState** (13 tests): Core State class operations
  - **TestShortTermState** (2 tests): Short-term state behavior
  - **TestLongTermState** (2 tests): Long-term state behavior
  - **TestParamState** (2 tests): Parameter state usage patterns
  - **TestBatchState** (2 tests): Batch state functionality
  - **TestHiddenState** (7 tests): Hidden state with different array types
  - **TestHiddenGroupState** (9 tests): Multiple hidden state management
  - **TestHiddenTreeState** (12 tests): PyTree hidden states with units
  - **TestFakeState** (4 tests): Lightweight state alternative
  - **TestStateDictManager** (6 tests): State collection management
  - **TestStateTraceStack** (11 tests): State tracing and recovery
  - **TestTreefyState** (6 tests): PyTree state references
  - **TestContextManagers** (6 tests): State context managers
  - **TestStateCatcher** (8 tests): State catching utilities
  - **TestIntegrationScenarios** (5 tests): Real-world use cases

### Bug Fixes

- Fixed `HiddenGroupState.set_value()` to work correctly with JAX arrays
- Improved error handling in hidden state value validation
- Enhanced type checking for hidden state initialization


### Documentation

#### Tutorial Reorganization

- **Basics Tutorials**: Complete rewrite and expansion
  - `01_getting_started.ipynb`: Enhanced introduction with practical examples
  - `02_state_management.ipynb`: Comprehensive state management guide
  - `03_random_numbers.ipynb`: In-depth random number generation tutorial

- **Neural Networks Tutorials**: Restructured and expanded
  - `01_module_basics.ipynb`: New comprehensive module system guide
  - `02_basic_layers.ipynb`: Enhanced layer documentation with examples
  - `03_activations_normalization.ipynb`: Detailed activation and normalization guide
  - `04_recurrent_networks.ipynb`: New RNN tutorial with practical examples
  - `05_dynamics_systems.ipynb`: New dynamical systems tutorial

- **Examples**: Reorganized and enhanced
  - Renamed `10_image_classification.ipynb` to `01_image_classification.ipynb`
  - Renamed `11_sequence_modeling.ipynb` to `02_sequence_modeling.ipynb`
  - Added `03_brain_inspired_computing.ipynb`: New brain-inspired computing examples
  - Renamed `18_optimization_tricks.ipynb` to `04_optimization_tricks.ipynb`
  - Renamed `19_model_deployment.ipynb` to `05_model_deployment.ipynb`

- **Transforms Tutorials**: Reorganized for better flow
  - `01_jit_compilation.ipynb`: New comprehensive JIT guide
  - `02_automatic_differentiation.ipynb`: Enhanced autodiff tutorial
  - `03_vectorization.ipynb`: Improved vmap/pmap guide
  - `04_loops_conditions.ipynb`: Enhanced control flow guide
  - `05_other_transforms.ipynb`: Other transformation utilities

- **Advanced Tutorials**: Renumbered for clarity
  - `01_graph_operations.ipynb` (formerly `14_graph_operations.ipynb`)
  - `02_mixin_system.ipynb` (formerly `15_mixin_system.ipynb`)
  - `03_typing_system.ipynb` (formerly `16_typing_system.ipynb`)
  - `04_utilities.ipynb` (formerly `17_utilities.ipynb`)

- **Migration Guides**: Updated and simplified
  - `01_migration_from_pytorch.ipynb`: Enhanced PyTorch migration guide
  - Removed outdated BrainPy integration notebook

- **Supplementary**: Reorganized
  - `01_performance_optimization.ipynb`
  - `02_debugging_tips.ipynb`
  - `03_faq.ipynb`: Updated FAQ with new content

#### API Documentation

- Enhanced module documentation in `nn.rst` with 306 line improvements
- Updated `transform.rst` with new transform APIs
- Improved `environ.rst` and `graph.rst` documentation

### Refactoring

- Removed deprecated `eval_shape` module and tests
- Removed deprecated `_random.py` transform module
- Cleaned up unused imports across all modules
- Improved code organization in neural network layers
- Enhanced type hints and docstrings throughout

### Infrastructure

- Added development dependency for tutorial generation
- Updated benchmark scripts for performance testing
- Improved test coverage across transformation modules




## Version 0.2.0

This is a major release with significant refactoring, new features, and comprehensive documentation improvements.

### Breaking Changes

- **Module Deprecations**: Deprecated `brainstate.transform`, `brainstate.transform`, and `brainstate.functional` modules in favor of `brainstate.transform` and `brainstate.nn`
  - Added deprecation proxies to guide users towards replacement modules
  - Updated all documentation and examples to use new module paths

- **State Management**: Replaced `write_back_state_values` with `assign_state_vals_v2` for improved state management

- **Import Path Changes**: Major refactoring of import paths across the codebase
  - Moved initialization references to use `brainstate.nn`
  - Updated random functions to use `brainstate.random`
  - Standardized imports across all modules

- **Type System**: Implemented `JointTypes` and `OneOfTypes` generic aliases to enhance type checking and avoid metaclass conflicts
  - Support for subscript syntax
  - Improved type hints across modules

- **Copyright**: Updated copyright notices to reflect new ownership by BrainX Ecosystem Limited

### New Features

#### Neural Network Components

- **Transposed Convolution Layers**: Complete implementations for upsampling operations
  - `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
  - Support for both channels-first and channels-last data formats via `channel_first` parameter
  - Configurable stride for controllable upsampling factors
  - Grouped transposed convolution support
  - Automatic padding computation for 'SAME' and 'VALID' modes

- **Convolution Enhancements**: Added support for both channels-first and channels-last data formats
  - New `channel_first` boolean parameter (default: `False`)
  - PyTorch-compatible format (e.g., `[B, C, H, W]`) when `channel_first=True`
  - Default JAX-style format (e.g., `[B, H, W, C]`) when `channel_first=False`

- **Padding Layers**: Added padding layers for 1D, 2D, and 3D tensors with various modes

- **Unpooling Layers**: Added `MaxUnpool1d`, `MaxUnpool2d`, and `MaxUnpool3d` with `return_indices` support

- **Gradient Utilities**: Implemented `clip_grad_norm` function for gradient clipping in PyTree structures

- **Embedding Enhancements**:
  - Added `padding_idx`, `max_norm`, and `norm_type` parameters
  - Improved gradient management with new `_contains_tracer` function
  - Optimized max_norm application with accessed mask for scaling

- **BatchNorm Improvements**: Added `feature_axis` and `track_running_stats` parameters

- **LoRA Layer**: Added `in_size` parameter for improved size handling

- **Activation Functions**: Added new activation functions and improved signatures

#### Transform & Compilation

- **StatefulMapping**: Introduced for enhanced state management in vmap transformations

- **Mixin Classes**: Added `Mode`, `JointMode`, `Batching`, and `Training` classes for computation behavior control

- **Bounded Cache**: Implemented thread-safe bounded cache for JAX Jaxpr with:
  - Comprehensive validation
  - Statistics tracking
  - Enhanced error handling

- **Input Validation**: Enhanced input size handling to support numpy integer types

- **Context Parameters**: Update method now accepts additional context parameters for improved environment settings

#### Random & Initialization

- **Dependencies**: Integrated `braintools` for initialization and surrogate gradient functions
  - Updated all initialization references
  - Refactored to use `braintools.surrogate` for spike functions

- **Random Functions**: Replaced `uniform_for_unit` with `jr.uniform` for consistency and performance

#### Utilities & Infrastructure

- **Filter Utilities**: Added comprehensive filter utilities for nested structures

- **Pretty Representation**: Enhanced pretty_pytree module with:
  - Comprehensive documentation
  - Mapping functions
  - JAX integration

- **Error Handling**: Improved state length validation by replacing assertions with `ValueError` exceptions

- **Collective Operations**: Updated function signatures to return target in collective operations

### Documentation

- **Comprehensive Docstrings**: Added detailed NumPy-style docstrings across all modules
  - Full parameter descriptions with types and default values
  - Multiple practical examples in code blocks
  - Comparison sections highlighting differences from PyTorch
  - Mathematical formulas where applicable
  - References to original papers
  - Best practices and use cases

- **New Documentation Pages**:
  - `brainstate.environ` module documentation
  - `brainstate.transform` (renamed from compile.rst)
  - Random number generation module
  - Pretty representation module
  - State management tutorial notebook

- **Enhanced Examples**: Updated documentation examples to use interactive prompts for clarity

- **Module Descriptions**: Enhanced documentation with detailed descriptions, key features, and usage examples

### Testing

- **Comprehensive Test Coverage**: Added extensive test suites for:
  - `_BoundedCache` and `StatefulFunction`
  - `brainstate.mixin` module
  - `brainstate.environ` module (context management, precision settings, callbacks)
  - DeprecatedModule and proxy creation functionality
  - Compatible import module
  - Metrics module
  - Node class and helper functions
  - Activation functions with shape and gradient checks
  - Dropout layers
  - Surrogate gradient functions
  - Filter utilities
  - Struct module
  - Pretty representation

- **Test Framework Updates**: Refactored tests to use `absltest` for better JAX compatibility

### Refactoring

- **File Reorganization**:
  - Renamed `metrics.py` to `_metrics.py`
  - Renamed `_rate_rnns.py` to `_rnns.py`
  - Renamed `_init.py` to `init.py`
  - Reorganized graph module files
  - Cleaned up unused imports and classes

- **Code Quality**:
  - Streamlined imports across all modules
  - Enhanced code formatting and whitespace consistency
  - Removed unnecessary inheritance and unused elements
  - Simplified type annotations
  - Improved method signatures for clarity

- **Neuron & Synapse Classes**: Refactored to use brainpy module and updated initialization methods

- **Base Classes**: Changed base class of `EINet` and `Net` from `DynamicsGroup` to `Module` for consistency

- **Evaluation Functions**: Refactored and updated method names for consistency

### Infrastructure

- **Version Bump**: Updated version to 0.2.0

- **Development Dependencies**: Added `braintools` to development requirements

- **Issue Templates**: Added bug report and feature request templates for improved issue tracking

- **CI/CD**: Refactored CI configurations to update pip installation commands

- **Git Ignore**: Updated to exclude example figures directory and build artifacts

### Bug Fixes

- Enhanced delay handling for multi-dimensional inputs
- Fixed gradient function references
- Improved deprecation handling in tests
- Fixed precision checks in complex number handling


## Version 0.1.0

The first version of the project.


