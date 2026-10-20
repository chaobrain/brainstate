# Spec: audit fixes (High/Medium bugs)

## Context
A read-only audit of the `brainstate` package found about 50 defects. Many were reproduced with small scripts on JAX 0.12. This plan fixes the ~30 **High and Medium** correctness bugs: wrong numerics, silently dropped side effects, hook loss, thread-unsafe environ, and tracer leaks. The Low items go on a follow-up list at the end.

Decisions already made:
- `environ` settings become process-global.
- `random.pareto` follows numpy/Lomax.
- All work happens on one worktree branch with one commit per area.

Workflow, per AGENTS.md:
1. Create the worktree branch `fix/audit-high-medium`. Nothing is committed to main.
2. Write a spec at `doc/specs/2026-10-audit-fixes.md`, a copy of this plan with the bug table.
3. For each bug, first write a failing regression test in the matching `*_test.py`, then make the fix.
4. Use `brainstate.random` in tests, never `jax.random`.
5. Version-specific code goes in `_compatible_import.py`, using feature detection.

---

## Commit 1 — core state & hooks (`_state.py`, `_state_hook_*.py`, `mixin.py`)

1. **`Mode` equality and hashing** (`mixin.py:999-1031`)
   - Problem: `Batching(2) == Batching(5)` is True and they hash the same, so `jit(static_argnums=mode)` reuses the wrong compiled code.
   - Fix: `__eq__` returns `NotImplemented` when `other` is not a Mode. Equality and hashing use `(type(self), self._key())`.
   - `Batching._key()` returns `(batch_size, batch_axis)`.
   - `JointMode._key()` returns `frozenset(modes)`.
   - The base Mode returns `()`.
   - Keep `is_a`/`has` (`mixin.py:1060`) type-based; they don't depend on `__eq__`.
2. **`update_from_ref` / `TreefyState.to_state` swap out the target's HookManager** (`_state.py:718`, `~2669`)
   - Fix: drop `_hooks_manager` and `_source_info` from the metadata before applying it. `to_state` creates a fresh `HookManager()`.
3. **`State.replace` shares the hook manager; `copy`, `replace` and `copy_from` share the mutable `tag` set** (`_state.py:750-887`)
   - Fix: `replace` gets a fresh `HookManager`. All three copy the tag with `set(tag)`.
4. **Hook errors** (`_state_hook_core.py:142`, `_state_hook_manager.py:382,586`)
   - Problem: `Hook.execute` wraps `HookCancellationError` in `HookExecutionError`, so a cancelled write still goes through, and the original error message is lost.
   - Fix: re-raise `HookCancellationError` unwrapped.
   - `_handle_hook_error` reports `error.__cause__ or error` in its message.
5. **HookManager thread safety** (`_state_hook_manager.py:522-545`)
   - Fix: rebuild the cache while holding `_lock`, with a version counter so a stale rebuild never marks the cache valid.
   - Take a snapshot of the hook list under the lock and run the callbacks outside it, which prevents deadlocks.
6. **`StateTraceStack.__exit__` hides the real exception** (`_state.py:2137`)
   - Fix: when an exception is propagating, remove `self` (and anything above it) from the stack. Raise the "out of order" `RuntimeError` only when `exc_type is None`.

## Commit 2 — environ (`environ.py`)

7. **Make settings process-global** (`environ.py:151-185`)
   - `settings`, `functions` and the locks move to module-level storage guarded by one `threading.RLock`.
   - Only the `contexts` stack stays per-thread in `threading.local`.
   - Update `environ_test.py:863` so it asserts that worker threads see `dt`, instead of swallowing `KeyError`.
8. **`import brainstate` no longer turns off x64** (`environ.py:1826`, `reset()` at `:233`)
   - The initial precision is 64 if `jax.config.jax_enable_x64` is already set, otherwise the default.
   - Import does not write to JAX config.
9. **`context(precision=...)` across threads** (`:376-424`)
   - Precision is process-global, so guard enter and exit with the global lock and a refcount.
   - Restore from the saved global value, not a per-thread one. Document this in the docstring.

## Commit 3 — transform / IR (`transform/`)

10. **DCE removes side-effecting equations** (`_ir_optim.py:198,207`)
    - Fix: always keep equations whose `eqn.effects` is non-empty, and seed their invars as used.
11. **Constant folding executes effectful nested jit at optimize time** (`_ir_utils.py:326`)
    - Fix: skip any equation with effects, including effects inside sub-jaxprs.
12. **`x - x → 0` gives the wrong shape and turns `inf - inf` into 0 instead of NaN** (`_ir_optim.py:535`)
    - Fix: apply the rule only to integer or boolean dtypes, and produce a value of the full output shape (or skip when the output isn't a scalar).
13. **`vmap_new_states` / `vmap2_new_states` leak BatchTracers into pre-existing states** (`_mapping1.py:346`, `_mapping2.py:860`)
    - Fix: record writes to pre-existing states and restore them in a `finally`.
14. **States excluded by `state_to_exclude` keep dead tracers** (`_mapping1.py:344-371`)
    - Fix: invalidate them and unwind their stack level, or return them with `out_axes=None`.
    - Remove the unused outer `catch_new_states`.
15. **Negative `static_argnums` in `StatefulFunction` / `make_jaxpr`** (`_make_jaxpr.py:156,360,1064`)
    - Fix: normalize the indices against `len(args)` at call time, reusing `_mapping_core._normalize_static_argnums`.
16. **`jit(static_argnames=...)` ignores a static argument passed positionally; `donate_argnames` has the same bug** (`_jit.py:96-115`)
    - Fix: resolve argnums and argnames from `inspect.signature` once, at construction, the way `jax.api_util.infer_argnums_and_argnames` does.
    - If that function is private or unstable, add a wrapper in `_compatible_import.py`.
17. **States with mixed batched and unbatched leaves are rejected** (`_mapping_core.py:765`)
    - Fix: ignore `None` dims when at least one leaf is batched.
18. **Generated Python code for `scan` with consts is broken** (`_ir_tocode.py`)
    - Pass `carries + xs` in the simple branch instead of `eqn.invars` (`:1245`).
    - Only fold equations whose inputs are concrete values, not `Var`/`Literal` (`:715`).
    - Keep sub-jaxpr invars when partial evaluation prunes unused ones (`:762`).
    - Make the existing test at `_ir_tocode_test.py:680` actually run the generated code.

## Commit 4 — nn (`nn/`)

19. **`BatchNorm` with `track_running_stats=False` returns the input unchanged** (`_normalizations.py:469`)
    - Fix: always compute batch statistics with `_compute_stats` and normalize with them.
20. **`BatchNorm` running stats get the wrong shape when `feature_axis` isn't last** (`_normalizations.py:464`)
    - Fix: reshape the batch mean and var to `feature_shape` before the EMA update.
21. **`MaxUnpool` decodes indices against the wrong shape when the input size is odd** (`_poolings.py:731-796`)
    - Fix: unravel the indices against the true input shape. Use `output_size` when it is given; otherwise store the shape on the indices or require it.
22. **`MaxPool(return_indices=True, in_size=...)` crashes in the constructor** (`_poolings.py:257`)
    - Fix: take `y[0]` when `return_indices` is set.
23. **`Delay` gets the delay step wrong because of `ceil` on a float ratio** (`_delay.py:359`)
    - Fix: round when the ratio is within 1e-6 of an integer, otherwise use `ceil`.
24. **`Delay.update_every_step` truncates with `int()`** (`_delay.py:567`)
    - Fix: use `int(round(...))` and check that the ratio is close to an integer.
25. **`init_maybe_prefetch(PrefetchDelay)` raises `TypeError`** (`_dynamics.py:823`)
    - Fix: pass `update_every` to `_get_prefetch_delay_key`, and flip the test at `_dynamics_test.py:456`.

## Commit 5 — random (`random/`)

26. **`pareto` returns Pareto-I instead of Lomax** (`_state.py:640`)
    - Fix: subtract 1 from the result. Add a changelog entry.
27. **`multinomial` drops draws when `pvals` doesn't sum to 1** (`_impl.py:38-40,109-113`)
    - Fix: set the last probability to `1 - sum(rest)`, as numpy does, and clip the category index to K-1.
28. **`truncated_normal` doesn't normalize `size`; `zipf` has the same problem** (`_state.py:803`, `:1508`)
    - Fix: apply `_size2shape` to `size` in both, and `to_size` in `init.TruncatedNormal`.

---

## Verification
- For each fix, the new regression test fails before the change and passes after. Run the tests for each area:
  - `pytest brainstate/_state_test.py brainstate/_state_hook*_test.py brainstate/mixin_test.py`
  - `pytest brainstate/environ_test.py`
  - `pytest brainstate/transform`
  - `pytest brainstate/nn`
  - `pytest brainstate/random`
- Run the full suite, `pytest brainstate -x -q`. Check coverage on the changed modules (> 90%) with `pytest --cov=brainstate`.
- Run the audit's reproducer snippets again:
  - `jit` with `Batching(2)` vs `Batching(5)`
  - a hook round-trip through `update_from_ref`
  - DCE keeping `debug_print`
  - BatchNorm with `track_running_stats=False` actually normalizing
  - `MaxUnpool` on odd sizes against a PyTorch-equivalent reference
  - pareto `min >= 0`
  - multinomial counts summing to n
- Thread test: `environ.set(dt=0.1)` in the main thread, then read `get_dt()` from a `ThreadPoolExecutor` worker.
- Import test: `JAX_ENABLE_X64=1 python -c "import brainstate, jax; assert jax.config.jax_enable_x64"`.
- After implementing, list the edge cases covered and suggest any further tests.

## Follow-up (Low, not in this round)
- Interpolation reads a stale ring slot in Delay.
- Dropout: unit stripping in Dropout1d/2d/3d and NaN gradients at `prob=0`.
- LoRA initializes `lora_b` to nonzero values.
- Linear builds `w_mask` with the wrong shape.
- GroupNorm assert and unbatched input.
- `restore_key` docstring example raises.
- `get_hooks(None)` leaves out init hooks.
- Instance `'init'` hooks can never fire.
- `_utils._deprecate` overwrites warning filters.
- Mode deprecation `__getattr__` can't be reached.
- `ParamDescriber` hashing.
- `_compatible_import` version checks, private APIs and `to_concrete_aval`.
- Copy propagation drops `device_put`.
- `eval_shape` restores dead tracers.
- `while_loop` fallback under `disable_jit`.
- `shard_map` and `checkify` re-trace on every call.
- `closed_call` folding ignores consts.
- Assorted docstring fixes.
