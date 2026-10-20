# JAX 0.12.0 compatibility audit

## Problem

JAX 0.12.0 (2026-10-19) and 0.11.2 (2026-09-17) shipped several breaking changes. The
`""` (latest) leg of the CI matrix now installs 0.12.0, so we need to know whether brainstate
is affected before that leg goes red.

## Method

1. Cross-check the 0.11.2 and 0.12.0 changelog entries against brainstate's source.
2. Run the full suite in an isolated venv with `jax[cpu]==0.12.0` and the CI dependency
   set (`requirements-dev.txt`). Repeat with `--run-slow` and
   `XLA_FLAGS=--xla_force_host_platform_device_count=8` to cover the multi-device tests.
3. Probe the changes the suite might not cover, and compare against a jax 0.11.1 venv.

## Findings

**No incompatibilities.** On jax 0.12.0, `import brainstate` is clean under
`-W error::DeprecationWarning`, and the suite passes: 5314 passed / 75 skipped by default,
5333 passed / 56 skipped with slow + 8 devices. No `brainstate/` change is needed.

| JAX change | brainstate exposure | Result |
|---|---|---|
| `jax.custom_remat` removed | not used | n/a |
| `jnp.atleast_*d` return tuples | not used | n/a |
| `jax.random.multinomial` `shape` is now the batch shape | `random/_impl.py` has its own implementation | shapes match NumPy for `size=None/int/tuple` |
| `custom_vjp` under `vmap` errors when the cotangent of an unbatched primal would be counted per example | `nn/_embedding.py` `_lookup` | `vmap(grad)` with weight unbatched/batched and `grad(vmap)` agree with per-example grads, for all three lookup variants |
| Duplicate / closed-over-and-passed `Ref` errors in `checkpoint`/`custom_vjp`/`shard_map` | brainstate does not pass jax `Ref`s; `State` values are threaded as arrays | `transform.remat` that reads the same State twice is fine; shard_map tests pass with 8 devices |
| `jit(inline=True)` now means `Inline.JAX_LATE` (0.11.2) | `transform/_jit.py` passes `inline` through | nested inlined jit is correct; the docstring doesn't promise early/late semantics |
| `jnp.array` `copy/order/ndmin` keyword-only; 2D `jnp.cross` removed; `shard_alike`, `psend/precv`, `ProfileOptions` | not used | n/a |
| NumPy >= 2.2 required | jax's own requirement takes care of it | n/a |
| Private `jax._src` imports (batching tracers, `trace_ctx`, `UNSPECIFIED`, `flatten_axes`, `TypedNdArray`, ...) | `_compatible_import.py` and others | all import and work |
| Optional interop frameworks | flax 0.12.10, equinox 0.13.8, optax 0.2.8 | interop tests run (not skipped) and pass |

## Follow-ups (not 0.12 regressions)

- `transform/_debug_test.py` triggers jax's `Python iterable type 'generator' is treated as a
  leaf in PyTree` DeprecationWarning (~30 times). It is present on 0.11.1 too. JAX says this
  will become an error in a future release. The warning comes from jaxlib's C++ flatten
  inside a callback path, and turning it into an error does not fail any test, so its exact
  source is still unidentified.

## CI

Added `"0.12.0"` to the `jax-version` matrix in `.github/workflows/CI.yml`, so 0.12 stays
tested after `""` moves on to 0.13.
