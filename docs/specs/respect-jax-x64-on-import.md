# Respect JAX's `jax_enable_x64` when `brainstate` is imported

## Problem

`import brainstate` forces `jax_enable_x64` to `False`. The last line of
`brainstate/environ.py` runs `set(precision=DEFAULT_PRECISION)` (`DEFAULT_PRECISION = 32`)
at module load, and `set()` calls `_set_jax_precision(32)`, which does
`config.update("jax_enable_x64", False)`.

This silently overrides a user who enabled float64 the way JAX documents it:

```
JAX_PLATFORMS=cpu python -c "import jax; jax.config.update('jax_enable_x64', True); import brainstate; print(jax.config.jax_enable_x64)"   # False
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -c "import brainstate, jax; print(jax.config.jax_enable_x64)"   # False
```

The user asked for float64 and gets float32 without any warning.

## Decision

Importing brainstate reads JAX's precision. It does not write it.

1. **Import-time precision.** At import, `environ.py` records
   `_INITIAL_PRECISION = 64 if config.jax_enable_x64 else DEFAULT_PRECISION`. The global
   environment starts at that precision and the JAX config is not touched. Because the value is
   derived from the config, the two always agree, so no `config.update` is needed. A default
   import (x64 off) gives precision 32 and float32 exactly as before.
2. **Explicit settings still win.** `environ.set(precision=...)`, `environ.set_precision(...)`
   and `environ.context(precision=...)` on the global environment still call
   `_set_jax_precision`, so they keep driving `jax_enable_x64` both ways.
3. **`reset()` returns to the import-time precision**, not to a hard-coded 32. For the global
   environment it also re-applies that precision to JAX.
4. **Fresh environments start from the import-time precision.** This covers
   `EnvironmentState.__post_init__`, `reset(env=custom)`, and the `get_precision()` /
   `_get_precision()` fallback used when the `precision` key has been popped.

`DEFAULT_PRECISION` stays a public constant equal to `32`. It is still the precision used when
JAX's x64 mode is off.

### Why `reset()` goes back to the import-time precision

- `reset()` is documented as restoring "the environment to its initial state". After this fix
  the initial state is the import-time precision, so this reading is the literal one.
- Resetting to 32 would bring the bug back through a side door. Test suites and notebooks call
  `reset()` in fixtures, so a run with `JAX_ENABLE_X64=1` would flip x64 off at the first
  `reset()`.
- For a default import, `_INITIAL_PRECISION == 32`, so `reset()` behaves exactly as it does
  today.

### Why `EnvironmentState.__post_init__` changes too

`EnvironmentState` subclasses `threading.local`, so `__init__` (and so `__post_init__`) runs
again the first time each new thread touches `_ENV_STATE`. If the default stayed 32, a worker
thread in an x64 process would report precision 32 and return float32 from `dftype()` while
JAX produces float64 arrays. Using `_INITIAL_PRECISION` keeps worker threads consistent with
the import-time JAX setting. The same default for custom `EnvironmentState()` objects and
`reset(env=custom)` keeps the rule simple: every fresh environment starts at the import-time
precision.

### Behaviour change

Every change applies only when x64 is already on at import time, which is exactly the case
that was broken. When x64 is on at import:

- `jax_enable_x64` stays `True` after `import brainstate` (it was forced to `False`);
- `environ.get_precision()` is 64, and `dftype()`, `ditype()`, `dutype()`, `dctype()` and
  `tolerance()` return their 64-bit types;
- `reset()` restores precision 64 and keeps x64 on;
- new threads, new `EnvironmentState()` objects, and a popped `precision` key fall back to 64.

When x64 is off at import (the JAX default), nothing changes.

### Out of scope

After an explicit `set_precision(64)` in the main thread, a *new* worker thread still starts
at the import-time precision rather than the main thread's current setting. That is how
thread-local settings already behaved, and this fix does not change it.

## Tests

`jax_enable_x64` is process-global, so each scenario runs in a fresh subprocess
(`sys.executable -c ...`, `JAX_PLATFORMS=cpu`, `PYTHONPATH` pointing at the brainstate under
test, `JAX_ENABLE_X64` cleared unless the case sets it). They live in
`brainstate/environ_test.py` (`TestImportRespectsJaxX64`):

- x64 enabled through `jax.config.update` before import: x64 still `True`, precision 64,
  `dftype()` float64, `ditype()` int64, default `jnp` float arrays are float64;
- `JAX_ENABLE_X64=1`: the same;
- `JAX_ENABLE_X64=1` overridden by `jax.config.update(..., False)` before import: precision 32
  and x64 off (the config, not the env var, is the source of truth);
- default import: x64 `False`, precision 32, float32 (unchanged);
- after a 64-bit start, `set(precision=32)` turns x64 off and `set_precision(64)` turns it on
  again;
- after a 64-bit start, `context(precision=32)` turns x64 off inside the block and restores it
  on exit;
- `reset()` after a 64-bit start plus `set(precision=32)`: precision 64, x64 on;
- `reset()` after a default start plus `set(precision=64)`: precision 32, x64 off (unchanged);
- after a 64-bit start, a new thread, a new `EnvironmentState()`, `reset(env=custom)` and a
  popped `precision` key all report 64.

The existing `environ_test.py` suite and the precision-related test files must still pass.

## Plan

1. Add the spec (this file).
2. Add the subprocess tests and confirm that the x64 cases fail on unmodified `main`.
3. In `environ.py`, add `_INITIAL_PRECISION`, use it in `EnvironmentState.__post_init__`,
   `reset()`, `get_precision()` and `_get_precision()`, and remove the module-load
   `set(precision=DEFAULT_PRECISION)`. `_ENV_STATE = EnvironmentState()` already starts at
   `_INITIAL_PRECISION` through `__post_init__`, and the JAX config is left alone.
4. Update the `reset()` docstring and add a changelog entry.
5. Run `environ_test.py` and the precision-related test files.
