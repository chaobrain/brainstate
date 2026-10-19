# Spec: jax 0.11.2 compatibility -- Jaxpr DebugInfo

## Problem
On jax 0.11.2 (latest) the full test suite passes, but the `brainstate.transform._ir_*`
utilities construct `Jaxpr` objects without a `DebugInfo`, triggering

    DeprecationWarning: core.Jaxpr is missing a DebugInfo object ... (jax-ml/jax#26480)

JAX states this will become an error in a future release.

## Offending sites
| File | Function | Construction |
|------|----------|--------------|
| `transform/_ir_processing.py` | `eqns_to_jaxpr` | `Jaxpr(...)` without `debug_info` |
| `transform/_ir_processing.py` | `eqns_to_closed_jaxpr` | same |
| `transform/_ir_optim.py` | `common_subexpression_elimination` | `jaxpr.replace(..., debug_info=None)` |
| `transform/_ir_tocode.py` | `constant_fold_jaxpr` | `jaxpr.replace(..., debug_info=None)` (prunes invars) |

## Approach
* `DebugInfo(traced_for, func_src_info, arg_names, result_paths)` and
  `DebugInfo.with_unknown_names()` have the same shape on jax 0.9.0 .. 0.11.2, so no
  version guard is needed.
* Add `make_debug_info(traced_for, name="<unknown>")` to `brainstate/_compatible_import.py`
  returning a `DebugInfo` with unknown (`None`) argument names and result paths, which is
  valid for any number of invars/outvars.
* Freshly built jaxprs (`_ir_processing`) use `make_debug_info(...)`.
* Rewritten jaxprs (`_ir_optim`, `_ir_tocode`) keep the source jaxpr's debug info with
  names dropped: `jaxpr.debug_info.with_unknown_names()`.
* Regression guard: pytest `filterwarnings` turns this specific warning into an error.

## Tests
* No DebugInfo warning from each of the four sites; resulting `debug_info` is populated.
* Rewritten jaxprs keep the original `func_src_info`.
* `make_debug_info` fields; jaxprs built with it pass `jax_enable_checks` count assertions.
* Edge cases: empty jaxpr, constvars without consts, constant folding pruning all invars.

## Out of scope
* flax 0.12.8 import failure on jax 0.11.2 (`jax.experimental.hijax.HiPrimitive`) -- upstream flax.
* `generator treated as a leaf` warning from JAX's own debugger callback under xdist.

## Part 2: raise the minimum jax version to 0.9.0
Verified: the full suite (incl. `--run-slow`) passes on jax 0.9.0 and 0.11.2; jax 0.8.0
has 3 pre-existing `_ir_visualize` literal-node failures.

* Pins: `jax>=0.9.0` in `pyproject.toml` (all extras) and `requirements.txt`.
* CI matrix: drop `0.8.0`.
* AGENTS.md rule 10: JAX >= 0.9.0.
* Remove dead version branches for jax < 0.9.0 in `_compatible_import.py`,
  `transform/_make_jaxpr.py`, `util/_tracers.py`. Keep the `< 0.10.0` / `< 0.11.0` guards.
