# Design: Hardening the `brainstate/transform/_ir*` Modules

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Scope:** `brainstate/transform/_ir_processing.py`, `_ir_inline.py`, `_ir_optim.py`, `_ir_tocode.py`, `_ir_visualize.py` and their tests.

## 1. Goal

Improve the robustness, correctness, functionality, and test coverage of the IR
(JAX `Jaxpr`) utilities under `brainstate/transform/`. Fix confirmed bugs,
harden against malformed inputs and edge cases, expand primitive / control-flow
coverage in the code-generation and visualization modules, and add a
comprehensive test suite. Public function signatures are preserved.

## 2. Constraints & Decisions

These were agreed during brainstorming and govern every choice below:

1. **Scope:** All five `_ir*` modules.
2. **API stability:** Public function signatures are preserved. New behavior is
   added only through internal helpers or by exposing already-existing internals
   (e.g. `register_prim_handler`) — never by changing existing signatures.
3. **Correctness over bug-for-bug compatibility:** Incorrect behavior is fixed
   even when the observable output changes. Every such change is recorded in the
   Behavior-Change Log (Section 11).
4. **Broad coverage expansion** for `_ir_tocode` and `_ir_visualize`: add
   handlers for many more primitives and deepen nested control-flow support, in
   addition to fixing bugs. Genuinely unsupported constructs must fail loudly
   with a clear error rather than emit wrong code or crash opaquely.
5. **Sequencing:** Module-by-module in dependency order, TDD within each module
   (write the failing test that pins the bug/edge case, then fix).
6. **Tests:** `unittest.TestCase` classes (the dominant convention — 14 of 17
   transform test files), executed under `pytest`.

**Environment:** JAX 0.10.1 installed; declared minimum `jax>=0.6.0`. The modules
use internal JAX APIs (`jax._src.core.JaxprEqnContext`,
`jax.extend.source_info_util`, `jax.extend.core` primitives), so version-fragile
imports are isolated and guarded.

**Baseline:** 69 existing IR tests pass before any change.

## 3. Architecture

### 3.1 Module layout (after)

```
_ir_utils.py          NEW — shared, hardened internals (no public API)
_ir_processing.py     eqns_to_jaxpr, eqns_to_closed_jaxpr
_ir_inline.py         inline_jit
_ir_optim.py          constant_fold, dead_code_elimination,
                      common_subexpression_elimination, copy_propagation,
                      algebraic_simplification, optimize_jaxpr
_ir_tocode.py         fn_to_python_code, jaxpr_to_python_code,
                      register_prim_handler (newly public)
_ir_visualize.py      draw, view_pydot, draw_dot_graph
```

### 3.2 `_ir_utils.py` — minimal shared layer

This is the only structural change. It consolidates logic currently **duplicated**
between `_ir_optim` and `_ir_tocode` so each bug is fixed once. Contents:

- **Identity collections:** `IdentitySet`, `IdentityMap` — today duplicated in
  `_ir_optim.py` and `_ir_tocode.py`. A single hardened implementation; the old
  names remain importable from their original modules for backward compatibility
  (re-exported), since `IdentitySet` has a docstring advertising
  `__module__ = 'brainstate.transform'`.
- **Constant-folding engine:** `partial_eval_jaxpr`, `eval_eqn` — today
  duplicated (`_partial_eval_jaxpr`/`_eval_eqn` in optim,
  `constant_fold_jaxpr`/`partial_eval_jaxpr`/`_eval_eqn` in tocode), each with
  its own gaps. One hardened implementation used by both.
- **Validation helpers:** `ensure_jaxpr(obj)`, `check_unique_outvars(...)`,
  `is_scalar_literal_value(var, value)`, `literal_with_dtype(value, aval)`.
- **Exception hierarchy** (Section 3.3).

Nothing else moves. No optimization passes, primitive handlers, or public
functions relocate. `_ir_processing` and `_ir_inline` may use the validation
helpers but keep all their own logic.

### 3.3 Error-handling conventions (uniform)

New exception hierarchy in `_ir_utils.py`:

- `IRError(Exception)` — base for all IR utility errors.
- `IRValidationError(IRError)` — malformed inputs (non-`Jaxpr` argument,
  `consts` length ≠ `constvars` length, duplicate output variables, etc.).
- `UnsupportedPrimitiveError(IRError)` — raised by `_ir_tocode` / `_ir_visualize`
  when a primitive or control-flow construct genuinely cannot be handled
  (replaces today's silent generation of calls to nonexistent functions, and
  replaces opaque crashes).

Rules applied everywhere:

- Public functions validate inputs up front and raise typed, message-bearing
  errors — not deep `AssertionError`s or `IndexError`s.
- Context-free `assert`s in hot paths become explicit checks raising `IRError`
  subclasses with actionable messages. (Asserts that encode genuine internal
  invariants may remain, but with messages.)
- Bare `except:` clauses are narrowed to specific exception types; nothing is
  swallowed silently. Where a fallback is intentional, it is logged via
  `warnings.warn`.

### 3.4 Determinism & dtype conventions

- All variable inference and any set-derived ordering becomes **deterministic**:
  ordered de-duplication (e.g. `dict.fromkeys`) instead of iterating a `set`.
- Any synthesized literal/constant is created with a value matching its
  `aval.dtype` via `literal_with_dtype` (fixes the `Literal(0, float_aval)`
  dtype-mismatch class of bug).

## 4. `_ir_processing.py`

Public API unchanged: `eqns_to_jaxpr`, `eqns_to_closed_jaxpr`.

### Bugs / correctness

- **P1 — Non-deterministic inference.** `invars`/`constvars`/`outvars` are
  inferred by iterating Python `set`s (`used_vars_set`,
  `produced_vars - consumed_vars`), giving non-deterministic ordering across runs.
  → Replace with insertion-ordered de-duplication so output ordering is stable
  and reproducible.
- **P2 — Outvar inference drops duplicates and consumed outputs.** Inferring
  `outvars = produced_vars - consumed_vars` (a) silently drops a value that is
  both an output and consumed internally, and (b) collapses duplicate outputs.
  → Document the limitation precisely and, where a deterministic
  last-write/produced-order rule is well-defined, preserve produced order. When
  inference is genuinely ambiguous, the docstring directs callers to pass
  `outvars` explicitly. (This is inference of an under-specified quantity, not a
  guarantee; behavior is documented, not silently "fixed".)

### Robustness / validation

- Validate that every provided `invar`/`outvar`/`constvar` is a `Var`.
- `eqns_to_closed_jaxpr`: keep the existing `consts`/`constvars` length check;
  raise `IRValidationError` with the same message content. `IRValidationError`
  **subclasses `ValueError`** (decided in Section 11) so existing
  `except ValueError` callers keep working.
- Handle the empty-equations case explicitly and deterministically.

### Tests (extend existing `_ir_processing_test.py`)

Determinism of inference ordering; duplicate-output handling; var-type
validation errors; consts/constvars mismatch error; round-trip execution
equivalence for inferred vs. explicit vars.

## 5. `_ir_inline.py`

Public API unchanged: `inline_jit`.

### Bugs / correctness

- **I1 — Variable collisions on repeated inlining.** When the same `call_jaxpr`
  object is inlined more than once (e.g. a helper called twice), its inner `Var`
  objects are reused verbatim (`inner_var_mapping[v] = v`), so two inlined copies
  share output variables — producing an invalid jaxpr where one variable is bound
  twice. → Generate **fresh** `Var`s for inner-scope variables per inlining site
  (via a gensym keyed on the original var), so each expansion is independent.
- **I2 — Nested constants ignored.** When `call_jaxpr` is a `ClosedJaxpr` with
  `consts`/`constvars`, inlining drops them, yielding a jaxpr that references
  unbound constvars. → Thread nested consts into the surrounding scope (lift to
  the enclosing `ClosedJaxpr`'s consts, or materialize as literals where scalar).

### Robustness / validation

- Validate input is `Jaxpr`/`ClosedJaxpr`; raise `IRValidationError` otherwise.
- Guard the `should_expand` predicate: a predicate raising or returning a
  non-bool is reported clearly.
- Verify result jaxpr well-formedness (no variable bound twice; all referenced
  vars defined) after inlining, behind a cheap internal check.

### Tests (extend existing `_ir_inline_test.py`, convert to `unittest.TestCase`)

Same helper inlined twice (regression for I1, asserting unique binders and
correct numerics); inlining a `ClosedJaxpr` with consts (I2); predicate that
rejects all / selects none; multi-output jits; nested jits with shared inner
jaxprs; invalid-input errors; numerical round-trip vs. the original function.

## 6. `_ir_optim.py`

Public API unchanged. Uses shared `partial_eval_jaxpr`/`IdentitySet` from
`_ir_utils`.

### Bugs / correctness

- **O1 — `optimize_jaxpr` default is a silent no-op.** `optimizations=None` is
  immediately reassigned to `[]` (line ~801), so the documented "apply all
  optimizations by default" never runs; the later `if optimizations is None`
  block (line ~839) is dead code. → Make `None` mean the documented default
  pipeline `['constant_fold', 'algebraic_simplification', 'copy_propagation',
  'cse', 'dce']`. Remove the dead branch. **(Behavior change — Section 11.)**
- **O2 — Float-unsafe algebraic identities.** `0 * x → 0` is wrong when
  `x ∈ {inf, nan}` (`0*inf = nan`); `0 / x → 0` is wrong when `x = 0`
  (`0/0 = nan`). Decision: **remove `0 / x → 0` entirely** (division by zero is
  unsafe for every dtype), and **keep `0 * x → 0` only when the output dtype is
  integer** (no inf/nan exist there), removing it for floating-point. Keep all
  unconditionally-safe rewrites: `x+0`, `0+x`, `x-0`, `x-x`, `1*x`, `x*1`, `x/1`.
  **(Behavior change.)**
- **O3 — Literal dtype mismatch.** `make_literal(0, outvar.aval)` builds a Python
  `int 0` tagged with a possibly-float aval. → Use `literal_with_dtype` to build
  a value matching `aval.dtype`.
- **O4 — CSE crashes on unhashable params.** `make_key` builds a dict key
  containing `eqn.params` values; primitives with `ndarray`/list/`dict`/jaxpr
  params (e.g. `pjit`, `convert_element_type` with array params) raise
  `TypeError` because the key is unhashable. → Build a robust, hashable param
  signature (canonicalize arrays to `(shape, dtype, bytes)`, jaxprs by identity,
  fall back to skipping CSE for an equation whose params cannot be canonicalized
  rather than crashing).
- **O5 — `constant_fold` executes arbitrary primitives, including `scan`.** It
  calls `primitive.bind(*vals, **params)` on any all-constant equation; the
  `scan` branch falls through to the generic bind. This can be expensive and can
  fold effectful/Random ops. → Reuse the shared, hardened `partial_eval_jaxpr`;
  keep the existing blacklist (`broadcast_in_dim`, `broadcast`) and extend it to
  skip primitives that should not be eagerly executed (effectful ops); document
  that constant folding evaluates pure primitives at trace time.

### Robustness / validation

- `copy_propagation`: the cycle guard in `get_canonical` exists; add a test and
  ensure outvar identity equations are dtype-correct.
- `optimize_jaxpr`: validate `max_iterations >= 1`; keep the existing
  invalid-optimization-name `ValueError`; keep the interface-preservation
  `RuntimeError` guard but compare by identity/count as today.

### Tests (extend existing `_ir_optim_test.py`)

`optimize_jaxpr()` with no args actually reduces equations (regression for O1);
`0*x`/`0/x` are NOT folded for floats and numerics stay correct for inf/nan/0
(O2); folded literals carry the correct dtype (O3); CSE over a jaxpr containing
`pjit`/array-param primitives does not crash and dedups correctly (O4);
constant-folding a jaxpr with `scan` behaves and respects the blacklist (O5);
control-flow-bearing jaxprs (`cond`, `scan`, `while`) pass through all passes
without corruption; semantic-equivalence checks (compile & run optimized vs.
original) on a battery of functions.

## 7. `_ir_tocode.py`

Public API: `fn_to_python_code`, `jaxpr_to_python_code`; additionally
**expose `register_prim_handler`** in `__all__` (it already exists and is the
designed extension point — exposing it is additive, not a signature change).

### Bugs / correctness

- **T1 — Silent wrong-code for unknown primitives.** Unknown primitives fall back
  to `normal_fn(prim.name)`, emitting a call to a function that may not exist
  (e.g. `gather(...)`). → For unmapped primitives, raise
  `UnsupportedPrimitiveError` listing the primitive and a pointer to
  `register_prim_handler`, UNLESS a correct generic mapping exists. Add many
  correct handlers (below) to shrink the unsupported set.
- **T2 — Context-free asserts.** Asserts in `partial_eval`/`_astify_*`
  (e.g. `assert not isinstance(out, Jaxpr)`) → explicit `IRError`s with messages.
- **T3 — Bare `except AttributeError` on `fn.__name__`** silently yields
  `"unknown"`. → Narrow and keep a sensible default, but allow an explicit
  `fn_name` path so callers aren't surprised.
- **T4 — Empty jaxpr.** Generates a function body that may be empty/invalid. →
  Emit a valid function returning its (possibly trivial) outputs.

### Functionality (broad coverage expansion)

Add correct handlers for commonly-emitted primitives currently missing, including
(non-exhaustive, validated case-by-case against round-trip execution):
`integer_pow`, `pow`, `sqrt`, `rsqrt`, `exp`, `log`, `log1p`, `expm1`, `sin`,
`cos`, `tan`, `tanh`, `logistic`, `abs`, `sign`, `floor`, `ceil`, `round`,
`erf`, `rem`, `atan2`, `and`/`or`/`xor`/`not`, `is_finite`, `reduce_max`,
`reduce_min`, `reduce_prod`, `reduce_and`, `reduce_or`, `argmax`, `argmin`,
`cumsum`/`cumprod`/`cummax`, `concatenate`, `pad`, `rev`, `expand_dims`,
`gather`, `scatter`/`scatter_add`, `sort`, `top_k`, `clamp`, `iota`,
`convert_element_type` (typed), `dot_general` (with the existing `einsum` TODO
addressed where tractable), `conv_general_dilated`, and `where`/`select_n`.
Deepen nested control-flow code generation for `cond`, `scan`, `while`, `pjit`,
`closed_call`, `remat2` so nested jaxprs are emitted as nested Python rather than
crashing or inlining incorrectly. Each handler is justified by a round-trip test;
primitives not handled raise `UnsupportedPrimitiveError`.

### Tests (NEW `_ir_tocode_test.py`)

Primary strategy is **execution round-trip**: generate code, `exec` it in a
controlled namespace (with `jax`, `jax.numpy`, `jax.lax` available), call the
generated function, and assert `allclose` against the original. Cover: each
arithmetic/comparison/unary/reduction handler; multiple outputs; literals and
constants; `convert_element_type`; `dot_general`/`reshape`/`broadcast_in_dim`;
each control-flow primitive (`cond`/`scan`/`while`/`pjit`); empty jaxpr;
unsupported-primitive error path; `register_prim_handler` extension; stateful
brainstate function end-to-end.

## 8. `_ir_visualize.py`

Public API unchanged: `draw`, `view_pydot`, `draw_dot_graph`. All behavior is
gated on optional `pydot`; this is preserved.

### Bugs / correctness

- **V1 — `is_literal` used unbound/stale (line ~421).** In `get_conditional`'s
  function-node branch, `is_literal` is referenced but only assigned in unrelated
  earlier code paths within the same function, so it is either `UnboundLocalError`
  or a stale value from a previous loop iteration. → Compute the correct
  per-edge literal-ness (mirroring the `var_is_literal`/`parent_is_literal`
  pattern used in the scan handler) and use it locally.
- **V2 — `draw_dot_graph` crashes on empty / multi-eqn top level.** It indexes
  `fn.eqns[0]` unconditionally — `IndexError` on an empty jaxpr, and it only
  visualizes the first equation, silently ignoring the rest when the top-level
  jaxpr has multiple equations. → Handle empty jaxpr (emit an empty/placeholder
  graph or raise `IRValidationError` with a clear message — chosen: empty graph
  with the invars/outvars shown), and iterate all top-level equations rather than
  only the first.
- **V3 — Return-arity inconsistencies.** `get_while_branch` and related helpers
  return tuples whose arity does not match the `sub_graph_return` 5-tuple
  contract used elsewhere, risking unpack errors on the `while` path. → Normalize
  all sub-graph helpers to a single documented return contract.
- **V4 — Underscore-variable filtering is unexplained (4× TODO).** Variables
  whose `str(var)` ends in `_` are skipped with a "what does this mean?" comment.
  → Investigate (these are JAX drop/`_` vars), document the rule precisely, and
  apply it consistently (or remove if incorrect).

### Functionality (broad coverage expansion)

Extend `is_not_primitive` / dispatch so control-flow beyond `cond`/`scan`/`while`/
`pjit` is recognized where it exists (`closed_call`, `remat2`, and any
`*_p`-with-jaxpr primitive), and ensure nested structures get unique node IDs
(prevent ID collisions across repeated sub-jaxprs by incorporating the counter
into IDs). Unsupported constructs raise `UnsupportedPrimitiveError` rather than
producing a malformed graph.

### Tests (NEW `_ir_visualize_test.py`)

All tests `@unittest.skipUnless(pydot_is_installed, ...)`. Structural assertions
(graph builds without error; expected node/edge counts; no `UnboundLocalError`):
simple arithmetic; nested jit/`pjit`; `cond` with a function-node branch
(regression for V1); `scan`; `while` (regression for V3); empty jaxpr (V2);
multi-equation top-level jaxpr (V2); the `NotImplementedError` stubs when pydot
is absent (simulated). A non-pydot smoke test ensures importing the module never
fails when pydot is missing.

## 9. Data flow

All five entry points consume a JAX `Jaxpr`/`ClosedJaxpr` (or a function +
example args that are traced to one) and return either a transformed
`Jaxpr`/`ClosedJaxpr` (`processing`, `inline`, `optim`), a Python source string
(`tocode`), or a `pydot` graph (`visualize`). The shared `_ir_utils` layer sits
beneath `optim` and `tocode` (constant folding, identity collections, validation,
errors) and provides validation/error types to `processing` and `inline`. No
public data structure changes shape.

## 10. Testing strategy (overall)

- **Framework:** `unittest.TestCase`, run under `pytest`. Convert the one
  function-style file (`_ir_inline_test.py`) to a `TestCase` for consistency.
- **TDD:** For every confirmed bug (P1–P2, I1–I2, O1–O5, T1–T4, V1–V4), write a
  failing test first, then fix.
- **Round-trip equivalence** is the backbone for `optim` (compile & run) and
  `tocode` (`exec` & run): the transformed/generated artifact must be
  numerically equivalent to the original on representative inputs, including
  edge inputs (`inf`, `nan`, `0`, integer dtypes).
- **Control-flow battery:** a shared set of small functions exercising `cond`,
  `scan`, `while`, `pjit`, nested combinations — reused across `optim`, `tocode`,
  `visualize` tests.
- **Failure-mode tests:** each new `IRError`/`IRValidationError`/
  `UnsupportedPrimitiveError` path is asserted with `assertRaises`.
- **Optional-dependency hygiene:** `visualize` tests skip cleanly without
  `pydot`; the module imports without `pydot`.
- **Determinism test:** `processing` inference produces identical orderings
  across repeated runs.
- **No regressions:** the existing 69 tests continue to pass (adjusted only where
  a documented behavior change in Section 11 requires it).

## 11. Behavior-Change Log

Each item changes observable output and is intentional:

1. **`optimize_jaxpr(jaxpr)` with default args now optimizes** (was a no-op).
   Any caller relying on the no-op default will now receive an optimized jaxpr.
   Documented in the docstring; the dead `None` branch is removed.
2. **`0 / x → 0` is no longer applied at all**, and **`0 * x → 0` is applied
   only for integer output dtypes** (removed for floating-point). Results
   involving `inf`/`nan`/`0` are now IEEE-correct.
3. **Unknown primitives in `_ir_tocode` now raise `UnsupportedPrimitiveError`**
   instead of emitting a call to a possibly-nonexistent function. Many
   previously-unknown primitives become *supported* (Section 7), so the net
   effect for common code is more success, not less.
4. **Synthesized literals/identity constants carry the correct dtype**, which
   can change the dtype tags in optimized jaxprs (now correct).
5. **Exception types:** validation failures raise `IRValidationError`. To avoid
   breaking `except ValueError`/`except TypeError` callers, `IRValidationError`
   subclasses `ValueError` (and a dedicated type-mismatch path keeps `TypeError`
   where one is currently raised). Net: existing `except` clauses keep working;
   callers can additionally catch the narrower `IRError`.

## 12. Risks & Mitigations

- **Internal JAX API drift.** The modules import from `jax._src.core` and
  `jax.extend`. *Mitigation:* isolate fragile imports (already partly via
  `brainstate._compatible_import`), test against the installed 0.10.1, and keep
  behavior within the declared `jax>=0.6.0` floor where feasible; guard
  optional/changed symbols.
- **`tocode` round-trip exec in tests.** Executing generated code is powerful but
  must run in a constrained namespace. *Mitigation:* a fixed, explicit globals
  dict (`jax`, `jnp`, `lax`, `numpy`) and no access to builtins beyond what the
  generated code needs.
- **CSE param canonicalization completeness.** Some exotic params may not
  canonicalize. *Mitigation:* skip CSE for that single equation (correctness
  preserved) and add a test asserting no crash.
- **Inline gensym correctness.** Fresh-var generation must not collide with
  existing names. *Mitigation:* gensym from a counter seeded above all existing
  var counts; post-inline well-formedness check.
- **Scope size.** Five modules including two large untested ones is substantial.
  *Mitigation:* module-by-module delivery so each is independently reviewable and
  shippable; the implementation plan phases the work accordingly.

## 13. Out of Scope

- Rewriting the architecture of `tocode`/`visualize` beyond the shared-util
  consolidation and the bug/coverage work described.
- Changing any public function signature.
- Performance optimization of the passes themselves (correctness first).
- Supporting every JAX primitive exhaustively — coverage expands broadly, but
  the long tail raises `UnsupportedPrimitiveError` by design.
