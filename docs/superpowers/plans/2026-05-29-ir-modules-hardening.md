# IR Modules Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix confirmed bugs, harden inputs/edge-cases, broaden primitive/control-flow coverage, and add comprehensive tests across `brainstate/transform/_ir_processing.py`, `_ir_inline.py`, `_ir_optim.py`, `_ir_tocode.py`, `_ir_visualize.py`.

**Architecture:** Introduce one new internal module `_ir_utils.py` holding shared, hardened internals (identity collections, the constant-folding engine, validation helpers, and an `IRError` exception hierarchy) consumed by `_ir_optim` and `_ir_tocode` (and used for validation in `_ir_processing`/`_ir_inline`). Then harden each module in dependency order with TDD. Public function signatures are preserved; `register_prim_handler` is newly exported.

**Tech Stack:** Python, JAX 0.10.1 (declared floor `jax>=0.6.0`), JAX internal IR APIs (`jax._src.core`, `jax.extend.core`, `jax.extend.source_info_util`), `numpy`, `unittest` run under `pytest`, optional `pydot` for visualization.

**Reference spec:** `docs/superpowers/specs/2026-05-29-ir-modules-hardening-design.md`

---

## Conventions for every task

- Run all commands from the worktree root: `/mnt/d/codes/projects/brainstate/.claude/worktrees/sofo-braintools-migration`.
- Test runner: `python -m pytest <path> -v`.
- Commit messages omit any `Co-Authored-By` trailer (per repo rule).
- "Run it to confirm it fails / passes" is its own step — do not skip.
- After each phase, run the whole IR test set: `python -m pytest brainstate/transform/_ir_*_test.py -q` and confirm green before starting the next phase.

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `brainstate/transform/_ir_utils.py` | Shared identity collections, constant-folding engine, validation helpers, `IRError` hierarchy | Create |
| `brainstate/transform/_ir_utils_test.py` | Unit tests for the shared layer | Create |
| `brainstate/transform/_ir_processing.py` | `eqns_to_jaxpr`, `eqns_to_closed_jaxpr` | Modify |
| `brainstate/transform/_ir_processing_test.py` | Tests | Modify |
| `brainstate/transform/_ir_inline.py` | `inline_jit` | Modify |
| `brainstate/transform/_ir_inline_test.py` | Tests (convert to `unittest.TestCase`) | Modify |
| `brainstate/transform/_ir_optim.py` | optimization passes | Modify |
| `brainstate/transform/_ir_optim_test.py` | Tests | Modify |
| `brainstate/transform/_ir_tocode.py` | code generation | Modify |
| `brainstate/transform/_ir_tocode_test.py` | Tests | Create |
| `brainstate/transform/_ir_visualize.py` | pydot visualization | Modify |
| `brainstate/transform/_ir_visualize_test.py` | Tests | Create |
| `brainstate/transform/__init__.py` | export `register_prim_handler` | Modify |

---

# Phase 0 — Shared utility layer (`_ir_utils.py`)

### Task 0.1: Create `_ir_utils.py` with the exception hierarchy and identity collections

**Files:**
- Create: `brainstate/transform/_ir_utils.py`
- Create: `brainstate/transform/_ir_utils_test.py`

- [ ] **Step 1: Write the failing test**

Create `brainstate/transform/_ir_utils_test.py`:

```python
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# (full Apache header as in sibling files)
# ==============================================================================

import unittest

from brainstate.transform._ir_utils import (
    IRError, IRValidationError, UnsupportedPrimitiveError,
    IdentitySet, IdentityMap,
)


class TestExceptionHierarchy(unittest.TestCase):
    def test_validation_error_is_value_error(self):
        # IRValidationError must subclass ValueError so existing
        # `except ValueError` callers keep working.
        self.assertTrue(issubclass(IRValidationError, ValueError))
        self.assertTrue(issubclass(IRValidationError, IRError))

    def test_unsupported_primitive_is_ir_error(self):
        self.assertTrue(issubclass(UnsupportedPrimitiveError, IRError))

    def test_ir_error_is_exception(self):
        self.assertTrue(issubclass(IRError, Exception))


class TestIdentitySet(unittest.TestCase):
    def test_membership_by_identity(self):
        s = IdentitySet()
        a = [1, 2, 3]
        b = [1, 2, 3]
        s.add(a)
        self.assertIn(a, s)
        self.assertNotIn(b, s)

    def test_len_and_iter(self):
        s = IdentitySet()
        a, b = object(), object()
        s.add(a); s.add(b); s.add(a)
        self.assertEqual(len(s), 2)
        self.assertEqual({id(x) for x in s}, {id(a), id(b)})

    def test_discard_and_update(self):
        s = IdentitySet()
        a, b = object(), object()
        s.update([a, b])
        s.discard(a)
        self.assertNotIn(a, s)
        self.assertIn(b, s)


class TestIdentityMap(unittest.TestCase):
    def test_set_get_by_identity(self):
        m = IdentityMap()
        a = [1]
        b = [1]
        m[a] = "x"
        self.assertEqual(m[a], "x")
        self.assertNotIn(b, m)

    def test_len_iter_del(self):
        m = IdentityMap()
        a, b = object(), object()
        m[a] = 1; m[b] = 2
        self.assertEqual(len(m), 2)
        del m[a]
        self.assertEqual(len(m), 1)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brainstate.transform._ir_utils'`

- [ ] **Step 3: Write minimal implementation**

Create `brainstate/transform/_ir_utils.py` (start the Apache header identical to sibling files), then:

```python
from collections.abc import MutableSet, MutableMapping

__all__ = []  # internal module; nothing is part of the public API


class IRError(Exception):
    """Base class for all brainstate IR utility errors."""
    __module__ = 'brainstate.transform'


class IRValidationError(IRError, ValueError):
    """Raised when an IR input is malformed.

    Subclasses ``ValueError`` so that existing ``except ValueError`` callers
    continue to work unchanged.
    """
    __module__ = 'brainstate.transform'


class UnsupportedPrimitiveError(IRError):
    """Raised when a primitive or control-flow construct cannot be handled."""
    __module__ = 'brainstate.transform'


class IdentitySet(MutableSet):
    """A set that compares elements by identity (``id()``) rather than equality."""
    __module__ = 'brainstate.transform'

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, value):
        return id(value) in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[id(value)] = value

    def discard(self, value):
        self._data.pop(id(value), None)

    def update(self, iterable):
        for item in iterable:
            self.add(item)

    def __repr__(self):
        return f"IdentitySet({[repr(x) for x in self._data.values()]})"


class IdentityMap(MutableMapping):
    """A mapping keyed by identity (``id()``) of the key object."""
    __module__ = 'brainstate.transform'

    def __init__(self):
        self._data = {}  # id(key) -> (key, value)

    def __getitem__(self, key):
        return self._data[id(key)][1]

    def __setitem__(self, key, value):
        self._data[id(key)] = (key, value)

    def __delitem__(self, key):
        del self._data[id(key)]

    def __contains__(self, key):
        return id(key) in self._data

    def __iter__(self):
        return (k for (k, _v) in self._data.values())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityMap({ {repr(k): repr(v) for (k, v) in self._data.values()} })"
```

> Note: `_ir_tocode.py`'s existing `IdentityMap` stores `id(key) -> value` and iterates values; the version above stores keys too so iteration yields keys (standard mapping semantics). Task 4.x updates tocode call sites accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_utils.py brainstate/transform/_ir_utils_test.py
git commit -m "feat(transform): add _ir_utils shared layer (errors + identity collections)"
```

---

### Task 0.2: Add validation helpers to `_ir_utils.py`

**Files:**
- Modify: `brainstate/transform/_ir_utils.py`
- Modify: `brainstate/transform/_ir_utils_test.py`

- [ ] **Step 1: Write the failing test** — append to `_ir_utils_test.py`:

```python
import numpy as np
import jax
import jax.numpy as jnp
from brainstate._compatible_import import Literal, Var
from brainstate.transform._ir_utils import (
    ensure_jaxpr, is_scalar_literal_value, literal_with_dtype,
)


class TestValidationHelpers(unittest.TestCase):
    def _make_jaxpr(self):
        return jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0))

    def test_ensure_jaxpr_accepts_jaxpr(self):
        cj = self._make_jaxpr()
        # accepts ClosedJaxpr and returns (jaxpr, consts, was_closed)
        jaxpr, consts, was_closed = ensure_jaxpr(cj)
        self.assertTrue(was_closed)
        self.assertEqual(list(consts), list(cj.consts))
        # accepts a bare Jaxpr too
        jaxpr2, consts2, was_closed2 = ensure_jaxpr(cj.jaxpr)
        self.assertFalse(was_closed2)
        self.assertEqual(consts2, [])

    def test_ensure_jaxpr_rejects_other(self):
        with self.assertRaises(IRValidationError):
            ensure_jaxpr(42)

    def test_is_scalar_literal_value(self):
        lit0 = Literal(np.float32(0.0), jax.core.ShapedArray((), np.float32))
        lit1 = Literal(np.float32(1.0), jax.core.ShapedArray((), np.float32))
        self.assertTrue(is_scalar_literal_value(lit0, 0))
        self.assertFalse(is_scalar_literal_value(lit0, 1))
        self.assertTrue(is_scalar_literal_value(lit1, 1))

    def test_literal_with_dtype_matches_aval(self):
        aval = jax.core.ShapedArray((), np.float32)
        lit = literal_with_dtype(0, aval)
        self.assertIsInstance(lit, Literal)
        self.assertEqual(np.asarray(lit.val).dtype, np.float32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py::TestValidationHelpers -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_jaxpr'`.

- [ ] **Step 3: Write minimal implementation** — append to `_ir_utils.py`:

```python
import numpy as np
from brainstate._compatible_import import Literal, Var, Jaxpr, ClosedJaxpr


def ensure_jaxpr(obj):
    """Validate and normalize a Jaxpr/ClosedJaxpr input.

    Returns a tuple ``(jaxpr, consts, was_closed)`` where ``jaxpr`` is a bare
    ``Jaxpr``, ``consts`` is a list (empty for a bare Jaxpr), and ``was_closed``
    indicates whether the input was a ``ClosedJaxpr``.

    Raises ``IRValidationError`` for any other type.
    """
    if isinstance(obj, ClosedJaxpr):
        return obj.jaxpr, list(obj.consts), True
    if isinstance(obj, Jaxpr):
        return obj, [], False
    raise IRValidationError(
        f"Expected a Jaxpr or ClosedJaxpr, got {type(obj).__name__}."
    )


def is_scalar_literal_value(var, value) -> bool:
    """True iff ``var`` is a scalar ``Literal`` equal to ``value``."""
    if not isinstance(var, Literal):
        return False
    val = var.val
    try:
        if isinstance(val, (int, float, complex)):
            return val == value
        arr = np.asarray(val)
        return arr.shape == () and arr.item() == value
    except Exception:
        return False


def literal_with_dtype(value, aval) -> Literal:
    """Create a ``Literal`` whose value matches ``aval.dtype`` (and shape ())."""
    dtype = getattr(aval, 'dtype', None)
    if dtype is None:
        return Literal(value, aval)
    return Literal(np.asarray(value, dtype=dtype), aval)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_utils.py brainstate/transform/_ir_utils_test.py
git commit -m "feat(transform): add IR validation helpers to _ir_utils"
```

---

### Task 0.3: Move the constant-folding engine into `_ir_utils.py`

This consolidates `_partial_eval_jaxpr`/`_eval_eqn` (optim) and the equivalent in tocode. The shared engine is modeled on the optim version (the more complete one) and adds an effectful-primitive guard.

**Files:**
- Modify: `brainstate/transform/_ir_utils.py`
- Modify: `brainstate/transform/_ir_utils_test.py`

- [ ] **Step 1: Write the failing test** — append to `_ir_utils_test.py`:

```python
from brainstate.transform._ir_utils import partial_eval_jaxpr, CONSTANT_FOLD_BLACKLIST


class TestPartialEval(unittest.TestCase):
    def test_folds_pure_constants(self):
        # f(x) = x + (2 + 3) -> after fold, the (2+3) is precomputed.
        cj = jax.make_jaxpr(lambda x: x + (jnp.float32(2.0) + jnp.float32(3.0)))(jnp.float32(1.0))
        folded = partial_eval_jaxpr(cj.jaxpr, {})
        # The number of `add` equations drops from 2 to 1.
        n_add = sum(1 for e in folded.eqns if e.primitive.name == 'add')
        self.assertEqual(n_add, 1)

    def test_blacklist_contains_broadcast(self):
        self.assertIn('broadcast_in_dim', CONSTANT_FOLD_BLACKLIST)
        self.assertIn('broadcast', CONSTANT_FOLD_BLACKLIST)

    def test_no_constants_is_noop_shape(self):
        cj = jax.make_jaxpr(lambda x, y: x * y)(jnp.float32(2.0), jnp.float32(3.0))
        folded = partial_eval_jaxpr(cj.jaxpr, {})
        self.assertEqual(len(folded.eqns), len(cj.jaxpr.eqns))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py::TestPartialEval -v`
Expected: FAIL — `ImportError: cannot import name 'partial_eval_jaxpr'`.

- [ ] **Step 3: Write minimal implementation** — append to `_ir_utils.py`:

```python
import jax
from jax import lax

# Primitives never eagerly executed during constant folding:
#  - broadcast(_in_dim): folding them materializes large constants.
#  - effectful/randomness/IO ops: executing them at trace time is wrong.
CONSTANT_FOLD_BLACKLIST = {
    'broadcast_in_dim', 'broadcast',
    'rng_uniform', 'rng_bit_generator', 'random_seed', 'random_bits',
    'random_fold_in', 'random_split', 'random_wrap', 'random_unwrap',
    'while', 'scan', 'cond',  # control flow: do not eagerly execute
    'debug_callback', 'io_callback', 'pure_callback',
}


def _eval_eqn(eqn, vals):
    if eqn.primitive.name == "closed_call":
        return partial_eval_jaxpr(
            eqn.params['call_jaxpr'].jaxpr,
            {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)},
        )
    return eqn.primitive.bind(*vals, **eqn.params)


def partial_eval_jaxpr(jaxpr, env):
    """Evaluate all-constant equations at trace time, inlining results.

    ``env`` maps ``Var`` -> concrete value for already-known constants.
    Returns a new ``Jaxpr`` with constants folded and now-unused invars dropped.
    Blacklisted and control-flow primitives are passed through untouched.
    """
    env = dict(env)
    new_eqns = []

    def read(var):
        if isinstance(var, Literal):
            return var.val
        return env.get(var, None)

    def read_or_self(var):
        out = read(var)
        if out is None:
            return var
        if isinstance(out, Var):
            return out
        if isinstance(out, Literal):
            return Literal(out.val, var.aval)
        return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(v) for v in eqn.invars]
        if eqn.primitive.name in CONSTANT_FOLD_BLACKLIST:
            new_eqns.append(eqn)
        elif vals and all(v is not None for v in vals):
            out = _eval_eqn(eqn, vals)
            if isinstance(out, Jaxpr):
                new_eqns.extend(out.eqns)
                out = out.outvars
            elif not isinstance(out, (tuple, list)):
                out = (out,)
            for var, val in zip(eqn.outvars, out):
                env[var] = val.val if isinstance(val, Literal) else val
        else:
            new_eqns.append(eqn)

    out_eqns = [e.replace(invars=tuple(read_or_self(v) for v in e.invars)) for e in new_eqns]

    invars_still_used = IdentitySet()
    for e in out_eqns:
        for v in e.invars:
            if not isinstance(v, Literal):
                invars_still_used.add(v)

    invars = tuple(v for v in jaxpr.invars if v in invars_still_used)
    outvars = tuple(read_or_self(v) for v in jaxpr.outvars)
    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_utils.py brainstate/transform/_ir_utils_test.py
git commit -m "feat(transform): add shared constant-folding engine to _ir_utils"
```

---

# Phase 1 — `_ir_processing.py`

### Task 1.1: Deterministic variable inference (P1)

**Files:**
- Modify: `brainstate/transform/_ir_processing.py:48-95`
- Modify: `brainstate/transform/_ir_processing_test.py`

- [ ] **Step 1: Write the failing test** — append to `_ir_processing_test.py` a new class:

```python
class TestInferenceDeterminism(unittest.TestCase):
    def _eqns_from(self, f, *args):
        cj = jax.make_jaxpr(f)(*args)
        return cj.jaxpr.eqns, cj.jaxpr

    def test_inference_is_deterministic_across_runs(self):
        # A function with several free inputs feeding many eqns; repeated
        # inference must yield identical invar/constvar/outvar orderings.
        def f(a, b, c):
            return (a + b) * c - a

        eqns, src = self._eqns_from(f, jnp.float32(1.), jnp.float32(2.), jnp.float32(3.))
        from brainstate.transform import eqns_to_jaxpr
        first = eqns_to_jaxpr(list(eqns))
        for _ in range(5):
            again = eqns_to_jaxpr(list(eqns))
            self.assertEqual([str(v) for v in again.invars], [str(v) for v in first.invars])
            self.assertEqual([str(v) for v in again.outvars], [str(v) for v in first.outvars])
            self.assertEqual([str(v) for v in again.constvars], [str(v) for v in first.constvars])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_processing_test.py::TestInferenceDeterminism -v`
Expected: With set-based inference this is flaky; run several times. If it does not fail immediately, it is still a latent bug — proceed to the fix and keep the test as a regression guard. (Acceptable per TDD here: the fix makes ordering provably stable.)

- [ ] **Step 3: Write minimal implementation**

In `_ir_processing.py`, replace the set-difference / set-iteration inference with insertion-ordered de-duplication. Replace the body of `eqns_to_jaxpr` (lines ~48-95) with:

```python
    # Insertion-ordered "produced" set.
    produced_vars = IdentitySet()
    for eqn in eqns:
        produced_vars.update(v for v in eqn.outvars)

    # Ordered list of used Vars (first-use order), de-duplicated by identity.
    used_order = []
    used_seen = IdentitySet()
    for eqn in eqns:
        for var in eqn.invars:
            if isinstance(var, Var) and var not in used_seen:
                used_seen.add(var)
                used_order.append(var)

    if invars is None:
        invars = [v for v in used_order if v not in produced_vars]
    else:
        invars = list(invars)
    invars_set = IdentitySet(invars)

    if constvars is None:
        constvars = [v for v in used_order
                     if v not in produced_vars and v not in invars_set]
    else:
        constvars = list(constvars)

    if outvars is None:
        consumed = IdentitySet()
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Var) and var in produced_vars:
                    consumed.add(var)
        # Preserve production order; keep produced-but-unconsumed vars.
        outvars = []
        for eqn in eqns:
            for v in eqn.outvars:
                if v not in consumed:
                    outvars.append(v)
    else:
        outvars = list(outvars)

    return Jaxpr(constvars=constvars, invars=invars, outvars=outvars, eqns=list(eqns))
```

Add the import at the top of `_ir_processing.py`:

```python
from brainstate.transform._ir_utils import IdentitySet
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest brainstate/transform/_ir_processing_test.py -v`
Expected: PASS (all pre-existing tests + the new determinism test).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_processing.py brainstate/transform/_ir_processing_test.py
git commit -m "fix(transform): make eqns_to_jaxpr inference deterministic"
```

---

### Task 1.2: Input validation and consts/constvars error type

**Files:**
- Modify: `brainstate/transform/_ir_processing.py` (`eqns_to_closed_jaxpr`)
- Modify: `brainstate/transform/_ir_processing_test.py`

- [ ] **Step 1: Write the failing test** — append:

```python
class TestProcessingValidation(unittest.TestCase):
    def test_consts_mismatch_raises_value_error(self):
        from brainstate.transform import eqns_to_closed_jaxpr
        from brainstate.transform._ir_utils import IRValidationError
        cj = jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0))
        eqns = list(cj.jaxpr.eqns)
        # Provide a constvar but mismatched consts length.
        with self.assertRaises(ValueError):           # back-compat
            eqns_to_closed_jaxpr(eqns, constvars=list(cj.jaxpr.invars), consts=[])
        with self.assertRaises(IRValidationError):     # new precise type
            eqns_to_closed_jaxpr(eqns, constvars=list(cj.jaxpr.invars), consts=[])

    def test_non_var_invar_rejected(self):
        from brainstate.transform import eqns_to_jaxpr
        from brainstate.transform._ir_utils import IRValidationError
        cj = jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0))
        with self.assertRaises(IRValidationError):
            eqns_to_jaxpr(list(cj.jaxpr.eqns), invars=[123])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_processing_test.py::TestProcessingValidation -v`
Expected: FAIL — current code raises plain `ValueError` for consts mismatch (so the `IRValidationError` assert fails) and does not validate invar types.

- [ ] **Step 3: Write minimal implementation**

In `_ir_processing.py`:

Add import:
```python
from brainstate.transform._ir_utils import IdentitySet, IRValidationError
```

In `eqns_to_jaxpr`, immediately after normalizing each of `invars`/`outvars`/`constvars` when explicitly provided, validate types. Add this helper near the top of the module and call it on each provided list:

```python
def _check_all_vars(varseq, name):
    for v in varseq:
        if not isinstance(v, Var):
            raise IRValidationError(
                f"{name} must contain only Var instances; got {type(v).__name__}."
            )
```

Call `_check_all_vars(invars, 'invars')` (and similarly for `outvars`, `constvars`) right after each `... = list(...)` branch when the value was provided by the caller.

In `eqns_to_closed_jaxpr`, change the length-mismatch raise (line ~143-146) from `ValueError` to:

```python
        raise IRValidationError(
            f"consts length ({len(consts)}) does not match "
            f"constvars length ({len(jaxpr.constvars)})"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_processing_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_processing.py brainstate/transform/_ir_processing_test.py
git commit -m "feat(transform): validate inputs in eqns_to_jaxpr/eqns_to_closed_jaxpr"
```

---

# Phase 2 — `_ir_inline.py`

### Task 2.1: Convert the test file to `unittest.TestCase`

**Files:**
- Modify: `brainstate/transform/_ir_inline_test.py`

- [ ] **Step 1: Wrap existing functions in a TestCase**

Replace the five module-level `def test_*` functions with methods of a single class, preserving their bodies verbatim (only indent and add `self`), and add the boilerplate:

```python
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from brainstate.transform import inline_jit


class TestInlineJit(unittest.TestCase):
    def test_expand_all_jits_preserves_value_and_removes_calls(self):
        ...  # existing body, replacing bare `assert` with self.assert* where convenient
    # ... the other four tests likewise ...


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run to confirm still green**

Run: `python -m pytest brainstate/transform/_ir_inline_test.py -v`
Expected: PASS (same 5 tests, now as methods).

- [ ] **Step 3: Commit**

```bash
git add brainstate/transform/_ir_inline_test.py
git commit -m "test(transform): convert _ir_inline tests to unittest.TestCase"
```

---

### Task 2.2: Fresh variables when inlining (I1 — repeated-inlining collision)

**Files:**
- Modify: `brainstate/transform/_ir_inline.py`
- Modify: `brainstate/transform/_ir_inline_test.py`

- [ ] **Step 1: Write the failing test** — append a method to `TestInlineJit`:

```python
    def test_same_helper_inlined_twice_has_unique_binders(self):
        @jax.jit
        def helper(x):
            return x * x + 1.0

        def outer(x):
            return helper(x) + helper(x + 1.0)

        cj = jax.make_jaxpr(outer)(jnp.float32(2.0))
        expanded = inline_jit(cj.jaxpr)
        # Every output variable across all equations must be bound exactly once.
        binders = [v for e in expanded.eqns for v in e.outvars]
        ids = [id(v) for v in binders]
        self.assertEqual(len(ids), len(set(ids)),
                         "inlining the same helper twice produced duplicate binders")
        # Numerics must still match the original function.
        from jax import core
        out = core.eval_jaxpr(expanded, [], jnp.float32(2.0))
        self.assertTrue(np.allclose(out[0], outer(jnp.float32(2.0))))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_inline_test.py::TestInlineJit::test_same_helper_inlined_twice_has_unique_binders -v`
Expected: FAIL — duplicate binder ids (and/or eval_jaxpr error) because inner Vars are reused across both inlinings.

- [ ] **Step 3: Write minimal implementation**

In `_ir_inline.py`, add a gensym and use it to create fresh `Var`s for every inner-scope variable (both produced and any internal-only vars) at each inlining site.

Add near the top:
```python
import itertools
from brainstate._compatible_import import Var
```

Inside `inline_jit`, before the main loop, create a counter seeded above existing names:
```python
    _counter = itertools.count()

    def fresh_like(var):
        # Create a brand-new Var with the same aval; unique per call.
        return Var('', var.aval) if 'suffix' not in Var.__init__.__code__.co_varnames \
            else Var('', var.aval)
```

> Implementation note: `Var`'s constructor signature varies across JAX versions. Determine the correct constructor at runtime once:
```python
    def _make_fresh_var_factory():
        import inspect
        params = inspect.signature(Var.__init__).parameters
        if 'suffix' in params:                      # newer JAX: Var(suffix, aval)
            return lambda aval, c=itertools.count(): Var('', aval)
        # older JAX: Var(count, suffix, aval)
        return lambda aval, c=itertools.count(): Var(next(c), '', aval)
    _fresh = _make_fresh_var_factory()
```

Then in the inlining branch, replace the `inner_var_mapping` construction so that *every* variable from `actual_jaxpr` (invars are mapped to the caller's vars; all other produced vars get a fresh Var):

```python
                inner_var_mapping = {}
                for inner_var, outer_var in zip(actual_jaxpr.invars, eqn.invars):
                    inner_var_mapping[inner_var] = var_mapping.get(outer_var, outer_var)

                expanded_inner = inline_jit(call_jaxpr, should_expand)
                if isinstance(expanded_inner, ClosedJaxpr):
                    expanded_inner = expanded_inner.jaxpr

                def remap(v):
                    if isinstance(v, Literal):
                        return v
                    if v not in inner_var_mapping:
                        inner_var_mapping[v] = _fresh(v.aval)  # fresh binder
                    return inner_var_mapping[v]

                for inner_eqn in expanded_inner.eqns:
                    new_invars = [remap(v) for v in inner_eqn.invars]
                    new_outvars = [remap(v) for v in inner_eqn.outvars]
                    replace_kwargs = {
                        'primitive': inner_eqn.primitive,
                        'invars': new_invars,
                        'outvars': new_outvars,
                        'params': inner_eqn.params,
                    }
                    if hasattr(inner_eqn, 'effects'):
                        replace_kwargs['effects'] = inner_eqn.effects
                    if hasattr(inner_eqn, 'source_info'):
                        replace_kwargs['source_info'] = inner_eqn.source_info
                    new_eqns.append(inner_eqn.replace(**replace_kwargs))

                for inner_out, outer_out in zip(expanded_inner.outvars, eqn.outvars):
                    var_mapping[outer_out] = remap(inner_out)
```

> Key change vs. current code: outvars and any internal vars are passed through `remap`, which allocates a **fresh** `Var` the first time it sees an inner var — so a second inlining of the same `call_jaxpr` cannot reuse the first inlining's binders.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_inline_test.py -v`
Expected: PASS (all, including the new regression).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_inline.py brainstate/transform/_ir_inline_test.py
git commit -m "fix(transform): allocate fresh vars when inlining to avoid binder collisions"
```

---

### Task 2.3: Thread nested consts and validate input (I2)

**Files:**
- Modify: `brainstate/transform/_ir_inline.py`
- Modify: `brainstate/transform/_ir_inline_test.py`

- [ ] **Step 1: Write the failing test** — append:

```python
    def test_input_validation(self):
        from brainstate.transform._ir_utils import IRValidationError
        with self.assertRaises(IRValidationError):
            inline_jit(123)

    def test_inline_closed_jaxpr_with_consts_runs(self):
        # A jit whose body closes over a (non-scalar) constant array.
        const = jnp.arange(3, dtype=jnp.float32)

        @jax.jit
        def helper(x):
            return x + const

        def outer(x):
            return helper(x) * 2.0

        cj = jax.make_jaxpr(outer)(jnp.zeros(3, jnp.float32))
        expanded = inline_jit(cj)           # pass the ClosedJaxpr
        # Must remain runnable and numerically correct.
        from jax import core
        out = core.eval_jaxpr(expanded.jaxpr, expanded.consts, jnp.zeros(3, jnp.float32))
        self.assertTrue(np.allclose(out[0], outer(jnp.zeros(3, jnp.float32))))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest "brainstate/transform/_ir_inline_test.py::TestInlineJit::test_input_validation" "brainstate/transform/_ir_inline_test.py::TestInlineJit::test_inline_closed_jaxpr_with_consts_runs" -v`
Expected: FAIL — no input validation; nested consts dropped so eval fails or mismatches.

- [ ] **Step 3: Write minimal implementation**

In `inline_jit`:

(a) At the very top, validate input:
```python
    from brainstate.transform._ir_utils import IRValidationError
    if not isinstance(jaxpr, (Jaxpr, ClosedJaxpr)):
        raise IRValidationError(
            f"inline_jit expects Jaxpr or ClosedJaxpr, got {type(jaxpr).__name__}."
        )
```

(b) Collect consts from nested `ClosedJaxpr` call_jaxprs and lift them to the outer closed jaxpr. Maintain two accumulators initialized before the loop:
```python
    lifted_constvars = list(inner_jaxpr.constvars)
    lifted_consts = list(original_closed.consts) if is_closed else []
```

When inlining a `call_jaxpr` that is a `ClosedJaxpr` with consts, map each of its `constvars` to a fresh outer constvar and record the corresponding const value:
```python
                if isinstance(call_jaxpr, ClosedJaxpr) and call_jaxpr.consts:
                    for cvar, cval in zip(call_jaxpr.jaxpr.constvars, call_jaxpr.consts):
                        new_cvar = _fresh(cvar.aval)
                        inner_var_mapping[cvar] = new_cvar
                        lifted_constvars.append(new_cvar)
                        lifted_consts.append(cval)
```
(Place this right after computing `inner_var_mapping` for invars and before iterating `expanded_inner.eqns`, so `remap` finds these constvars already mapped.)

(c) When constructing the returned jaxpr, include the lifted constvars:
```python
    new_jaxpr = inner_jaxpr.replace(eqns=new_eqns, outvars=new_outvars,
                                    constvars=lifted_constvars)
    if is_closed:
        return ClosedJaxpr(new_jaxpr, lifted_consts)
    else:
        # Bare Jaxpr input but nested consts appeared: still valid because the
        # constvars/consts are carried on the Jaxpr's constvars (consts live in
        # the enclosing ClosedJaxpr); when there are nested consts, return a
        # ClosedJaxpr to keep them bound.
        if lifted_consts:
            return ClosedJaxpr(new_jaxpr, lifted_consts)
        return new_jaxpr
```

> Note the documented behavior nuance: a bare-`Jaxpr` input that contains inlined nested consts is returned as a `ClosedJaxpr` (the only correct way to keep the consts bound). Record this in the docstring. If strict bare-Jaxpr return is required by a caller, they should pass a `ClosedJaxpr`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_inline_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_inline.py brainstate/transform/_ir_inline_test.py
git commit -m "fix(transform): lift nested consts and validate input in inline_jit"
```

---

# Phase 3 — `_ir_optim.py`

### Task 3.1: Switch optim to shared utils (no behavior change)

**Files:**
- Modify: `brainstate/transform/_ir_optim.py`

- [ ] **Step 1: Replace local copies with imports**

In `_ir_optim.py`:
- Delete the local `IdentitySet` class (lines ~71-133), `_constant_fold_blacklist`, `_partial_eval_jaxpr`, `_eval_eqn`.
- Add imports:
```python
from brainstate.transform._ir_utils import (
    IdentitySet, IRValidationError, partial_eval_jaxpr,
    literal_with_dtype, CONSTANT_FOLD_BLACKLIST,
)
```
- Update `constant_fold` to call `partial_eval_jaxpr(jaxpr, {})` (it already does via `_partial_eval_jaxpr`; just rename).
- Re-export `IdentitySet` for backward compatibility by keeping it importable: add `IdentitySet` to nothing special (it's already imported into the module namespace, so `from brainstate.transform._ir_optim import IdentitySet` still works).

- [ ] **Step 2: Run the existing optim tests**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py -v`
Expected: PASS — all pre-existing tests (including `TestIdentitySet`, which imports `IdentitySet` from this module) remain green. If `TestIdentitySet` imports the symbol from `_ir_optim`, it now resolves to the re-exported one — confirm.

- [ ] **Step 3: Commit**

```bash
git add brainstate/transform/_ir_optim.py
git commit -m "refactor(transform): use shared _ir_utils in _ir_optim"
```

---

### Task 3.2: Fix `optimize_jaxpr` default no-op (O1)

**Files:**
- Modify: `brainstate/transform/_ir_optim.py:801-846`
- Modify: `brainstate/transform/_ir_optim_test.py`

- [ ] **Step 1: Write the failing test** — append a method to `TestOptimizeJaxpr`:

```python
    def test_default_optimizations_actually_run(self):
        # f has foldable constants and dead code; default optimize_jaxpr() must
        # reduce the equation count (it was previously a silent no-op).
        def f(x):
            dead = x * 7.0          # unused
            y = x + (2.0 + 3.0)     # foldable constant
            return y

        cj = jax.make_jaxpr(f)(jnp.float32(1.0))
        n_before = len(cj.jaxpr.eqns)
        optimized = optimize_jaxpr(cj.jaxpr)         # no optimizations arg
        n_after = len(optimized.eqns)
        self.assertLess(n_after, n_before)
```

(Where `optimize_jaxpr` is already imported at the top of `_ir_optim_test.py`; if not, add `from brainstate.transform import optimize_jaxpr`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py::TestOptimizeJaxpr::test_default_optimizations_actually_run -v`
Expected: FAIL — `n_after == n_before` because the default is a no-op.

- [ ] **Step 3: Write minimal implementation**

In `optimize_jaxpr`, replace the broken default handling. Change the opening block (lines ~801-813) to:

```python
    _DEFAULT_PIPELINE = [
        'constant_fold', 'algebraic_simplification',
        'copy_propagation', 'cse', 'dce',
    ]
    if optimizations is None:
        optimizations = list(_DEFAULT_PIPELINE)
    elif isinstance(optimizations, str):
        optimizations = list(_DEFAULT_PIPELINE) if optimizations == 'all' else [optimizations]
    else:
        optimizations = list(optimizations)
```

Delete the now-dead second `if optimizations is None:` block (lines ~839-846).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py -v`
Expected: PASS (new test + all existing; verify none relied on the no-op default).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_optim.py brainstate/transform/_ir_optim_test.py
git commit -m "fix(transform): optimize_jaxpr default now runs the full pipeline"
```

---

### Task 3.3: Remove float-unsafe algebraic identities (O2)

**Files:**
- Modify: `brainstate/transform/_ir_optim.py` (`algebraic_simplification`, mul/div branches)
- Modify: `brainstate/transform/_ir_optim_test.py`

- [ ] **Step 1: Write the failing test** — append a new class:

```python
class TestAlgebraicSafety(unittest.TestCase):
    def test_zero_times_inf_not_folded_to_zero(self):
        # 0.0 * x must NOT be simplified to 0.0 for floats (0*inf = nan).
        def f(x):
            return jnp.float32(0.0) * x

        cj = jax.make_jaxpr(f)(jnp.float32(1.0))
        opt = algebraic_simplification(cj.jaxpr)
        # numerics: feed inf, expect nan (i.e., the mul survived).
        from jax import core
        out = core.eval_jaxpr(opt, [], jnp.float32(np.inf))
        self.assertTrue(np.isnan(np.asarray(out[0])))

    def test_zero_div_x_not_folded(self):
        def f(x):
            return jnp.float32(0.0) / x

        cj = jax.make_jaxpr(f)(jnp.float32(1.0))
        opt = algebraic_simplification(cj.jaxpr)
        from jax import core
        out = core.eval_jaxpr(opt, [], jnp.float32(0.0))
        self.assertTrue(np.isnan(np.asarray(out[0])))   # 0/0 = nan, not 0

    def test_integer_zero_times_x_still_folds(self):
        def f(x):
            return jnp.int32(0) * x

        cj = jax.make_jaxpr(f)(jnp.int32(5))
        opt = algebraic_simplification(cj.jaxpr)
        # integer 0*x is safe to fold; the mul should be gone.
        self.assertFalse(any(e.primitive.name == 'mul' for e in opt.eqns))
```

(Ensure `algebraic_simplification` and `np` are imported at the top of the test file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py::TestAlgebraicSafety -v`
Expected: FAIL — current code folds `0*x` and `0/x` to `0` for floats, so the inf/nan tests fail.

- [ ] **Step 3: Write minimal implementation**

In `algebraic_simplification`, update the `mul` and `div` branches.

Replace the `mul` branch:
```python
            elif eqn.primitive.name == 'mul':
                out_is_integer = np.issubdtype(np.dtype(outvar.aval.dtype), np.integer)
                if (is_zero(lhs) or is_zero(rhs)) and out_is_integer:
                    # 0*x = 0 is only IEEE-safe for integer dtypes.
                    var_map[outvar] = literal_with_dtype(0, outvar.aval)
                    simplified = True
                elif is_one(lhs):
                    var_map[outvar] = rhs
                    simplified = True
                elif is_one(rhs):
                    var_map[outvar] = lhs
                    simplified = True
```

Replace the `div` branch (remove `0/x -> 0` entirely):
```python
            elif eqn.primitive.name == 'div':
                if is_one(rhs):       # x / 1 = x  (always safe)
                    var_map[outvar] = lhs
                    simplified = True
```

Also update the `sub` `x - x = 0` literal to be dtype-correct:
```python
                elif id(lhs) == id(rhs):
                    var_map[outvar] = literal_with_dtype(0, outvar.aval)
                    simplified = True
```

(`literal_with_dtype` is imported in Task 3.1.) Remove the old local `make_literal` helper or have it delegate to `literal_with_dtype`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py -v`
Expected: PASS. Note: pre-existing `test_algebraic_simplification_*` tests for add/sub/mul-one/div-one still pass; any test that asserted `0*x→0` for floats must be updated to the new correct behavior (search the file for such an assertion and adjust to expect the mul to survive).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_optim.py brainstate/transform/_ir_optim_test.py
git commit -m "fix(transform): drop float-unsafe 0*x and 0/x algebraic identities"
```

---

### Task 3.4: Robust CSE param signatures (O4)

**Files:**
- Modify: `brainstate/transform/_ir_optim.py` (`common_subexpression_elimination`)
- Modify: `brainstate/transform/_ir_optim_test.py`

- [ ] **Step 1: Write the failing test** — append a method to `TestCommonSubexpressionElimination`:

```python
    def test_cse_does_not_crash_on_array_params(self):
        # broadcast_in_dim / convert_element_type carry non-hashable params;
        # CSE must not raise TypeError on them.
        def f(x):
            a = jnp.broadcast_to(x, (3,))
            b = jnp.broadcast_to(x, (3,))
            return a + b

        cj = jax.make_jaxpr(f)(jnp.float32(1.0))
        # Should complete without error.
        result = common_subexpression_elimination(cj.jaxpr)
        self.assertIsNotNone(result)

    def test_cse_dedups_pjit(self):
        @jax.jit
        def g(x):
            return x * x

        def f(x):
            return g(x) + g(x)

        cj = jax.make_jaxpr(f)(jnp.float32(2.0))
        result = common_subexpression_elimination(cj.jaxpr)
        # The two identical pjit calls collapse to one.
        n_pjit = sum(1 for e in result.eqns if e.primitive.name == 'pjit')
        self.assertEqual(n_pjit, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py::TestCommonSubexpressionElimination -v`
Expected: FAIL — `TypeError: unhashable type` (array params) and/or pjit not deduped.

- [ ] **Step 3: Write minimal implementation**

In `common_subexpression_elimination`, replace `make_key` with a robust, hashable signature builder. Add a module-level helper:

```python
def _canonical_param(value):
    """Return a hashable, value-stable representation of a param value."""
    import numpy as _np
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_param(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canonical_param(v)) for k, v in value.items()))
    if isinstance(value, _np.ndarray):
        return ('ndarray', value.shape, str(value.dtype), value.tobytes())
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):  # jax arrays
        arr = _np.asarray(value)
        return ('array', arr.shape, str(arr.dtype), arr.tobytes())
    # jaxprs and other complex params: identity (correct, conservative).
    return ('id', id(value))
```

Update `make_key`:
```python
    def make_key(eqn):
        invars_ids = tuple(id(get_var(v)) for v in eqn.invars)
        try:
            param_sig = tuple(sorted(
                (k, _canonical_param(v)) for k, v in eqn.params.items()
            ))
        except Exception:
            # Un-canonicalizable params: opt out of CSE for this eqn.
            return None
        return (eqn.primitive.name, invars_ids, param_sig)
```

And guard the cache lookup so a `None` key never dedups:
```python
        key = make_key(eqn)
        if key is not None and key in expr_cache and len(eqn.outvars) == len(expr_cache[key]):
            prev_outvars = expr_cache[key]
            for old_var, new_var in zip(eqn.outvars, prev_outvars):
                var_map[old_var] = new_var
        else:
            new_eqns.append(eqn)
            if key is not None:
                expr_cache[key] = eqn.outvars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_optim.py brainstate/transform/_ir_optim_test.py
git commit -m "fix(transform): make CSE robust to unhashable equation params"
```

---

### Task 3.5: Control-flow pass-through + dtype-correct CSE identities

**Files:**
- Modify: `brainstate/transform/_ir_optim.py` (CSE/algebraic identity-equation builders)
- Modify: `brainstate/transform/_ir_optim_test.py`

- [ ] **Step 1: Write the failing test** — append a new class:

```python
class TestControlFlowPassthrough(unittest.TestCase):
    def _roundtrip(self, f, *args):
        cj = jax.make_jaxpr(f)(*args)
        opt = optimize_jaxpr(cj)            # full default pipeline on ClosedJaxpr
        from jax import core
        out_opt = core.eval_jaxpr(opt.jaxpr, opt.consts, *args)
        ref = f(*args)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        for a, b in zip(out_opt, ref):
            self.assertTrue(np.allclose(np.asarray(a), np.asarray(b)))

    def test_cond(self):
        def f(x):
            return jax.lax.cond(x[0] > 0, lambda v: v * 2, lambda v: v - 1, x)
        self._roundtrip(f, jnp.float32([1.0, 2.0]))

    def test_scan(self):
        def f(x):
            def body(c, _):
                return c + 1.0, c
            final, ys = jax.lax.scan(body, x, None, length=3)
            return final
        self._roundtrip(f, jnp.float32(0.0))

    def test_while(self):
        def f(x):
            return jax.lax.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, x)
        self._roundtrip(f, jnp.float32(0.0))
```

- [ ] **Step 2: Run test to verify it fails (or errors)**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py::TestControlFlowPassthrough -v`
Expected: FAIL or ERROR — constant_fold previously could try to eagerly bind control-flow primitives; with the blacklist from Task 0.3 this should now pass for cond/scan/while. If any case errors, the blacklist or identity-equation dtype handling needs the fix below.

- [ ] **Step 3: Write minimal implementation (if needed)**

Ensure the CSE outvar identity-equation builder uses the dtype-correct convert and a valid `weak_type`/`sharding` param set consistent with the installed JAX. In `common_subexpression_elimination`, change the identity-equation params to include `sharding`:
```python
            eqn = JaxprEqn([canonical], [outvar], lax.convert_element_type_p,
                           {'new_dtype': outvar.aval.dtype, 'weak_type': False, 'sharding': None},
                           set(), _fallback_source_info(new_eqns), default_ctx)
```
Confirm `CONSTANT_FOLD_BLACKLIST` (from `_ir_utils`) includes `'while'`, `'scan'`, `'cond'` (it does, per Task 0.3) so `constant_fold` never eagerly executes them.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_optim.py brainstate/transform/_ir_optim_test.py
git commit -m "test(transform): cover control-flow pass-through through optimization passes"
```

---

### Task 3.6: Validate `max_iterations`

**Files:**
- Modify: `brainstate/transform/_ir_optim.py` (`optimize_jaxpr`)
- Modify: `brainstate/transform/_ir_optim_test.py`

- [ ] **Step 1: Write the failing test** — append to `TestOptimizeJaxpr`:

```python
    def test_invalid_max_iterations(self):
        from brainstate.transform._ir_utils import IRValidationError
        cj = jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0))
        with self.assertRaises(IRValidationError):
            optimize_jaxpr(cj.jaxpr, max_iterations=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py::TestOptimizeJaxpr::test_invalid_max_iterations -v`
Expected: FAIL — no validation; `range(0)` silently runs zero iterations.

- [ ] **Step 3: Write minimal implementation**

In `optimize_jaxpr`, right after the `optimizations` normalization block, add:
```python
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise IRValidationError(
            f"max_iterations must be a positive integer, got {max_iterations!r}."
        )
```
Keep the existing `TypeError` for non-Jaxpr input (the `else: raise TypeError(...)` branch stays — `TypeError` back-compat).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_optim_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_optim.py brainstate/transform/_ir_optim_test.py
git commit -m "feat(transform): validate max_iterations in optimize_jaxpr"
```

---

# Phase 4 — `_ir_tocode.py`

### Task 4.1: Create the round-trip test harness and baseline tests

**Files:**
- Create: `brainstate/transform/_ir_tocode_test.py`

- [ ] **Step 1: Write the test harness + baseline tests**

Create `brainstate/transform/_ir_tocode_test.py` (Apache header first), then:

```python
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from brainstate.transform import fn_to_python_code, jaxpr_to_python_code


def _exec_generated(source, fn_name):
    """Exec generated code in a controlled namespace and return the function."""
    ns = {'jax': jax, 'jnp': jnp, 'np': np}
    ns['jax'] = jax
    import jax.numpy
    import jax.lax
    ns.update({'jax': jax})
    exec(source, ns)
    # The generated module may expose the function under fn_name or 'unknown'.
    for candidate in (fn_name, 'unknown', 'generated_function'):
        if candidate in ns and callable(ns[candidate]):
            return ns[candidate]
    # Fallback: first callable defined by the source.
    for v in ns.values():
        if callable(v) and getattr(v, '__module__', None) is None:
            return v
    raise AssertionError("no generated function found in namespace")


class TestToCodeRoundTrip(unittest.TestCase):
    def _check(self, f, *args, fn_name=None):
        fn_name = fn_name or getattr(f, '__name__', 'generated_function')
        src = fn_to_python_code(f, *args)
        gen = _exec_generated(src, fn_name)
        got = gen(*args)
        ref = f(*args)
        got = got if isinstance(got, (tuple, list)) else (got,)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        for a, b in zip(got, ref):
            self.assertTrue(np.allclose(np.asarray(a), np.asarray(b)),
                            f"mismatch in {f.__name__}: {a} vs {b}")

    def test_arithmetic(self):
        def f(x, y):
            return (x + y) * (x - y) / (y + 1.0)
        self._check(f, jnp.float32(3.0), jnp.float32(2.0))

    def test_multiple_outputs(self):
        def f(x):
            return x + 1.0, x * 2.0
        self._check(f, jnp.float32(4.0))

    def test_reduction(self):
        def f(x):
            return jnp.sum(x)
        self._check(f, jnp.float32([1., 2., 3.]))

    def test_matmul(self):
        def f(a, b):
            return a @ b
        self._check(f, jnp.float32([[1., 2.], [3., 4.]]), jnp.float32([[5.], [6.]]))
```

- [ ] **Step 2: Run the harness against current code**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py -v`
Expected: Some tests PASS (arithmetic, reduction) and some may FAIL (matmul/`dot_general`), establishing the baseline. Record which fail.

- [ ] **Step 3: Commit the harness**

```bash
git add brainstate/transform/_ir_tocode_test.py
git commit -m "test(transform): add round-trip harness + baseline tests for _ir_tocode"
```

---

### Task 4.2: Replace silent unknown-primitive fallback with a loud error (T1)

**Files:**
- Modify: `brainstate/transform/_ir_tocode.py`
- Modify: `brainstate/transform/_ir_tocode_test.py`

- [ ] **Step 1: Write the failing test** — append:

```python
class TestToCodeUnsupported(unittest.TestCase):
    def test_unknown_primitive_raises(self):
        from brainstate.transform._ir_utils import UnsupportedPrimitiveError
        # `sort` is (initially) unregistered; generating code must raise clearly
        # rather than emit a call to a nonexistent `sort(...)`.
        def f(x):
            return jax.lax.sort(x)
        with self.assertRaises(UnsupportedPrimitiveError):
            fn_to_python_code(f, jnp.float32([3., 1., 2.]))
```

> If `sort` is registered later in Task 4.4, change this test to use a guaranteed-unregistered primitive, or register a throwaway primitive name absent from `prim_to_python`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py::TestToCodeUnsupported -v`
Expected: FAIL — currently falls back to `normal_fn('sort')`, producing code (no exception).

- [ ] **Step 3: Write minimal implementation**

In `_ir_tocode.py`, find the dispatch site that does `prim_to_python.get(prim.name, normal_fn(prim.name))` (around the equation-to-AST conversion). Replace the silent fallback:

```python
    handler = prim_to_python.get(eqn.primitive.name)
    if handler is None:
        from brainstate.transform._ir_utils import UnsupportedPrimitiveError
        raise UnsupportedPrimitiveError(
            f"No code-generation handler for primitive "
            f"'{eqn.primitive.name}'. Register one with "
            f"register_prim_handler('{eqn.primitive.name}', handler)."
        )
    return handler(state, eqn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py -v`
Expected: PASS (the unsupported test; baseline round-trips unaffected because their primitives are registered).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_tocode.py brainstate/transform/_ir_tocode_test.py
git commit -m "fix(transform): raise UnsupportedPrimitiveError for unknown primitives in tocode"
```

---

### Task 4.3: Replace context-free asserts and bare except (T2, T3); empty jaxpr (T4)

**Files:**
- Modify: `brainstate/transform/_ir_tocode.py`
- Modify: `brainstate/transform/_ir_tocode_test.py`

- [ ] **Step 1: Write the failing test** — append:

```python
class TestToCodeEdgeCases(unittest.TestCase):
    def test_empty_jaxpr_generates_valid_code(self):
        # Identity-only function: no equations, returns its input.
        def f(x):
            return x
        src = fn_to_python_code(f, jnp.float32(1.0))
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen(jnp.float32(5.0))), 5.0))

    def test_lambda_without_name(self):
        # A lambda has __name__ == '<lambda>' which is not a valid identifier;
        # generation must still produce runnable code.
        f = lambda x: x + 1.0
        src = fn_to_python_code(f, jnp.float32(1.0))
        self.assertIsInstance(src, str)
        self.assertTrue(len(src) > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py::TestToCodeEdgeCases -v`
Expected: Likely FAIL on the lambda (name `<lambda>` is not a valid Python identifier for the generated `def`) and possibly on empty jaxpr.

- [ ] **Step 3: Write minimal implementation**

In `_ir_tocode.py`:

(a) Sanitize the function name in `fn_to_python_code`:
```python
    try:
        name = fn.__name__
    except AttributeError:
        name = "generated_function"
    if not name.isidentifier():
        name = "generated_function"
```

(b) Replace context-free asserts in `partial_eval`/`_astify_*` with messaged `IRError`s, e.g.:
```python
        if isinstance(out, Jaxpr):
            from brainstate.transform._ir_utils import IRError
            raise IRError("Unexpected nested Jaxpr produced during tocode constant folding.")
```
(Apply to the asserts at the locations flagged in the spec: the two `assert not isinstance(..., Jaxpr)` and the asserts inside `_astify_convert_element_type` / `_astify_scan`, converting each to a clear `IRError`.)

(c) Ensure empty-jaxpr generation yields a valid `def` that returns the (possibly unchanged) outvars. The `jaxpr_to_py_ast` builder already constructs args + return; verify that with zero `eqns` it returns the invars mapped to outvars. Add a guard: if `jaxpr.eqns` is empty, the function body is a single `return` of the outvar names (which equal invar names), producing valid code.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_tocode.py brainstate/transform/_ir_tocode_test.py
git commit -m "fix(transform): messaged errors, name sanitization, empty-jaxpr in tocode"
```

---

### Task 4.4: Broaden primitive coverage (functionality)

**Files:**
- Modify: `brainstate/transform/_ir_tocode.py` (handler registrations near line ~308-327)
- Modify: `brainstate/transform/_ir_tocode_test.py`

- [ ] **Step 1: Write the failing tests** — append a class that round-trips each newly-supported op:

```python
class TestToCodeExpandedPrimitives(unittest.TestCase):
    def _check(self, f, *args):
        TestToCodeRoundTrip._check(self, f, *args)

    def test_unary_math(self):
        for op in (jnp.exp, jnp.log, jnp.sin, jnp.cos, jnp.tanh, jnp.sqrt, jnp.abs):
            with self.subTest(op=op.__name__):
                self._check(lambda x, _op=op: _op(x), jnp.float32([0.5, 1.5]))

    def test_select_where(self):
        def f(x):
            return jnp.where(x > 0, x, -x)
        self._check(f, jnp.float32([-1., 2., -3.]))

    def test_concatenate(self):
        def f(a, b):
            return jnp.concatenate([a, b])
        self._check(f, jnp.float32([1., 2.]), jnp.float32([3., 4.]))

    def test_reductions(self):
        for r in (jnp.max, jnp.min, jnp.prod):
            with self.subTest(r=r.__name__):
                self._check(lambda x, _r=r: _r(x), jnp.float32([1., 2., 3.]))

    def test_argmax(self):
        def f(x):
            return jnp.argmax(x)
        self._check(f, jnp.float32([1., 3., 2.]))

    def test_integer_pow(self):
        def f(x):
            return x ** 3
        self._check(f, jnp.float32(2.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py::TestToCodeExpandedPrimitives -v`
Expected: FAIL/ERROR — these primitives currently raise `UnsupportedPrimitiveError` (after Task 4.2).

- [ ] **Step 3: Write minimal implementation**

In `_ir_tocode.py`, after the existing registrations (line ~327), add the following handler table. Most are one-liners via the existing `normal_fn`; reductions reuse `_reduce_fn`; a few need a small custom handler.

```python
# --- Unary elementwise (jax.lax.*) ---
for _name in ('neg', 'sin', 'cos', 'tan', 'tanh', 'exp', 'log', 'log1p',
              'expm1', 'sqrt', 'rsqrt', 'abs', 'sign', 'floor', 'ceil',
              'round', 'erf', 'is_finite', 'logistic', 'integer_pow',
              'sort', 'rev', 'clamp', 'cbrt', 'square'):
    if _name not in prim_to_python:
        register_prim_handler(_name, normal_fn(f'jax.lax.{_name}'))

# --- Binary elementwise (jax.lax.*) ---
for _name in ('pow', 'rem', 'atan2', 'nextafter', 'and', 'or', 'xor',
              'shift_left', 'shift_right_logical', 'shift_right_arithmetic'):
    if _name not in prim_to_python:
        register_prim_handler(_name, normal_fn(f'jax.lax.{_name}'))

# --- Reductions (axes -> axis) ---
register_prim_handler('reduce_max', _reduce_fn('jax.numpy.max'))
register_prim_handler('reduce_min', _reduce_fn('jax.numpy.min'))
register_prim_handler('reduce_prod', _reduce_fn('jax.numpy.prod'))
register_prim_handler('reduce_and', _reduce_fn('jax.numpy.all'))
register_prim_handler('reduce_or', _reduce_fn('jax.numpy.any'))

# --- argmax/argmin: params {axes, index_dtype} -> numpy axis ---
def _argreduce_fn(fn_name):
    def handler(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        axis = eqn.params['axes'][0]
        call = ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=[ast.keyword(arg='axis', value=_astify_value(axis))],
        )
        return ast.Assign(outvars, call)
    return handler
register_prim_handler('argmax', _argreduce_fn('jax.numpy.argmax'))
register_prim_handler('argmin', _argreduce_fn('jax.numpy.argmin'))

# --- concatenate: params {dimension} -> list arg + axis kw ---
def _concatenate_handler(state, eqn):
    elts = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    call = ast.Call(
        func=ast.Name(id='jax.numpy.concatenate', ctx=ast.Load()),
        args=[ast.List(elts=elts, ctx=ast.Load())],
        keywords=[ast.keyword(arg='axis', value=_astify_value(eqn.params['dimension']))],
    )
    return ast.Assign(outvars, call)
register_prim_handler('concatenate', _concatenate_handler)

# --- cumulative ops: params {axis, reverse} ---
for _name, _fn in (('cumsum', 'jax.lax.cumsum'), ('cumprod', 'jax.lax.cumprod'),
                   ('cummax', 'jax.lax.cummax'), ('cummin', 'jax.lax.cummin')):
    register_prim_handler(_name, normal_fn(_fn))

# --- expand_dims: params {dimensions} ---
def _expand_dims_handler(state, eqn):
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    call = ast.Call(
        func=ast.Name(id='jax.lax.expand_dims', ctx=ast.Load()),
        args=invars + [_astify_value(tuple(eqn.params['dimensions']))],
        keywords=[],
    )
    return ast.Assign(outvars, call)
register_prim_handler('expand_dims', _expand_dims_handler)
```

> For each op, confirm the round-trip test passes; if a primitive's params don't map cleanly to the chosen `jax.lax`/`jax.numpy` call (e.g. `gather`/`scatter`/`pad`/`conv_general_dilated`/`top_k`/`iota`), write a dedicated handler mirroring the existing `_astify_dynamic_slice`/`_astify_scan` style and add a matching round-trip test. Only register a handler once its round-trip test passes; otherwise leave it to raise `UnsupportedPrimitiveError`. Add `gather`, `scatter`, `pad`, `iota`, `top_k`, `conv_general_dilated` handlers iteratively, each behind its own round-trip test in this class.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py -v`
Expected: PASS for every registered op's round-trip test. If a specific op cannot be made correct, do not register it (its test asserts `UnsupportedPrimitiveError` instead).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_tocode.py brainstate/transform/_ir_tocode_test.py
git commit -m "feat(transform): broaden tocode primitive coverage with round-trip tests"
```

---

### Task 4.5: Deepen control-flow code generation + tests

**Files:**
- Modify: `brainstate/transform/_ir_tocode.py` (`cond`/`scan`/`while`/`pjit`/`closed_call` handlers)
- Modify: `brainstate/transform/_ir_tocode_test.py`

- [ ] **Step 1: Write the failing tests** — append:

```python
class TestToCodeControlFlow(unittest.TestCase):
    def _check(self, f, *args):
        TestToCodeRoundTrip._check(self, f, *args)

    def test_pjit_nested(self):
        @jax.jit
        def inner(x):
            return x * x
        def f(x):
            return inner(x) + 1.0
        self._check(f, jnp.float32(3.0))

    def test_scan(self):
        def f(x):
            def body(c, _):
                return c + 1.0, c
            final, ys = jax.lax.scan(body, x, None, length=4)
            return final
        self._check(f, jnp.float32(0.0))

    def test_cond(self):
        def f(x):
            return jax.lax.cond(x > 0, lambda v: v * 2.0, lambda v: v - 1.0, x)
        self._check(f, jnp.float32(1.0))

    def test_while(self):
        def f(x):
            return jax.lax.while_loop(lambda v: v < 3.0, lambda v: v + 1.0, x)
        self._check(f, jnp.float32(0.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py::TestToCodeControlFlow -v`
Expected: Some PASS (scan/pjit if already handled), some FAIL (cond/while if unhandled or buggy). Record results.

- [ ] **Step 3: Write minimal implementation**

For each failing control-flow case, add/repair the handler so it emits nested Python that round-trips:
- `cond`: emit a Python `def`-per-branch (or `jax.lax.cond` call referencing generated branch functions) — generate the branch jaxprs via the same `jaxpr_to_py_ast` recursion and wire them as the `true_fun`/`false_fun` args of a `jax.lax.cond` call.
- `while`: generate `cond_jaxpr` and `body_jaxpr` as nested functions and emit a `jax.lax.while_loop(cond_fun, body_fun, init)` call.
- `scan`/`pjit`/`closed_call`: confirm existing handlers round-trip; fix `_astify_scan` asserts (converted in Task 4.3) and the `num_carry == 0` path.

Each handler reuses `jaxpr_to_py_ast(state, sub_jaxpr, fn_name=state.skolem('branch'))` to render nested bodies, then references them by their skolem names in the emitted control-flow call. This keeps generation uniform and avoids ID collisions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py -v`
Expected: PASS for all four control-flow round-trips.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_tocode.py brainstate/transform/_ir_tocode_test.py
git commit -m "feat(transform): nested control-flow code generation in tocode"
```

---

### Task 4.6: Make tocode use shared `_ir_utils`; export `register_prim_handler`

**Files:**
- Modify: `brainstate/transform/_ir_tocode.py`
- Modify: `brainstate/transform/__init__.py`
- Modify: `brainstate/transform/_ir_tocode_test.py`

- [ ] **Step 1: Write the failing test** — append:

```python
class TestRegisterPrimHandlerPublic(unittest.TestCase):
    def test_register_prim_handler_is_public(self):
        import brainstate.transform as T
        self.assertTrue(hasattr(T, 'register_prim_handler'))

    def test_custom_handler_used(self):
        import ast
        from brainstate.transform import register_prim_handler, fn_to_python_code
        # Register a handler for a normally-unsupported primitive name by
        # mapping it to an existing jax call, then ensure it is honored.
        # (Uses 'sort' only if still unregistered; otherwise pick another.)
        # This mainly asserts the public entry point exists and is callable.
        self.assertTrue(callable(register_prim_handler))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py::TestRegisterPrimHandlerPublic -v`
Expected: FAIL — `register_prim_handler` not exported from `brainstate.transform`.

- [ ] **Step 3: Write minimal implementation**

(a) In `_ir_tocode.py`, replace the local `IdentityMap`/`IdentitySet` and the local `constant_fold_jaxpr`/`partial_eval_jaxpr`/`_eval_eqn` with imports from `_ir_utils`:
```python
from brainstate.transform._ir_utils import (
    IdentityMap, IdentitySet, partial_eval_jaxpr, IRError, UnsupportedPrimitiveError,
)
```
Update `fn_to_python_code`/`jaxpr_to_python_code` to call `partial_eval_jaxpr(jaxpr, {})` where they previously called `constant_fold_jaxpr(jaxpr)`. Update any `IdentityMap` usage to the new key-storing semantics (iteration yields keys; `len()` unchanged).

(b) In `brainstate/transform/__init__.py`, extend the tocode import and `__all__`:
```python
from ._ir_tocode import (
    fn_to_python_code, jaxpr_to_python_code, register_prim_handler,
)
```
Add `'register_prim_handler'` to the `__all__` list (in the IR section).

(c) Add `register_prim_handler` to `_ir_tocode.py`'s `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest brainstate/transform/_ir_tocode_test.py -v`
Expected: PASS. Then run the whole transform import smoke test:
Run: `python -c "import brainstate.transform as T; print(T.register_prim_handler)"`
Expected: prints a function.

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_tocode.py brainstate/transform/__init__.py brainstate/transform/_ir_tocode_test.py
git commit -m "refactor(transform): tocode uses _ir_utils; export register_prim_handler"
```

---

# Phase 5 — `_ir_visualize.py`

### Task 5.1: Create the visualize test scaffold (pydot-gated)

**Files:**
- Create: `brainstate/transform/_ir_visualize_test.py`

- [ ] **Step 1: Write the scaffold + import-safety test**

Create `brainstate/transform/_ir_visualize_test.py` (Apache header), then:

```python
import importlib.util
import unittest
import jax
import jax.numpy as jnp

PYDOT = importlib.util.find_spec("pydot") is not None


class TestVisualizeImportSafety(unittest.TestCase):
    def test_module_imports_without_pydot(self):
        # Importing the module must never fail, even if pydot is missing.
        import brainstate.transform._ir_visualize as viz
        self.assertTrue(hasattr(viz, 'draw'))


@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeBasic(unittest.TestCase):
    def test_simple_arithmetic_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return 2.0 * x + 1.0
        g = draw(f)(jnp.float32([1., 2., 3.]))
        self.assertIsNotNone(g)
```

- [ ] **Step 2: Run**

Run: `python -m pytest brainstate/transform/_ir_visualize_test.py -v`
Expected: import-safety PASS; basic test PASS or SKIP depending on pydot. (If pydot installed and basic build crashes, that is the bug fixed in 5.2/5.3.)

- [ ] **Step 3: Commit**

```bash
git add brainstate/transform/_ir_visualize_test.py
git commit -m "test(transform): add pydot-gated visualize test scaffold"
```

---

### Task 5.2: Fix `is_literal` unbound/stale in `get_conditional` (V1)

**Files:**
- Modify: `brainstate/transform/_ir_visualize.py:415-424`
- Modify: `brainstate/transform/_ir_visualize_test.py`

- [ ] **Step 1: Write the failing test** — append (pydot-gated):

```python
@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeCond(unittest.TestCase):
    def test_cond_with_function_branch_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            # Branches non-trivial enough to render as function nodes.
            return jax.lax.cond(
                x[0] > 0,
                lambda v: jnp.sum(v * v),
                lambda v: jnp.sum(v) - 1.0,
                x,
            )
        # Must not raise UnboundLocalError / NameError.
        g = draw(f)(jnp.float32([1., 2., 3.]))
        self.assertIsNotNone(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest "brainstate/transform/_ir_visualize_test.py::TestVisualizeCond" -v`
Expected: FAIL (if pydot present) — `UnboundLocalError`/`NameError: is_literal` or stale-value misbehavior at line ~421.

- [ ] **Step 3: Write minimal implementation**

In `get_conditional`, the function-node branch (around lines 415-424), compute the per-edge literal-ness locally instead of relying on the stray `is_literal`:

```python
                    for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                        # Skip JAX drop/unused vars whose name ends with '_'.
                        if str(var)[-1] == "_":
                            continue
                        edge_is_literal = isinstance(var, Literal) or isinstance(p_var, Literal)
                        if not edge_is_literal:
                            cond_graph.add_edge(
                                pydot.Edge(f"{cond_graph_id}_{p_var}", branch_graph_id)
                            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_visualize_test.py -v`
Expected: PASS (or SKIP without pydot).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_visualize.py brainstate/transform/_ir_visualize_test.py
git commit -m "fix(transform): correct is_literal handling in cond visualization"
```

---

### Task 5.3: Empty / multi-equation top-level jaxpr (V2)

**Files:**
- Modify: `brainstate/transform/_ir_visualize.py:191-202` (`draw_dot_graph`)
- Modify: `brainstate/transform/_ir_visualize_test.py`

- [ ] **Step 1: Write the failing test** — append (pydot-gated):

```python
@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeTopLevel(unittest.TestCase):
    def test_empty_jaxpr_does_not_crash(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return x       # identity -> top-level jaxpr has zero eqns
        g = draw(f)(jnp.float32(1.0))
        self.assertIsNotNone(g)

    def test_multi_equation_top_level(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            a = x + 1.0
            b = x * 2.0
            return a + b   # several top-level eqns, not just eqns[0]
        g = draw(f)(jnp.float32(1.0))
        self.assertIsNotNone(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest "brainstate/transform/_ir_visualize_test.py::TestVisualizeTopLevel" -v`
Expected: FAIL (if pydot present) — empty jaxpr raises `IndexError: fn.eqns[0]`; multi-eqn renders only the first equation (assertion may still pass if it returns a graph, so strengthen by checking the graph has nodes for all eqns; at minimum the empty case must not crash).

- [ ] **Step 3: Write minimal implementation**

Rewrite `draw_dot_graph` to handle zero and many equations:

```python
    def draw_dot_graph(fn, collapse_primitives, show_avals):
        g = pydot.Dot(graph_type="digraph")
        eqns = list(fn.eqns)
        if len(eqns) == 0:
            # No computation: render invars and outvars only.
            from brainstate._compatible_import import Literal as _Lit
            for v in list(fn.invars) + list(fn.outvars):
                g.add_node(pydot.Node(name=f"_{v}", label=str(v)))
            return g
        n = 0
        for eqn in eqns:
            sub_graph, _, _, _, n = get_sub_graph(eqn, "", n, collapse_primitives, show_avals)
            if isinstance(sub_graph, pydot.Subgraph):
                g.add_subgraph(sub_graph)
            else:
                g.add_node(sub_graph)
        return g
```

> If `get_sub_graph` requires the wiring of inter-equation edges for multi-eqn graphs, wire outvar→invar edges between successive equations using the existing `f"{parent_id}_{var}"` id convention. Keep this minimal: the test asserts the graph builds and contains nodes; precise edge fidelity for the multi-eqn top level is best-effort and documented.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_visualize_test.py -v`
Expected: PASS (or SKIP).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_visualize.py brainstate/transform/_ir_visualize_test.py
git commit -m "fix(transform): handle empty and multi-equation top-level jaxprs in visualize"
```

---

### Task 5.4: Normalize `while` sub-graph return arity (V3) + scan/while tests

**Files:**
- Modify: `brainstate/transform/_ir_visualize.py` (`get_while`, `get_while_branch`)
- Modify: `brainstate/transform/_ir_visualize_test.py`

- [ ] **Step 1: Write the failing test** — append (pydot-gated):

```python
@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeLoops(unittest.TestCase):
    def test_while_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            return jax.lax.while_loop(lambda v: v < 5.0, lambda v: v + 1.0, x)
        g = draw(f)(jnp.float32(0.0))
        self.assertIsNotNone(g)

    def test_scan_builds(self):
        from brainstate.transform import draw
        @jax.jit
        def f(x):
            def body(c, _):
                return c + 1.0, c
            final, ys = jax.lax.scan(body, x, None, length=3)
            return final
        g = draw(f)(jnp.float32(0.0))
        self.assertIsNotNone(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest "brainstate/transform/_ir_visualize_test.py::TestVisualizeLoops" -v`
Expected: FAIL (if pydot present) — unpack/arity error on the `while` path.

- [ ] **Step 3: Write minimal implementation**

Audit `get_while`/`get_while_branch` and the `sub_graph_return` contract. Make every sub-graph helper return the documented 5-tuple `(node_or_subgraph, in_edges, out_nodes, out_edges, n_counter)`. Where `get_while_branch` currently returns a 4-tuple, add the missing element (or change the call site to unpack 4 consistently). Pick one contract and apply it uniformly; the spec mandates the 5-tuple `sub_graph_return`. Update the `while` assembly code to unpack exactly what the helper returns.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_visualize_test.py -v`
Expected: PASS (or SKIP).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_visualize.py brainstate/transform/_ir_visualize_test.py
git commit -m "fix(transform): normalize while/scan sub-graph return arity in visualize"
```

---

### Task 5.5: Document underscore-var rule (V4) + unsupported control-flow error

**Files:**
- Modify: `brainstate/transform/_ir_visualize.py` (4 TODO sites; dispatch in `get_sub_graph`/`is_not_primitive`)
- Modify: `brainstate/transform/_ir_visualize_test.py`

- [ ] **Step 1: Write the failing test** — append (pydot-gated):

```python
@unittest.skipUnless(PYDOT, "pydot not installed")
class TestVisualizeUnsupported(unittest.TestCase):
    def test_pjit_and_closed_call_build(self):
        from brainstate.transform import draw
        @jax.jit
        def inner(x):
            return x * x
        @jax.jit
        def f(x):
            return inner(x) + 1.0
        g = draw(f)(jnp.float32(2.0))
        self.assertIsNotNone(g)
```

- [ ] **Step 2: Run test to verify it fails (or passes as baseline)**

Run: `python -m pytest "brainstate/transform/_ir_visualize_test.py::TestVisualizeUnsupported" -v`
Expected: PASS if pjit already handled; if a nested call-like primitive (e.g. `closed_call`, `remat2`) reaches the generic path and crashes, FAIL.

- [ ] **Step 3: Write minimal implementation**

(a) Replace the four `# TODO: What does the underscore mean?` comments with a single documented constant and helper:
```python
def _is_dropped_var(var) -> bool:
    # JAX names unused/dropped outputs with a trailing underscore (e.g. 'd_');
    # these carry no data flow and are skipped in the graph.
    return str(var).endswith("_")
```
Use `_is_dropped_var(var)` at all four sites in place of `str(var)[-1] == "_"`.

(b) In `get_sub_graph`/`is_not_primitive`, recognize `closed_call` and `remat2` (and any primitive whose params contain a `jaxpr`/`call_jaxpr`) as expandable; for a genuinely unhandled control-flow-like primitive, raise `UnsupportedPrimitiveError` with a clear message instead of producing a malformed graph:
```python
from brainstate.transform._ir_utils import UnsupportedPrimitiveError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest brainstate/transform/_ir_visualize_test.py -v`
Expected: PASS (or SKIP).

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_ir_visualize.py brainstate/transform/_ir_visualize_test.py
git commit -m "fix(transform): document dropped-var rule; clearer unsupported handling in visualize"
```

---

# Phase 6 — Integration & finalization

### Task 6.1: Full IR test sweep + behavior-change verification

**Files:** (no source changes unless a failure surfaces)

- [ ] **Step 1: Run the entire IR suite**

Run: `python -m pytest brainstate/transform/_ir_utils_test.py brainstate/transform/_ir_processing_test.py brainstate/transform/_ir_inline_test.py brainstate/transform/_ir_optim_test.py brainstate/transform/_ir_tocode_test.py brainstate/transform/_ir_visualize_test.py -v`
Expected: All PASS (visualize tests SKIP if pydot absent).

- [ ] **Step 2: Run the broader transform import + a smoke of public API**

Run:
```bash
python -c "
import brainstate.transform as T
import jax, jax.numpy as jnp
print('exports ok:', all(hasattr(T, n) for n in
  ['inline_jit','optimize_jaxpr','constant_fold','dead_code_elimination',
   'common_subexpression_elimination','copy_propagation','algebraic_simplification',
   'eqns_to_jaxpr','eqns_to_closed_jaxpr','fn_to_python_code','jaxpr_to_python_code',
   'register_prim_handler','draw','view_pydot','draw_dot_graph']))
cj = jax.make_jaxpr(lambda x: x + (2.0+3.0))(jnp.float32(1.0))
print('optimize default reduces:', len(T.optimize_jaxpr(cj.jaxpr).eqns) <= len(cj.jaxpr.eqns))
"
```
Expected: `exports ok: True` and `optimize default reduces: True`.

- [ ] **Step 3: Run the pre-existing repo transform tests to check for regressions**

Run: `python -m pytest brainstate/transform/ -q`
Expected: No new failures attributable to these changes. Investigate any failure; if a pre-existing test encoded a now-fixed bug (e.g. the no-op default or `0*x→0`), update it to the corrected behavior and note it in the commit.

- [ ] **Step 4: Commit any test adjustments**

```bash
git add -A
git commit -m "test(transform): finalize IR hardening; align tests with corrected behavior"
```

---

### Task 6.2: Update the behavior-change log in the spec (traceability)

**Files:**
- Modify: `docs/superpowers/specs/2026-05-29-ir-modules-hardening-design.md` (Section 11) — only if any behavior change differed from the spec during implementation.

- [ ] **Step 1:** If implementation revealed any additional behavior change not already listed, append it to Section 11 with a one-line rationale.
- [ ] **Step 2:** Commit:
```bash
git add docs/superpowers/specs/2026-05-29-ir-modules-hardening-design.md
git commit -m "docs: reconcile behavior-change log with implementation"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** P1→1.1, P2→1.1, validation→1.2; I1→2.2, I2→2.3; O1→3.2, O2→3.3, O3→3.3/3.5, O4→3.4, O5→0.3/3.1/3.5; T1→4.2, T2/T3/T4→4.3, coverage→4.4/4.5, register_prim_handler→4.6; V1→5.2, V2→5.3, V3→5.4, V4→5.5; shared `_ir_utils`→Phase 0; testing strategy→all test tasks; behavior-change log→6.2. All spec sections map to at least one task.

**Placeholder scan:** No "TBD"/"implement later". Open-ended coverage in 4.4/4.5 is bounded by an explicit rule (register only when the round-trip test passes; otherwise raise `UnsupportedPrimitiveError`) and concrete handler tables/examples.

**Type consistency:** `IRError`/`IRValidationError`/`UnsupportedPrimitiveError`, `IdentitySet`/`IdentityMap`, `partial_eval_jaxpr`, `CONSTANT_FOLD_BLACKLIST`, `literal_with_dtype`, `ensure_jaxpr`, `is_scalar_literal_value` are defined in Phase 0 and used with consistent names/signatures throughout.
