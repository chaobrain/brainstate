# vmap2 Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-found `brainstate.transform`'s state-aware mapping on a single hardened engine (the `StatefulFunction`/`make_jaxpr` "Engine B"), with `vmap2`/`pmap2`/`StatefulMapping` as façades, automatic output-axis inference, filter+instance state selection, and `vmap`/`vmap_new_states` reimplemented as signature-preserving shims.

**Architecture:** A new `_mapping_core.py` holds the engine: it (1) **probes** the function with `make_jaxpr` + a `set_new_arg` hook to enumerate touched states and strip the batch axis off filter/instance-matched input states (no JAX-private tracer manufacturing); (2) runs a **discovery** `jax.vmap` pass that reads each written state's read-only `BatchTracer.batch_dim` to learn output axes (auto-dim inference); (3) runs the **execution** `jax.vmap`/`jax.pmap` pass over explicit state-value pytrees and scatters results back. All caching is via `StatefulFunction.get_arg_cache_key`, which folds closed-over state shapes into the key (so the old cache-staleness bug cannot occur). `vmap2`/`pmap2` select states by `Filter` or `State` instance; the old `vmap` shim translates `in_states`/`out_states` dicts into instance selectors and forces strict error-on-undeclared-batch.

**Tech Stack:** Python, JAX (`jax.vmap`/`jax.pmap`/`make_jaxpr`), `brainstate._state` (`State`, `StateTraceStack`, `catch_new_states`, `TRACE_CONTEXT`), `brainstate.random.RandomState`, `brainstate.util.filter`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-29-vmap2-unification-design.md`

---

## Conventions for every task

- Work on branch `worktree-vmap2-unification` (already checked out).
- All randomness in code and tests uses `brainstate.random`, never `jax.random`.
- Run tests from the worktree root with `pytest`.
- Commit after each task with the message shown. Do **not** add `Co-Authored-By` trailers.
- Public docstrings are NumPy-style per project `CLAUDE.md` (added in Phase 10).

---

## File structure

| File | Responsibility |
|------|----------------|
| `brainstate/transform/_mapping_core.py` *(new)* | Engine: helpers, axis normalization, probe/discovery/execution, RNG, stack-level unwinding. |
| `brainstate/transform/_mapping2.py` *(modify)* | Public façades: `StatefulMapping`, `vmap2`, `pmap2`, `vmap2_new_states`, `pmap2_new_states`, `map`. |
| `brainstate/transform/_mapping1.py` *(modify)* | `vmap`, `vmap_new_states` reimplemented as shims; re-export moved helpers. |
| `brainstate/transform/_mapping_core_test.py` *(new)* | Engine unit tests. |
| `brainstate/transform/_mapping2_test.py` *(rewrite)* | Public-API tests + composition matrix. |
| `brainstate/transform/_mapping1_test.py` *(unchanged)* | Backward-compat regression gate. |

---

# Phase 1 — Scaffolding & shared helpers

Move the proven helpers from `_mapping1.py` into `_mapping_core.py` and re-export them, so `_mapping1_test.py` (which imports them) stays green.

### Task 1.1: Create `_mapping_core.py` with moved helpers

**Files:**
- Create: `brainstate/transform/_mapping_core.py`
- Modify: `brainstate/transform/_mapping1.py` (remove helper bodies, import from core)
- Test: `brainstate/transform/_mapping_core_test.py`

- [ ] **Step 1: Write the failing test**

Create `brainstate/transform/_mapping_core_test.py`:

```python
import unittest

import jax.numpy as jnp

import brainstate as bst
from brainstate.transform._mapping_core import (
    _remove_axis,
    _get_batch_size,
    _flatten_in_out_states,
    _format_state_axes,
)


class TestHelpersMoved(unittest.TestCase):
    def test_remove_axis(self):
        x = jnp.arange(12).reshape(3, 4)
        self.assertEqual(_remove_axis(x, 0).shape, (4,))
        self.assertEqual(_remove_axis(x, 1).shape, (3,))
        self.assertEqual(_remove_axis(x, -1).shape, (3,))
        with self.assertRaises(IndexError):
            _remove_axis(x, 5)

    def test_get_batch_size_from_args(self):
        args = (jnp.arange(30).reshape(5, 6),)
        self.assertEqual(_get_batch_size(args, 0, {}), 5)

    def test_get_batch_size_inconsistent(self):
        args = (jnp.zeros((4, 5)), jnp.zeros((3, 5)))
        with self.assertRaises(ValueError):
            _get_batch_size(args, (0, 0), {})

    def test_flatten_list_of_states(self):
        s1 = bst.ShortTermState(jnp.array(1.0))
        s2 = bst.ShortTermState(jnp.array(2.0))
        axis_to_states, state_to_axis = _flatten_in_out_states([s1, s2])
        self.assertEqual(state_to_axis[s1], 0)
        self.assertEqual(state_to_axis[s2], 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping_core_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brainstate.transform._mapping_core'`

- [ ] **Step 3: Create `_mapping_core.py` and move helper bodies**

Create `brainstate/transform/_mapping_core.py` with the license header followed by the helpers **copied verbatim** from `_mapping1.py` (`_flatten_in_out_states`, `_remove_axis`, `_get_batch_size`, `_format_state_axes`, `_compile_stateful_function`) plus their imports:

```python
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# (full Apache-2.0 header, copied from _mapping1.py lines 1-14)
# ==============================================================================

import functools
import warnings
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from brainstate._compatible_import import BatchTracer
from brainstate._error import BatchAxisError
from brainstate._state import (
    State, StateTraceStack, NonBatchState, catch_new_states, TRACE_CONTEXT,
)
from brainstate.typing import Filter
from brainstate.util import filter as filter_module
from ._make_jaxpr import StatefulFunction, get_arg_cache_key

AxisName = Hashable
AxisToState = Dict[int, List[State]]
StateToAxis = Dict[State, int]

_rand = None


def _import_rand_state():
    global _rand
    if _rand is None:
        from brainstate.random import RandomState
        _rand = RandomState
    return _rand


# ---- moved verbatim from _mapping1.py ----
# _flatten_in_out_states, _remove_axis, _get_batch_size,
# _format_state_axes, _compile_stateful_function
```

Copy the **exact bodies** of those five functions from the current `_mapping1.py` (lines 49-182 and 86-120) into this file.

- [ ] **Step 4: Re-export from `_mapping1.py`**

In `brainstate/transform/_mapping1.py`, delete the five moved function definitions and add, after the existing imports:

```python
from ._mapping_core import (
    _flatten_in_out_states,
    _remove_axis,
    _get_batch_size,
    _format_state_axes,
    _compile_stateful_function,
)
```

- [ ] **Step 5: Run both test files to verify green**

Run: `pytest brainstate/transform/_mapping_core_test.py brainstate/transform/_mapping1_test.py -v`
Expected: PASS (all helper tests + all 40 legacy tests)

- [ ] **Step 6: Commit**

```bash
git add brainstate/transform/_mapping_core.py brainstate/transform/_mapping1.py brainstate/transform/_mapping_core_test.py
git commit -m "refactor(transform): extract shared mapping helpers into _mapping_core"
```

---

# Phase 2 — Axis specification normalization (filters + instances)

Add `_normalize_state_axes`, which turns any `state_in_axes`/`state_out_axes` value into `Dict[AxisName, predicate]`, accepting Filters **and** `State` instances. This is the basis for fixing finding #8 and for the shims.

### Task 2.1: `_normalize_state_axes`

**Files:**
- Modify: `brainstate/transform/_mapping_core.py`
- Test: `brainstate/transform/_mapping_core_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping_core_test.py`:

```python
from brainstate.transform._mapping_core import _normalize_state_axes
from brainstate.util import filter as filter_module


class TestNormalizeStateAxes(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(_normalize_state_axes(None), {})

    def test_single_filter_defaults_axis_0(self):
        spec = _normalize_state_axes(filter_module.OfType(bst.ParamState))
        self.assertIn(0, spec)
        p = bst.ParamState(jnp.zeros(3))
        s = bst.ShortTermState(jnp.zeros(3))
        self.assertTrue(spec[0]((), p))
        self.assertFalse(spec[0]((), s))

    def test_single_instance_defaults_axis_0(self):
        s = bst.ShortTermState(jnp.zeros(3))
        other = bst.ShortTermState(jnp.zeros(3))
        spec = _normalize_state_axes(s)
        self.assertTrue(spec[0]((), s))
        self.assertFalse(spec[0]((), other))

    def test_dict_mixed_filter_and_instance(self):
        p = bst.ParamState(jnp.zeros(3))
        s = bst.ShortTermState(jnp.zeros(3))
        spec = _normalize_state_axes({0: s, 1: filter_module.OfType(bst.ParamState)})
        self.assertTrue(spec[0]((), s))
        self.assertTrue(spec[1]((), p))

    def test_collection_of_instances(self):
        s1 = bst.ShortTermState(jnp.zeros(3))
        s2 = bst.ShortTermState(jnp.zeros(3))
        spec = _normalize_state_axes({0: [s1, s2]})
        self.assertTrue(spec[0]((), s1))
        self.assertTrue(spec[0]((), s2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestNormalizeStateAxes -v`
Expected: FAIL with `ImportError: cannot import name '_normalize_state_axes'`

- [ ] **Step 3: Implement `_normalize_state_axes`**

Add to `_mapping_core.py`:

```python
def _make_identity_predicate(states):
    """Return a predicate matching exactly the given State instances by identity."""
    ids = frozenset(id(s) for s in states)

    def predicate(path, state):
        return id(state) in ids

    return predicate


def _coerce_axis_value_to_predicate(value):
    """Coerce one axis-spec value (Filter | State | collection) into a predicate."""
    if isinstance(value, State):
        return _make_identity_predicate((value,))
    # collection of States?
    if isinstance(value, (list, tuple, set)) and all(isinstance(v, State) for v in value) and len(value) > 0:
        return _make_identity_predicate(tuple(value))
    # otherwise treat as a Filter
    return filter_module.to_predicate(value)


def _normalize_state_axes(spec):
    """Normalize a state_in_axes/state_out_axes spec into ``{axis: predicate}``.

    Accepts ``None``, a ``Filter``, a ``State``, a collection of ``State``s, or a
    ``dict`` mapping axes to any of those. A bare (non-dict) value is shorthand
    for ``{0: value}``.
    """
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        return {0: _coerce_axis_value_to_predicate(spec)}
    return {axis: _coerce_axis_value_to_predicate(value) for axis, value in spec.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestNormalizeStateAxes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping_core.py brainstate/transform/_mapping_core_test.py
git commit -m "feat(transform): normalize state axis specs to support filters and instances"
```

---

# Phase 3 — RNG split helper and robust stack-level unwinding

Two small, independently-testable utilities the engine needs.

### Task 3.1: `_split_rng_keys` and `_restore_rng_keys`

**Files:**
- Modify: `brainstate/transform/_mapping_core.py`
- Test: `brainstate/transform/_mapping_core_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping_core_test.py`:

```python
import brainstate.random
from brainstate.transform._mapping_core import _split_rng_keys


class TestSplitRngKeys(unittest.TestCase):
    def test_split_shapes_and_advance(self):
        rng = brainstate.random.RandomState(0)
        before = rng.value
        states = [rng]
        keys, backups = _split_rng_keys(states, batch_size=4)
        # one key-array per rng, leading dim == batch_size
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].shape[0], 4)
        # backup captures the advanced key, and global key changed
        self.assertEqual(len(backups), 1)
        self.assertFalse(bool((rng.value == before).all()))

    def test_empty(self):
        keys, backups = _split_rng_keys([], batch_size=4)
        self.assertEqual(keys, tuple())
        self.assertEqual(backups, tuple())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestSplitRngKeys -v`
Expected: FAIL with `ImportError: cannot import name '_split_rng_keys'`

- [ ] **Step 3: Implement the helper**

Add to `_mapping_core.py`:

```python
def _split_rng_keys(rng_states, batch_size):
    """Split each RandomState into ``batch_size`` keys, advancing the global key.

    Returns ``(keys, backups)`` where ``keys[i]`` has leading dim ``batch_size``
    and ``backups[i]`` is the advanced global key to restore after mapping.
    """
    keys = tuple(rng.split_key(batch_size) for rng in rng_states)
    backups = tuple(rng.value for rng in rng_states)
    return keys, backups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestSplitRngKeys -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping_core.py brainstate/transform/_mapping_core_test.py
git commit -m "feat(transform): add RNG key-split helper for state-aware mapping"
```

### Task 3.2: `_unwind_new_state_levels`

**Files:**
- Modify: `brainstate/transform/_mapping_core.py`
- Test: `brainstate/transform/_mapping_core_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping_core_test.py`:

```python
from brainstate._state import TRACE_CONTEXT
from brainstate.transform._mapping_core import _unwind_new_state_levels


class TestUnwindNewStateLevels(unittest.TestCase):
    def test_levels_return_to_base(self):
        base = TRACE_CONTEXT.get_trace_stack_level()
        s = bst.ShortTermState(jnp.zeros(3))
        # simulate two nested transform traces having raised the level
        s.increase_stack_level()
        s.increase_stack_level()
        self.assertEqual(s.stack_level, base + 2)
        _unwind_new_state_levels([s], base)
        self.assertEqual(s.stack_level, base)

    def test_never_below_zero(self):
        s = bst.ShortTermState(jnp.zeros(3))
        _unwind_new_state_levels([s], base_level=0)
        self.assertGreaterEqual(s.stack_level, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestUnwindNewStateLevels -v`
Expected: FAIL with `ImportError: cannot import name '_unwind_new_state_levels'`

- [ ] **Step 3: Implement the helper**

Add to `_mapping_core.py`:

```python
def _unwind_new_state_levels(states, base_level):
    """Unwind each new state's trace stack level back to ``base_level``.

    Replaces the fragile hardcoded ``decrease_stack_level()`` counts: each state
    created inside ``n`` nested transform traces has ``stack_level == base + n``;
    we decrease exactly that many, flooring at zero.
    """
    for st in states:
        delta = st.stack_level - base_level
        for _ in range(max(delta, 0)):
            st.decrease_stack_level()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestUnwindNewStateLevels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping_core.py brainstate/transform/_mapping_core_test.py
git commit -m "feat(transform): add delta-based new-state stack-level unwinding"
```

---

# Phase 4 — The unified engine

This is the heart of the plan: `_state_map_transform`, an evolution of `_mapping1.py`'s `_vmap_transform` that (a) discovers input states via filters/instances, (b) auto-detects output axes, (c) supports `kwargs` (broadcast), and (d) is parameterized by the mapping primitive (`jax.vmap`/`jax.pmap`).

### Task 4.1: Input-state discovery via probe

**Files:**
- Modify: `brainstate/transform/_mapping_core.py`
- Test: `brainstate/transform/_mapping_core_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping_core_test.py`:

```python
from brainstate.transform._mapping_core import _discover_in_states


class TestDiscoverInStates(unittest.TestCase):
    def test_filter_selects_param_state_axis0(self):
        p = bst.ParamState(jnp.zeros((4, 3)))   # batched on axis 0, per-example (3,)
        s = bst.ShortTermState(jnp.zeros(3))     # broadcast

        def f(x):
            p.value = p.value + x
            s.value = s.value + 1.0
            return p.value

        in_preds = _normalize_state_axes({0: filter_module.OfType(bst.ParamState)})
        result = _discover_in_states(
            f, args=(jnp.zeros((4, 3)),), kwargs={}, in_predicates=in_preds, in_axes=0,
        )
        # p discovered on axis 0; s not an in-state; no rng
        self.assertIn(p, result.dim_to_in_states[0])
        self.assertNotIn(s, result.dim_to_in_states.get(0, []))
        self.assertEqual(result.rng_states, [])
        self.assertIn(p, result.all_states)
        self.assertIn(s, result.all_states)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestDiscoverInStates -v`
Expected: FAIL with `ImportError: cannot import name '_discover_in_states'`

- [ ] **Step 3: Implement `_discover_in_states`**

Add to `_mapping_core.py`:

```python
class _Discovery:
    """Result of the input-state discovery probe."""
    __slots__ = ('dim_to_in_states', 'in_state_to_axis', 'rng_states',
                 'all_states', 'write_states', 'state_trace')

    def __init__(self, dim_to_in_states, in_state_to_axis, rng_states,
                 all_states, write_states, state_trace):
        self.dim_to_in_states = dim_to_in_states
        self.in_state_to_axis = in_state_to_axis
        self.rng_states = rng_states
        self.all_states = all_states
        self.write_states = write_states
        self.state_trace = state_trace


def _strip_args(args, in_axes):
    """Remove the mapped axis from positional args for per-example tracing."""
    if isinstance(in_axes, int):
        return jax.tree.map(lambda x: _remove_axis(x, in_axes), args)
    if isinstance(in_axes, (tuple, list)):
        return tuple(
            a if ax is None else jax.tree.map(lambda x: _remove_axis(x, ax), a)
            for a, ax in zip(args, in_axes)
        )
    if in_axes is None:
        return args
    raise TypeError(f"Unsupported in_axes type: {type(in_axes)}")


def _discover_in_states(f, args, kwargs, in_predicates, in_axes):
    """Probe ``f`` with ``make_jaxpr`` to enumerate touched states and classify
    filter/instance-matched input states, stripping their batch axis for
    per-example tracing. Uses a ``set_new_arg`` hook (no JAX-private tracers).
    """
    RandomState = _import_rand_state()
    dim_to_in_states = defaultdict(list)
    in_state_to_axis = {}
    rng_states = []

    def new_arg_hook(state):
        if isinstance(state, RandomState):
            rng_states.append(state)
            # per-example: a single split key (host-side, value irrelevant to trace)
            return state.split_key()
        for axis, pred in in_predicates.items():
            if pred((), state):
                dim_to_in_states[axis].append(state)
                in_state_to_axis[state] = axis
                return jax.tree.map(lambda v: _remove_axis(v, axis), state.value)
        return state.value

    stripped_args = _strip_args(args, in_axes)
    state_trace = StateTraceStack(name='vmap_discover')
    state_trace.set_new_arg(new_arg_hook)
    with state_trace:
        f(*stripped_args, **kwargs)
    # restore original (probe mutated values via the hook)
    state_trace.recovery_original_values()

    return _Discovery(
        dim_to_in_states=dict(dim_to_in_states),
        in_state_to_axis=in_state_to_axis,
        rng_states=rng_states,
        all_states=list(state_trace.states),
        write_states=list(state_trace.get_write_states()),
        state_trace=state_trace,
    )
```

> **Implementation note:** confirm `StateTraceStack` exposes `recovery_original_values()` (it is used in `_mapping2.py:457`). If absent on this engine, restore by iterating `state_trace.states` and calling `restore_value` with the values captured before the probe.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestDiscoverInStates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping_core.py brainstate/transform/_mapping_core_test.py
git commit -m "feat(transform): discover input states via make_jaxpr probe + set_new_arg hook"
```

### Task 4.2: The execution engine `_state_map_transform`

This generalizes `_mapping1.py`'s `_vmap_transform`. Read that function (current `_mapping1.py:185-435`) first — the structure below mirrors it, adding discovery, auto-dim, kwargs, and a pluggable `mapping_fn`.

**Files:**
- Modify: `brainstate/transform/_mapping_core.py`
- Test: `brainstate/transform/_mapping_core_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping_core_test.py`:

```python
from brainstate.transform._mapping_core import _state_map_transform
import jax


class TestStateMapTransform(unittest.TestCase):
    def test_filter_batched_param_state(self):
        p = bst.ParamState(jnp.zeros((3,)))  # will be batched to (3,) over axis 0

        def f(x):
            p.value = p.value + x
            return p.value

        mapped = _state_map_transform(
            f, in_axes=0, out_axes=0,
            state_in_axes={0: filter_module.OfType(bst.ParamState)},
            state_out_axes={0: filter_module.OfType(bst.ParamState)},
            mapping_fn=jax.vmap, mapping_kwargs={},
            unexpected_out_state_mapping='auto',
        )
        # p starts (3,); used as a batched input means its current value IS the batch
        p.value = jnp.zeros((3,))
        xs = jnp.array([1.0, 2.0, 3.0])
        out = mapped(xs)
        self.assertTrue(jnp.allclose(out, xs))
        self.assertTrue(jnp.allclose(p.value, xs))

    def test_autodim_new_batched_output(self):
        # A written state batched on axis 0, not declared in state_out_axes,
        # is auto-scattered under the default 'auto' policy.
        acc = bst.ShortTermState(jnp.zeros(4))

        def f(x):
            acc.value = acc.value + x
            return acc.value

        mapped = _state_map_transform(
            f, in_axes=0, out_axes=0,
            state_in_axes={0: filter_module.OfType(bst.ShortTermState)},
            state_out_axes=None,                # nothing declared
            mapping_fn=jax.vmap, mapping_kwargs={},
            unexpected_out_state_mapping='auto',
        )
        acc.value = jnp.zeros(4)
        xs = jnp.arange(4.0)
        out = mapped(xs)
        self.assertTrue(jnp.allclose(out, xs))
        self.assertTrue(jnp.allclose(acc.value, xs))

    def test_raise_policy_errors_on_undeclared_batch(self):
        leak = bst.ShortTermState(jnp.zeros(()))
        out_state = bst.ShortTermState(jnp.zeros(3))

        def f(x):
            leak.value = x         # becomes batched, undeclared
            out_state.value = x
            return out_state.value

        mapped = _state_map_transform(
            f, in_axes=0, out_axes=0,
            state_in_axes={0: out_state}, state_out_axes={0: out_state},
            mapping_fn=jax.vmap, mapping_kwargs={},
            unexpected_out_state_mapping='raise',
        )
        out_state.value = jnp.zeros(3)
        with self.assertRaises(BatchAxisError):
            mapped(jnp.arange(3.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestStateMapTransform -v`
Expected: FAIL with `ImportError: cannot import name '_state_map_transform'`

- [ ] **Step 3: Implement `_state_map_transform`**

Add to `_mapping_core.py`. This mirrors `_vmap_transform` but builds the in/out axis maps from discovery + auto-dim, supports kwargs (broadcast), and takes a `mapping_fn`:

```python
def _leaf_batch_dim(value):
    """Return the single batch dim of a (possibly pytree) value under vmap, or None."""
    dims = set()
    for leaf in jax.tree.leaves(value):
        dims.add(leaf.batch_dim if isinstance(leaf, BatchTracer) else None)
    if len(dims) != 1:
        raise BatchAxisError(
            f"Inconsistent batch dimensions across leaves: {dims}. "
            "All leaves of a state must share one batch dimension."
        )
    return dims.pop()


def _state_map_transform(
    f,
    *,
    in_axes=0,
    out_axes=0,
    state_in_axes=None,
    state_out_axes=None,
    axis_size=None,
    axis_name=None,
    spmd_axis_name=None,
    mapping_fn=jax.vmap,
    mapping_kwargs=None,
    unexpected_out_state_mapping='auto',
):
    RandomState = _import_rand_state()
    mapping_kwargs = {} if mapping_kwargs is None else dict(mapping_kwargs)
    in_predicates = _normalize_state_axes(state_in_axes)
    out_predicates = _normalize_state_axes(state_out_axes)
    if isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    # per-call caches keyed by StatefulFunction arg-cache-key (state-shape aware)
    cache = {}

    def _build_plan(args, kwargs):
        """Probe + auto-dim discovery; returns the cached execution plan."""
        disc = _discover_in_states(f, args, kwargs, in_predicates, in_axes)

        # classify outputs: declared filters first, then auto/raise/warn/ignore.
        out_axis_of = {}                          # state -> output axis (int)
        rng_set = set(disc.rng_states)
        # batch size for splitting rng / inferring axis size
        batch_size = _get_batch_size(args, in_axes,
                                     {ax: sts for ax, sts in disc.dim_to_in_states.items()},
                                     axis_size)

        # discovery vmap pass: learn each write state's actual batch dim
        detected = {}

        def probe_body(rng_keys, in_vmap_vals, args_):
            _restore_in_states(disc, rng_keys, in_vmap_vals)
            f(*args_, **kwargs)
            for st in disc.write_states:
                if st in rng_set:
                    continue
                detected[st] = _leaf_batch_dim(st.value)
            return 0.0  # dummy; out_axes=None

        in_vmap_axes, in_vmap_states = _grouped_in_axes(disc)
        rng_axis = 0 if disc.rng_states else None
        probe_keys = tuple(jnp.zeros((batch_size, 2), dtype='uint32') for _ in disc.rng_states)
        probe_in_vals = [[st.value for st in grp] for grp in in_vmap_states]
        jax.vmap(
            probe_body,
            in_axes=(rng_axis, in_vmap_axes, in_axes),
            out_axes=None,
            axis_size=axis_size,
            axis_name=axis_name,
            spmd_axis_name=spmd_axis_name,
        )(probe_keys, probe_in_vals, args)
        disc.state_trace.recovery_original_values()

        # assign output axes
        for st, dim in detected.items():
            # declared out filter wins
            declared = None
            for axis, pred in out_predicates.items():
                if pred((), st):
                    declared = axis
                    break
            if declared is not None:
                out_axis_of[st] = declared
                continue
            if dim is None:
                continue  # not batched -> broadcast (axis None)
            # batched + undeclared
            if st in disc.in_state_to_axis:
                out_axis_of[st] = disc.in_state_to_axis[st]
            elif unexpected_out_state_mapping == 'auto':
                out_axis_of[st] = dim
            elif unexpected_out_state_mapping == 'raise':
                st.raise_error_with_source_info(
                    BatchAxisError(
                        f"State {st} is batched on output axis {dim} but is not "
                        f"covered by state_out_axes. Declare it in state_out_axes "
                        f"or set unexpected_out_state_mapping to 'auto'/'warn'/'ignore'."
                    )
                )
            elif unexpected_out_state_mapping == 'warn':
                import warnings
                warnings.warn(
                    f"State {st} is batched on output axis {dim} but is not in "
                    f"state_out_axes; scattering at axis {dim}.", UserWarning,
                )
                out_axis_of[st] = dim
            elif unexpected_out_state_mapping == 'ignore':
                out_axis_of[st] = dim
            else:
                raise ValueError(
                    f"Invalid unexpected_out_state_mapping: {unexpected_out_state_mapping!r}"
                )

        # group out states by axis
        axis_to_out_states = defaultdict(list)
        for st, axis in out_axis_of.items():
            axis_to_out_states[axis].append(st)

        plan = dict(
            disc=disc, batch_size=batch_size,
            in_vmap_axes=in_vmap_axes, in_vmap_states=in_vmap_states,
            axis_to_out_states=dict(axis_to_out_states),
            rng_set=rng_set,
        )
        return plan

    def mapped_fn(*args, **kwargs):
        plan = _build_plan(args, kwargs)
        return _execute_plan(
            f, plan, args, kwargs,
            in_axes=in_axes, out_axes=out_axes, axis_size=axis_size,
            axis_name=axis_name,
            mapping_fn=mapping_fn, mapping_kwargs=mapping_kwargs,
        )

    return functools.wraps(f)(mapped_fn)
```

Add the supporting helpers `_grouped_in_axes`, `_restore_in_states`, and `_execute_plan`:

```python
def _grouped_in_axes(disc):
    """Return (axes_list, states_lists) for batched input-state groups (excl. rng).

    The axes are returned as a **list** (not a tuple) so the in_axes spec matches
    the list-of-lists structure of the batched state values — mirroring the proven
    structure in ``_mapping1.py`` (``st_in_axes = list(...)``). A tuple here would
    fail jax.vmap's in_axes prefix-structure check.
    """
    axes, groups = [], []
    for axis, states in disc.dim_to_in_states.items():
        axes.append(axis)
        groups.append(states)
    return list(axes), groups


def _restore_in_states(disc, rng_keys, in_vmap_vals):
    """Inside the mapped function: restore rng keys then batched input states."""
    for rng, key in zip(disc.rng_states, rng_keys):
        rng.restore_value(key)
    _, groups = _grouped_in_axes(disc)
    for states, vals in zip(groups, in_vmap_vals):
        for st, val in zip(states, vals):
            st.restore_value(val)


def _execute_plan(f, plan, args, kwargs, *, in_axes, out_axes, axis_size,
                  axis_name, mapping_fn, mapping_kwargs):
    # NOTE: primitive-specific kwargs (spmd_axis_name for vmap; devices/backend/
    # donate_argnums/static_broadcasted_argnums for pmap) are already bound into
    # ``mapping_fn`` by the façade (see Tasks 5.2/5.3). _execute_plan passes only
    # the kwargs every mapping primitive shares.
    disc = plan['disc']
    axis_to_out_states = plan['axis_to_out_states']
    in_vmap_axes, in_vmap_states = plan['in_vmap_axes'], plan['in_vmap_states']

    out_axes_keys = list(axis_to_out_states.keys())
    rng_axis = 0 if disc.rng_states else None

    def fn_to_map(rng_keys, in_vmap_vals, args_):
        _restore_in_states(disc, rng_keys, in_vmap_vals)
        out = f(*args_, **kwargs)
        out_state_vals = [[st.value for st in axis_to_out_states[ax]] for ax in out_axes_keys]
        return out, out_state_vals

    rng_keys, rng_backups = _split_rng_keys(disc.rng_states, plan['batch_size'])
    in_vals = [[st.value for st in grp] for grp in in_vmap_states]

    mapped = mapping_fn(
        fn_to_map,
        in_axes=(rng_axis, in_vmap_axes, in_axes),
        out_axes=(out_axes, out_axes_keys),
        axis_size=axis_size,
        axis_name=axis_name,
        **mapping_kwargs,
    )
    out, out_state_vals = mapped(rng_keys, in_vals, args)

    # scatter written state values back
    for ax, vals in zip(out_axes_keys, out_state_vals):
        for st, val in zip(axis_to_out_states[ax], vals):
            st.restore_value(val)
    # restore advanced rng keys
    for rng, key in zip(disc.rng_states, rng_backups):
        rng.restore_value(key)
    return out
```

> **Implementation notes for the executor:**
> - `jax.pmap` does not accept `spmd_axis_name`. When `mapping_fn` is `jax.pmap`, omit it — the `pmap2` façade (Task 5.3) passes a `mapping_fn` partial that drops these kwargs; `_execute_plan` must therefore only forward `spmd_axis_name`/`axis_name`/`axis_size` keys that the chosen primitive accepts. Implement a small `_call_mapping_fn(mapping_fn, ...)` that filters kwargs by `inspect.signature`, OR keep `jax.vmap` passing all three and have the `pmap2` mapping partial accept-and-ignore via a wrapper.
> - Confirm `state.raise_error_with_source_info` exists (`_state.py:537`).
> - The probe and discovery both call `recovery_original_values`; ensure states are pristine before `_execute_plan`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping_core_test.py::TestStateMapTransform -v`
Expected: PASS (filter batching, auto-dim, and raise policy)

- [ ] **Step 5: Add caching by arg-shape key**

Wrap `_build_plan` so plans are cached and reused across calls with the same abstract signature. Replace the `cache = {}` and `mapped_fn` body with:

```python
    def mapped_fn(*args, **kwargs):
        key = get_arg_cache_key((), (), args, kwargs)  # state-shape-aware via args+kwargs
        if key not in cache:
            cache[key] = _build_plan(args, kwargs)
        return _execute_plan(
            f, cache[key], args, kwargs,
            in_axes=in_axes, out_axes=out_axes, axis_size=axis_size,
            axis_name=axis_name,
            mapping_fn=mapping_fn, mapping_kwargs=mapping_kwargs,
        )
```

> **Note:** the plan stores `State` objects (closed-over identities), so closed-over state *shape* changes must invalidate the plan. Fold the touched states' abstract shapes into the key: after the first `_build_plan`, recompute the key as `get_arg_cache_key((), (), args + tuple(st.value for st in plan['disc'].all_states), kwargs)` on subsequent calls. Add the regression test in Task 9.x (cache invalidation) and adjust until it passes.

- [ ] **Step 6: Run the full engine test file**

Run: `pytest brainstate/transform/_mapping_core_test.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add brainstate/transform/_mapping_core.py brainstate/transform/_mapping_core_test.py
git commit -m "feat(transform): unified state-aware map engine with auto-dim inference"
```

---

# Phase 5 — Public façades (`vmap2`, `pmap2`, `StatefulMapping`)

Rewire `_mapping2.py` to delegate to the engine. Fixes bugs #1 (trailing comma), #2 (empty error — now in the engine), #9 (signature drift), #10 (docstring example, in Phase 10).

### Task 5.1: `StatefulMapping` as a thin façade

**Files:**
- Modify: `brainstate/transform/_mapping2.py`
- Test: `brainstate/transform/_mapping2_test.py`

- [ ] **Step 1: Write the failing test**

Create `brainstate/transform/_mapping2_test.py` (replacing the old one) with:

```python
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import unittest
import jax
import jax.numpy as jnp

import brainstate
import brainstate.random
from brainstate.transform import StatefulMapping, vmap2, pmap2, map as bmap
from brainstate.util import filter


class TestStatefulMappingFacade(unittest.TestCase):
    def test_basic_batched_state(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        sm = StatefulMapping(
            lambda x: _accumulate(counter, x),
            in_axes=0, out_axes=0,
            state_in_axes={0: filter.OfType(brainstate.ShortTermState)},
            state_out_axes={0: filter.OfType(brainstate.ShortTermState)},
        )
        xs = jnp.array([1.0, 2.0, 3.0])
        out = sm(xs)
        self.assertTrue(jnp.allclose(out, xs))
        self.assertTrue(jnp.allclose(counter.value, xs))


def _accumulate(counter, x):
    counter.value = counter.value + x
    return counter.value


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping2_test.py::TestStatefulMappingFacade -v`
Expected: FAIL (old `StatefulMapping` signature/behavior mismatch or import error after edits)

- [ ] **Step 3: Replace `StatefulMapping` with a façade**

In `brainstate/transform/_mapping2.py`, replace the entire `StatefulMapping` class body with a thin wrapper that builds the engine transform once and caches it:

```python
from ._mapping_core import _state_map_transform


class StatefulMapping:
    __module__ = "brainstate.transform"

    def __init__(
        self,
        fun,
        in_axes=0,
        out_axes=0,
        state_in_axes=None,
        state_out_axes=None,
        unexpected_out_state_mapping='auto',
        axis_size=None,
        axis_name=None,
        name=None,
        mapping_fn=jax.vmap,
        mapping_kwargs=None,
        spmd_axis_name=None,
    ):
        self.origin_fun = fun
        self.name = name
        self.in_axes = in_axes
        self.out_axes = out_axes
        self.state_in_axes = state_in_axes
        self.state_out_axes = state_out_axes
        self.axis_size = axis_size
        self.axis_name = axis_name
        self.mapping_fn = mapping_fn
        self.unexpected_out_state_mapping = unexpected_out_state_mapping
        self._mapped = _state_map_transform(
            fun,
            in_axes=in_axes, out_axes=out_axes,
            state_in_axes=state_in_axes, state_out_axes=state_out_axes,
            axis_size=axis_size, axis_name=axis_name, spmd_axis_name=spmd_axis_name,
            mapping_fn=mapping_fn, mapping_kwargs=mapping_kwargs,
            unexpected_out_state_mapping=unexpected_out_state_mapping,
        )

    def __call__(self, *args, **kwargs):
        return self._mapped(*args, **kwargs)
```

Remove the now-dead private methods (`__infer_batch_size`, `__new_batch_arg`, `__find_batch_dim`, `__fn_to_eval`, `__eval`, `__assign_*`, `__get_*`) and the broken trailing-comma assignments. Keep the module imports the engine needs.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping2_test.py::TestStatefulMappingFacade -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping2_test.py
git commit -m "refactor(transform): StatefulMapping delegates to unified engine"
```

### Task 5.2: `vmap2` façade

**Files:**
- Modify: `brainstate/transform/_mapping2.py`
- Test: `brainstate/transform/_mapping2_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping2_test.py`:

```python
class TestVmap2(unittest.TestCase):
    def test_decorator_form(self):
        counter = brainstate.ShortTermState(jnp.zeros(3))

        @vmap2(in_axes=0, out_axes=0,
               state_in_axes={0: filter.OfType(brainstate.ShortTermState)},
               state_out_axes={0: filter.OfType(brainstate.ShortTermState)})
        def acc(x):
            counter.value = counter.value + x
            return counter.value

        xs = jnp.array([1.0, 2.0, 3.0])
        self.assertTrue(jnp.allclose(acc(xs), xs))
        self.assertTrue(jnp.allclose(counter.value, xs))

    def test_stateless_matches_jax_vmap(self):
        @vmap2
        def f(x):
            return x * 2.0
        xs = jnp.arange(5.0)
        self.assertTrue(jnp.allclose(f(xs), jax.vmap(lambda x: x * 2.0)(xs)))

    def test_kwargs_broadcast(self):
        @vmap2(in_axes=0)
        def f(x, scale=2.0):
            return x * scale
        xs = jnp.arange(4.0)
        self.assertTrue(jnp.allclose(f(xs, scale=3.0), xs * 3.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping2_test.py::TestVmap2 -v`
Expected: FAIL (current `vmap2` builds the old engine; kwargs/stateless paths differ)

- [ ] **Step 3: Update `vmap2`**

Replace the `vmap2` function body in `_mapping2.py` so it constructs the façade with `mapping_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name)` and default `unexpected_out_state_mapping='auto'`. Keep the `Missing()` decorator branch. The signature stays as today (it is already correct); only the construction changes to pass through to `StatefulMapping`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping2_test.py::TestVmap2 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping2_test.py
git commit -m "feat(transform): vmap2 delegates to unified engine, supports kwargs"
```

### Task 5.3: `pmap2` façade (multi-device)

**Files:**
- Modify: `brainstate/transform/_mapping2.py`
- Test: `brainstate/transform/_mapping2_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping2_test.py`:

```python
class TestPmap2(unittest.TestCase):
    def test_pmap_sharded_param(self):
        n = jax.local_device_count()
        self.assertGreaterEqual(n, 2)  # XLA_FLAGS forces 8
        param = brainstate.ParamState(jnp.ones((n, 4)))

        @pmap2(axis_name='d', in_axes=0, out_axes=0,
               state_in_axes={0: filter.OfType(brainstate.ParamState)},
               state_out_axes={0: filter.OfType(brainstate.ParamState)})
        def update(delta):
            param.value = param.value + delta
            return param.value

        deltas = jnp.arange(n * 4.0).reshape(n, 4)
        out = update(deltas)
        self.assertEqual(out.shape, (n, 4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping2_test.py::TestPmap2 -v`
Expected: FAIL (old `pmap2` path / signature)

- [ ] **Step 3: Update `pmap2`**

Make `axis_name` keyword-only (consistent with `vmap2`). Build the façade with a `mapping_fn` that adapts `jax.pmap`'s kwargs:

```python
def _pmap_mapping_fn(static_broadcasted_argnums, devices, backend,
                     donate_argnums, global_arg_shapes):
    def make(fn, *, in_axes, out_axes, axis_size, axis_name, spmd_axis_name=None,
             **kw):
        # jax.pmap ignores spmd_axis_name and uses axis-0 mapping semantics
        return jax.pmap(
            fn, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes,
            axis_size=axis_size,
            static_broadcasted_argnums=static_broadcasted_argnums,
            devices=devices, backend=backend, donate_argnums=donate_argnums,
            global_arg_shapes=global_arg_shapes,
        )
    return make
```

and pass `mapping_fn=_pmap_mapping_fn(...)`. Update `_execute_plan`/`_state_map_transform` so the engine calls `mapping_fn(fn, in_axes=..., out_axes=..., axis_size=..., axis_name=..., spmd_axis_name=...)` and the pmap adapter absorbs `spmd_axis_name`. Default `axis_size` to `jax.local_device_count()` when `None` for pmap.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping2_test.py::TestPmap2 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping2_test.py
git commit -m "feat(transform): pmap2 delegates to unified engine; consistent axis_name"
```

---

# Phase 6 — `*_new_states`

### Task 6.1: Robust `vmap2_new_states` / `pmap2_new_states`

**Files:**
- Modify: `brainstate/transform/_mapping2.py`
- Test: `brainstate/transform/_mapping2_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping2_test.py`:

```python
class TestVmap2NewStates(unittest.TestCase):
    def test_new_states_vectorized_and_levels_clean(self):
        from brainstate._state import TRACE_CONTEXT

        class Counter(brainstate.nn.Module):
            def init_state(self, **kw):
                self.count = brainstate.ShortTermState(jnp.zeros(()))

        base = TRACE_CONTEXT.get_trace_stack_level()
        m = Counter()
        states = brainstate.transform.vmap2_new_states(m, init_kwargs={}, axis_size=5)
        self.assertEqual(m.count.value.shape, (5,))
        # no leaked trace levels
        self.assertEqual(m.count.stack_level, base)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping2_test.py::TestVmap2NewStates -v`
Expected: FAIL (old hardcoded `decrease_stack_level()` count leaves a non-base level, or shape mismatch)

- [ ] **Step 3: Reimplement `_map_new_states` using the engine + robust unwinding**

In `_mapping2.py`, rewrite `_map_new_states` to: capture `base = TRACE_CONTEXT.get_trace_stack_level()`; run initialization under the engine map; collect new states via `catch_new_states`; restore vectorized values; then `_unwind_new_state_levels(new_states, base)` instead of fixed `decrease_stack_level()` calls. Keep `state_out_axes` and `NonBatchState`/`INIT_NO_BATCHING` routing. `vmap2_new_states`/`pmap2_new_states` keep their current signatures and call `_map_new_states` with the respective façade.

```python
from ._mapping_core import _unwind_new_state_levels
from brainstate._state import TRACE_CONTEXT
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping2_test.py::TestVmap2NewStates -v`
Expected: PASS

- [ ] **Step 5: Multi-device new-states test + commit**

Append a `pmap2_new_states` test mirroring the above with `axis_size=jax.local_device_count()`, run the whole file, then:

```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping2_test.py
git commit -m "feat(transform): robust *_new_states stack-level unwinding via engine"
```

---

# Phase 7 — `map` hardening

### Task 7.1: Build `vmap2(f)` once; harden batch/remainder

**Files:**
- Modify: `brainstate/transform/_mapping2.py`
- Test: `brainstate/transform/_mapping2_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_mapping2_test.py`:

```python
class TestMap(unittest.TestCase):
    def test_matches_vmap(self):
        xs = jnp.arange(6.0).reshape(6, 1)
        f = lambda x: x + 1.0
        self.assertTrue(jnp.allclose(bmap(f, xs), jax.vmap(f)(xs)))

    def test_batch_size_with_remainder(self):
        xs = jnp.arange(5.0)
        f = lambda a: a * a
        self.assertTrue(jnp.allclose(bmap(f, xs, batch_size=2), jax.vmap(f)(xs)))

    def test_unequal_lengths_raise(self):
        with self.assertRaises(ValueError):
            bmap(lambda a, b: a + b, jnp.zeros(4), jnp.zeros(3), batch_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest brainstate/transform/_mapping2_test.py::TestMap -v`
Expected: FAIL on `test_unequal_lengths_raise` (current `_batch_and_remainder` validates only within itself / different message) or on construction.

- [ ] **Step 3: Harden `map`**

In `_mapping2.py`, modify `map`: construct `batched = vmap2(f)` **once** before the scan; in the `batch_size` branch use `g = lambda _, x: ((), batched(*x))`; keep the remainder path calling the same `batched`. Add an explicit leading-length check raising `ValueError("All inputs must share the same leading length...")` before batching.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest brainstate/transform/_mapping2_test.py::TestMap -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping2_test.py
git commit -m "perf(transform): build map's vmap2 once; validate input lengths"
```

---

# Phase 8 — Backward-compat shims (`vmap`, `vmap_new_states`)

### Task 8.1: Reimplement `vmap` on the engine

**Files:**
- Modify: `brainstate/transform/_mapping1.py`
- Test: `brainstate/transform/_mapping1_test.py` (unchanged — the gate)

- [ ] **Step 1: Confirm the gate currently passes**

Run: `pytest brainstate/transform/_mapping1_test.py -v`
Expected: PASS (40 tests) — baseline before refactor.

- [ ] **Step 2: Replace `_vmap_transform` with a shim over the engine**

In `_mapping1.py`, replace `_vmap_transform`'s body so it converts `in_states`/`out_states` (dicts/lists of instances) into the engine's instance-based `state_in_axes`/`state_out_axes` and forces strict errors:

```python
from ._mapping_core import _state_map_transform, _flatten_in_out_states


def _vmap_transform(f, *, in_axes=0, out_axes=0, in_states=None, out_states=None,
                    axis_size=None, axis_name=None, spmd_axis_name=None):
    axis_to_in, _ = _flatten_in_out_states(in_states)
    axis_to_out, _ = _flatten_in_out_states(out_states)
    state_in_axes = {ax: list(sts) for ax, sts in axis_to_in.items()} or None
    state_out_axes = {ax: list(sts) for ax, sts in axis_to_out.items()} or None
    return _state_map_transform(
        f, in_axes=in_axes, out_axes=out_axes,
        state_in_axes=state_in_axes, state_out_axes=state_out_axes,
        axis_size=axis_size, axis_name=axis_name, spmd_axis_name=spmd_axis_name,
        mapping_fn=functools.partial(jax.vmap, spmd_axis_name=spmd_axis_name),
        mapping_kwargs={},
        unexpected_out_state_mapping='raise',   # legacy declare-or-error
    )
```

`vmap(...)` keeps its current signature and continues to call `_vmap_transform`.

- [ ] **Step 3: Run the gate**

Run: `pytest brainstate/transform/_mapping1_test.py -v`
Expected: PASS (all 40). Fix engine/shim until green. Pay attention to:
`test_vmap_batched_state_not_in_out_states` (relies on `'raise'`),
`test_vmap_with_out_axes_different_from_zero`, and the RandomState test.

- [ ] **Step 4: Commit**

```bash
git add brainstate/transform/_mapping1.py
git commit -m "refactor(transform): reimplement vmap as a shim over the unified engine"
```

### Task 8.2: Reimplement `vmap_new_states` on the engine

**Files:**
- Modify: `brainstate/transform/_mapping1.py`
- Test: `brainstate/transform/_mapping1_test.py`

- [ ] **Step 1: Replace `_vmap_new_states_transform` internals**

Keep the public `vmap_new_states` signature. Change the inner restore to use the shared `_unwind_new_state_levels(vmap_states, base)` (capturing `base` before entering) instead of the single hardcoded `decrease_stack_level()`.

```python
from ._mapping_core import _unwind_new_state_levels
from brainstate._state import TRACE_CONTEXT
```

- [ ] **Step 2: Run the gate**

Run: `pytest brainstate/transform/_mapping1_test.py::TestVmapNewStates -v`
Expected: PASS (5 tests)

- [ ] **Step 3: Run full gate + collective-ops consumer**

Run: `pytest brainstate/transform/_mapping1_test.py brainstate/nn/_collective_ops.py -v` (and any `_collective_ops` test module if present)
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add brainstate/transform/_mapping1.py
git commit -m "refactor(transform): vmap_new_states uses shared stack-level unwinding"
```

---

# Phase 9 — Composition & integration tests

Each task adds real, oracle-backed tests. Put them in `_mapping2_test.py`.

### Task 9.1: `vmap2` × autodiff

- [ ] **Step 1: Write the tests**

```python
class TestComposeAutodiff(unittest.TestCase):
    def test_vmap2_of_grad_per_sample(self):
        w = brainstate.ParamState(jnp.array([1.0, 2.0, 3.0]))

        def loss(x):
            return jnp.sum((w.value * x) ** 2)

        per_sample = vmap2(brainstate.transform.grad(loss, grad_states=w))
        xs = jnp.arange(12.0).reshape(4, 3)
        got = per_sample(xs)
        # oracle: manual per-example grad
        want = jnp.stack([2.0 * (w.value * x) * x for x in xs])
        self.assertTrue(jnp.allclose(got, want))

    def test_grad_through_vmap2(self):
        w = brainstate.ParamState(jnp.array([1.0, 2.0, 3.0]))

        def batched_loss(xs):
            ys = vmap2(lambda x: jnp.sum(w.value * x))(xs)
            return jnp.sum(ys)

        g = brainstate.transform.grad(batched_loss, grad_states=w)
        xs = jnp.arange(12.0).reshape(4, 3)
        got = g(xs)
        want = jnp.sum(xs, axis=0)  # d/dw sum_i sum_j w_j x_ij
        self.assertTrue(jnp.allclose(got, want))
```

> Verified API: `brainstate.transform.grad(fun, grad_states=<State|Seq|Dict>)` returns the gradient w.r.t. the given states.

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestComposeAutodiff -v` → PASS
```bash
git add brainstate/transform/_mapping2_test.py
git commit -m "test(transform): vmap2 composition with autodiff"
```

### Task 9.2: `vmap2` × control flow (`scan`/`for_loop`/`while_loop`)

- [ ] **Step 1: Write the tests**

```python
class TestComposeControlFlow(unittest.TestCase):
    def test_vmap2_of_scan_stateful(self):
        h = brainstate.ShortTermState(jnp.zeros(3))

        def step(carry, x):
            h.value = h.value + x
            return carry, h.value

        def run(xs):  # xs: (T,) per example
            _, ys = brainstate.transform.scan(step, 0.0, xs)
            return ys

        mapped = vmap2(run, in_axes=0, out_axes=0,
                       state_in_axes={0: filter.OfType(brainstate.ShortTermState)},
                       state_out_axes={0: filter.OfType(brainstate.ShortTermState)})
        h.value = jnp.zeros(3)
        xs = jnp.arange(15.0).reshape(3, 5)  # 3 examples, T=5
        got = mapped(xs)
        # oracle: cumulative sums per example
        want = jnp.cumsum(xs, axis=1)
        self.assertEqual(got.shape, (3, 5))
        self.assertTrue(jnp.allclose(got, want))

    def test_scan_over_vmap2_body(self):
        # the map() pattern: scan a vmap2'd body
        f = lambda x: x * 2.0
        xs = jnp.arange(8.0)
        self.assertTrue(jnp.allclose(bmap(f, xs, batch_size=4), xs * 2.0))
```

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestComposeControlFlow -v` → PASS
```bash
git add brainstate/transform/_mapping2_test.py
git commit -m "test(transform): vmap2 composition with scan/control flow"
```

### Task 9.3: `vmap2` × conditionals (scalar and batched predicates)

- [ ] **Step 1: Write the tests**

```python
class TestComposeCond(unittest.TestCase):
    def test_vmap2_batched_predicate(self):
        # batched predicate -> lowered to select; both branches run
        def f(x):
            return brainstate.transform.cond(x > 0, lambda: x * 2.0, lambda: x * -1.0)
        xs = jnp.array([-2.0, 3.0, -1.0, 4.0])
        got = vmap2(f)(xs)
        want = jnp.where(xs > 0, xs * 2.0, xs * -1.0)
        self.assertTrue(jnp.allclose(got, want))
```

> Verified API: `brainstate.transform.cond(pred, true_fun, false_fun, *operands)`; with no operands the branch callables take no arguments (as used above).

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestComposeCond -v` → PASS
```bash
git add brainstate/transform/_mapping2_test.py
git commit -m "test(transform): vmap2 composition with conditionals"
```

### Task 9.4: `vmap2` × `jit`, RNG, nested, cache-invalidation

- [ ] **Step 1: Write the tests**

```python
class TestComposeMisc(unittest.TestCase):
    def test_jit_of_vmap2(self):
        @brainstate.transform.jit
        def run(xs):
            return vmap2(lambda x: x * 3.0)(xs)
        xs = jnp.arange(4.0)
        self.assertTrue(jnp.allclose(run(xs), xs * 3.0))

    def test_rng_advances_between_calls(self):
        rng = brainstate.random.RandomState(0)

        @vmap2(in_axes=0)
        def noisy(x):
            return x + rng.randn()
        xs = jnp.zeros(4)
        a = noisy(xs)
        b = noisy(xs)
        self.assertFalse(bool(jnp.allclose(a, b)))  # keys advanced, not reused

    def test_nested_vmap2(self):
        f = lambda x: x + 1.0
        xs = jnp.arange(6.0).reshape(2, 3)
        got = vmap2(vmap2(f))(xs)
        self.assertTrue(jnp.allclose(got, xs + 1.0))

    def test_cache_invalidation_on_state_reshape(self):
        s = brainstate.ShortTermState(jnp.zeros(3))

        @vmap2(in_axes=0,
               state_in_axes={0: filter.OfType(brainstate.ShortTermState)},
               state_out_axes={0: filter.OfType(brainstate.ShortTermState)})
        def f(x):
            s.value = s.value + x
            return s.value

        s.value = jnp.zeros(3)
        self.assertTrue(jnp.allclose(f(jnp.arange(3.0)), jnp.arange(3.0)))
        # reshape the closed-over state; must re-trace, not corrupt
        s.value = jnp.zeros(5)
        self.assertTrue(jnp.allclose(f(jnp.arange(5.0)), jnp.arange(5.0)))
```

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestComposeMisc -v` → PASS
(If `test_cache_invalidation_on_state_reshape` fails, implement the shape-aware key note from Task 4.2 Step 5.)
```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping2_test.py
git commit -m "test(transform): vmap2 with jit/rng/nesting/cache invalidation"
```

### Task 9.4b: `vmap2` × remat, `pmap2` × `vmap2`, and `warn`/`ignore` policies

- [ ] **Step 1: Write the tests**

```python
class TestComposeRematPmapPolicy(unittest.TestCase):
    def test_vmap2_of_remat(self):
        @brainstate.transform.remat
        def f(x):
            return jnp.sin(x) * 2.0
        xs = jnp.arange(5.0)
        self.assertTrue(jnp.allclose(vmap2(f)(xs), jnp.sin(xs) * 2.0))

    def test_grad_vmap2_remat(self):
        w = brainstate.ParamState(jnp.array(0.5))

        @brainstate.transform.remat
        def step(x):
            return jnp.sum(jnp.tanh(w.value * x))

        def batched(xs):
            return jnp.sum(vmap2(step, in_axes=0)(xs))

        g = brainstate.transform.grad(batched, grad_states=w)
        xs = jnp.arange(6.0).reshape(3, 2) / 5.0
        gv = g(xs)
        eps = 1e-3
        w.value = jnp.array(0.5 + eps); hi = batched(xs)
        w.value = jnp.array(0.5 - eps); lo = batched(xs)
        w.value = jnp.array(0.5)
        self.assertTrue(jnp.allclose(gv, (hi - lo) / (2 * eps), atol=1e-2))

    def test_pmap2_of_vmap2(self):
        n = jax.local_device_count()
        # outer device-parallel over n, inner vectorized over m
        m = 4

        @pmap2(axis_name='d', in_axes=0, out_axes=0)
        def per_device(batch):           # batch: (m,)
            return vmap2(lambda x: x * 2.0)(batch)

        data = jnp.arange(float(n * m)).reshape(n, m)
        out = per_device(data)
        self.assertEqual(out.shape, (n, m))
        self.assertTrue(jnp.allclose(out, data * 2.0))

    def test_warn_policy_scatters_with_warning(self):
        leak = brainstate.ShortTermState(jnp.zeros(()))

        @vmap2(in_axes=0, unexpected_out_state_mapping='warn')
        def f(x):
            leak.value = x          # batched, undeclared
            return x

        leak.value = jnp.zeros(())
        with self.assertWarns(UserWarning):
            out = f(jnp.arange(3.0))
        self.assertTrue(jnp.allclose(out, jnp.arange(3.0)))
        self.assertTrue(jnp.allclose(leak.value, jnp.arange(3.0)))

    def test_ignore_policy_scatters_silently(self):
        leak = brainstate.ShortTermState(jnp.zeros(()))

        @vmap2(in_axes=0, unexpected_out_state_mapping='ignore')
        def f(x):
            leak.value = x
            return x

        leak.value = jnp.zeros(())
        out = f(jnp.arange(3.0))
        self.assertTrue(jnp.allclose(leak.value, jnp.arange(3.0)))
```

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestComposeRematPmapPolicy -v` → PASS
```bash
git add brainstate/transform/_mapping2_test.py
git commit -m "test(transform): remat/pmap-of-vmap composition and warn/ignore policies"
```

### Task 9.5: Deep composition gate — `grad(vmap2(scan(rnn_step)))`

- [ ] **Step 1: Write the test**

```python
class TestDeepComposition(unittest.TestCase):
    def test_grad_vmap2_scan_rnn(self):
        w = brainstate.ParamState(jnp.array(0.5))

        def rnn_step(h, x):
            h2 = jnp.tanh(w.value * h + x)
            return h2, h2

        def example_loss(seq):           # seq: (T,)
            _, ys = brainstate.transform.scan(rnn_step, 0.0, seq)
            return jnp.sum(ys ** 2)

        def batched_loss(seqs):          # seqs: (B, T)
            losses = vmap2(example_loss, in_axes=0)(seqs)
            return jnp.mean(losses)

        g = brainstate.transform.grad(batched_loss, grad_states=w)
        seqs = jnp.arange(12.0).reshape(3, 4) / 10.0
        grad_val = g(seqs)
        # oracle: finite-difference on w
        eps = 1e-3
        w.value = jnp.array(0.5 + eps); hi = batched_loss(seqs)
        w.value = jnp.array(0.5 - eps); lo = batched_loss(seqs)
        w.value = jnp.array(0.5)
        fd = (hi - lo) / (2 * eps)
        self.assertTrue(jnp.allclose(grad_val, fd, atol=1e-2))
```

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestDeepComposition -v` → PASS
```bash
git add brainstate/transform/_mapping2_test.py
git commit -m "test(transform): deep grad(vmap2(scan)) composition gate"
```

### Task 9.6: Integration — `nn.Module` + shim/`vmap2` parity

- [ ] **Step 1: Write the tests**

```python
class TestIntegration(unittest.TestCase):
    def test_module_vmap2_new_states_forward(self):
        net = brainstate.nn.Linear(4, 3)
        brainstate.transform.vmap2_new_states(net, init_kwargs={}, axis_size=8)
        self.assertEqual(net.weight.value['weight'].shape[0], 8)

    def test_vmap_shim_matches_vmap2(self):
        s1 = brainstate.ShortTermState(jnp.zeros(3))

        @brainstate.transform.vmap(in_axes=0,
                                   in_states={0: {'s': s1}}, out_states={0: {'s': s1}})
        def f_old(x):
            s1.value = s1.value + x
            return s1.value

        s1.value = jnp.zeros(3)
        out_old = f_old(jnp.arange(3.0))

        s2 = brainstate.ShortTermState(jnp.zeros(3))

        @vmap2(in_axes=0,
               state_in_axes={0: s2}, state_out_axes={0: s2})
        def f_new(x):
            s2.value = s2.value + x
            return s2.value

        s2.value = jnp.zeros(3)
        out_new = f_new(jnp.arange(3.0))
        self.assertTrue(jnp.allclose(out_old, out_new))
```

> Verified: `brainstate.nn.Linear(4, 3).weight` is a `ParamState` whose `.value` is `{'weight': (4, 3), 'bias': (3,)}`; after `vmap2_new_states(..., axis_size=8)` the `'weight'` leaf is `(8, 4, 3)`, so `.value['weight'].shape[0] == 8`.

- [ ] **Step 2: Run / fix / commit**

Run: `pytest brainstate/transform/_mapping2_test.py::TestIntegration -v` → PASS
```bash
git add brainstate/transform/_mapping2_test.py
git commit -m "test(transform): nn integration + vmap/vmap2 parity"
```

### Task 9.7: Full transform + nn regression sweep

- [ ] **Step 1: Run the whole affected surface**

Run:
```bash
pytest brainstate/transform/ -v
pytest brainstate/nn/ -q
pytest brainstate/random/ -q
```
Expected: PASS. Fix regressions before proceeding.

- [ ] **Step 2: Commit any fixes**

```bash
git add -A
git commit -m "fix(transform): address regressions from mapping unification"
```

---

# Phase 10 — Documentation

### Task 10.1: NumPy-style docstrings + corrected examples

**Files:**
- Modify: `brainstate/transform/_mapping2.py`, `brainstate/transform/_mapping1.py`

- [ ] **Step 1: Write/repair docstrings**

For each public symbol (`StatefulMapping`, `vmap2`, `pmap2`, `vmap2_new_states`, `pmap2_new_states`, `map`, `vmap`, `vmap_new_states`) ensure a NumPy-style docstring with sections in the canonical order (Short summary, Extended summary, Parameters, Returns, Raises, See Also, Notes, Examples). Every `Examples` block must be doctest-accurate and self-contained. Replace the incorrect scalar-counter `vmap2` example with this corrected one:

```python
"""
Examples
--------
.. code-block:: python

    >>> import brainstate
    >>> import jax.numpy as jnp
    >>> from brainstate.util.filter import OfType
    >>>
    >>> counter = brainstate.ShortTermState(jnp.zeros(3))
    >>>
    >>> @brainstate.transform.vmap2(
    ...     in_axes=0,
    ...     state_in_axes={0: OfType(brainstate.ShortTermState)},
    ...     state_out_axes={0: OfType(brainstate.ShortTermState)},
    ... )
    ... def accumulate(x):
    ...     counter.value = counter.value + x
    ...     return counter.value
    >>>
    >>> accumulate(jnp.array([1., 2., 3.]))
    Array([1., 2., 3.], dtype=float32)
    >>> counter.value
    Array([1., 2., 3.], dtype=float32)
"""
```

- [ ] **Step 2: Verify examples run**

Run: `python -c "import brainstate; help(brainstate.transform.vmap2)"` and manually execute the corrected example in a REPL; confirm the printed arrays match.

- [ ] **Step 3: Add a module-level narrative**

At the top of `_mapping2.py`, add a module docstring covering: discovery→execution model, filter-vs-instance selection, RNG behavior, the `unexpected_out_state_mapping` policy values, and a short `vmap` → `vmap2` migration note.

- [ ] **Step 4: Commit**

```bash
git add brainstate/transform/_mapping2.py brainstate/transform/_mapping1.py
git commit -m "docs(transform): NumPy-style docstrings and corrected examples for mapping"
```

### Task 10.2: Final full-suite verification

- [ ] **Step 1: Run everything**

Run: `pytest brainstate/transform/ brainstate/nn/ brainstate/random/ -q`
Expected: PASS.

- [ ] **Step 2: Confirm no `jax.random` was introduced**

Run: `git diff main --stat && grep -rn "jax.random" brainstate/transform/_mapping_core.py brainstate/transform/_mapping2.py brainstate/transform/_mapping1.py`
Expected: no `jax.random` usages in the changed mapping files.

- [ ] **Step 3: Commit any final touch-ups**

```bash
git add -A
git commit -m "chore(transform): finalize vmap2 unification"
```

---

## Definition of done (from spec §9)

- Confirmed bugs #1–#3 fixed structurally; #4–#10 addressed.
- `vmap2`/`pmap2`/`StatefulMapping`/`map`/`*_new_states` on the unified Engine-B core with auto-dim.
- `vmap`/`vmap_new_states` shims pass `_mapping1_test.py` unchanged.
- New/rewritten suites pass, incl. real multi-device `pmap2` (`XLA_FLAGS=8`).
- Transformation-composition matrix passes against oracles, incl. the
  `grad(vmap2(scan(rnn_step)))` deep gate.
- All public symbols carry doctest-accurate NumPy-style docstrings.
- No direct `jax.random` usage in changed mapping files.
