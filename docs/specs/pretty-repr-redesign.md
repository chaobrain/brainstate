# Pretty-repr redesign (Node / Module / State)

## Goal

Replace the one-attribute-per-line Flax-style renderer behind `graph.Node`,
`nn.Module`, `State`, `TreefyState` and the `brainstate.util` containers with a
width-aware compact tree renderer, plus a summary footer, configurable
options, and opt-in ANSI color.

## Target output (width 88)

```
Sequential(
  (3,) -> (2,),
  layers=[
    [0] Linear((3,) -> (4,), weight=ParamState{bias: f32[4], weight: f32[3,4]}),  # 16 params
    [1] ReLU(),
    [2] Linear((4,) -> (2,), weight=ParamState{bias: f32[2], weight: f32[4,2]}),  # 10 params
  ],
)  # 26 params (104 B)
```

## Protocol (backward compatible)

`__pretty_repr__` still yields one `PrettyType` followed by `PrettyAttr`s.
Additive fields:

| field                   | meaning                                                         |
|-------------------------|-----------------------------------------------------------------|
| `PrettyAttr.key == ''`  | positional item, rendered without `key<sep>`                    |
| `PrettyAttr.comment`    | trailing `# comment`, shown only when the parent group is broken |
| `PrettyType.ref`        | when set, the object renders as this literal marker             |

`PrettyAttr.value` that is a `str` is still inserted verbatim (protocol
unchanged); `PrettyObject` attribute walking now passes `repr(str)` so string
attributes are quoted (`name='w'`).

## Rendering

1. **Build** a document tree. `PrettyRepr` values, exact `list`, `tuple` and
   `dict`, and dataclasses using the *generated* `__repr__` are expanded
   natively; classes render by `__name__`; array-likes are summarised by
   `format_array_like`; everything else is a `repr()` leaf (multi-line leaves
   force a break).
2. **Layout** (black/Wadler rule): a group is printed flat when its flat width
   fits in the remaining width and nothing inside forces a break. Otherwise it
   is broken one item per line, indented by `len(PrettyType.elem_indent)`,
   with trailing commas. Comments are emitted only on broken lines and never
   count toward width.
3. **Lists/tuples** that contain `PrettyRepr` items get `[i] ` index labels
   when broken.

### Shared and cyclic references

The renderer records the path (`.a`, `[0]`, `['k']`) of each `PrettyRepr`
object it expands. A repeat renders as `<shared: .a>`; a reference to an
ancestor still being rendered renders as `<cycle: .path>` (`.` is the root).
Plain containers and dataclasses get cycle protection only (a repeated plain
list is printed again, not as `<shared>`).

### Array formatting

`jax.Array`, tracers, `np.ndarray`, `ShapedArray`, `ShapeDtypeStruct` render as
`f32[3,4]` (`bool`, `i32`, `u8`, `bf16`, `c64`, ...). 0-d concrete arrays
render as `f32(0.5)`. `brainunit.Quantity` appends its unit: `f32[3] mV`; 0-d
concrete quantities use `str(q)`. With `short_dtypes=False` the long dtype
name (`float32[3,4]`) is used.

## Summary

`Node.__pretty_repr_summary__()` returns a `ReprSummary(n_params,
param_bytes, n_state_values, state_bytes)` computed over the unique `State`s
reachable from the node (`ParamState` → params, other states → state
values). The renderer shows:

- root footer: `)  # 26 params (104 B)`; `, 10 state values (40 B)` appended
  when non-param states exist; omitted when everything is zero;
- child comment on broken parents: `# 16 params` (or `# 10 state values`).

Disabled with `summary=False`.

## Options

`ReprOptions(width=88, max_depth=None, max_items=64, summary=True,
color=False, short_dtypes=True)`.

- `set_repr_options(**kw)` — process-wide default.
- `repr_options(**kw)` — thread-local context manager, nestable, restored on
  exit/exception.
- `max_depth`: number of named-object levels expanded below the root; deeper
  objects render as `Type(...)` (`0` collapses all children). Anonymous
  wrappers (`PrettyList`, `PrettyDict`) do not count as a level.
- `max_items`: longer groups are cut with a `... (+N more)` item.
- `color`: ANSI styling, applied only by `str()` / `print()`, never by
  `repr()`. Disabled whenever `NO_COLOR` is set. `BRAINSTATE_REPR_COLOR=1`
  turns the default on at import.

## Class-specific formats

- `Module`: `in_size`/`out_size` collapse into one leading positional
  `(3,) -> (4,)` when both are set.
- `State`: value is positional: `ParamState(f32[3])`,
  `ParamState(f32[3], name='w')`; a dict value without other metadata uses
  the brace form `ParamState{bias: f32[4], weight: f32[3,4]}`.
- `TreefyState`: value abstracted like `State`; `_hooks_manager` hidden;
  `None` metadata hidden.
- `Map`/`Param` item hooks delegate to `Node.__pretty_repr_item__` first.

## Formerly hand-written reprs

- `nn/_elementwise.py` activations: f-string `__repr__`s removed; the Module
  engine renders them (`LeakyReLU(negative_slope=0.01)`).
- `nn.Transform` and `interop.Transform`: `PrettyRepr` subclasses whose
  subclasses report constructor arguments through `_repr_items()` (`(name,
  value)` pairs, `''` = positional). Fixes `PermuteAxes` printing as
  `Transpose`.
- `HiData`: `PrettyRepr`; prints the real subclass name; long non-structured
  values are still truncated at 60 characters.
- `RandomState`: `RandomState(key<fry>[], data=[0, 42])`, read without hooks.
- `mixin.Mode` family: `Training`, `JointMode(Training, Batching(...))`;
  `Batching` now labels its fields `batch_size`/`batch_axis` (was a
  mislabelled `in_size`). `ParamDescriber` keeps call syntax but formats its
  arguments with `pretty_format`.
- `JointTypes[...]` / `OneOfTypes[...]` keep typing-alias syntax.
- `PrettyRepr.__str__` only uses the engine when `__repr__` is engine-backed
  (`__repr__._uses_pretty_repr = True`), so subclasses with their own
  `__repr__` print consistently.

## Out of scope

HTML/treescope rendering, re-executing notebooks.
