# The State Model

A {class}`~brainstate.State` is a typed, mutable container for a value. The value is read through
the `.value` property and replaced by assigning to it:

```python
s = brainstate.State(jnp.zeros(3))
s.value = s.value + 1.0
```

Everything else about the state model follows from two questions: *what may the value be*, and
*what does the state's type mean*.

## Value semantics

The value of a state is an arbitrary **PyTree of arrays** — a single array, a tuple of arrays, a
nested dict, and so on. This matters because JAX transformations operate on PyTrees, so a state
can hold exactly the structure a layer needs (a weight and a bias together, a list of per-layer
buffers) and still participate in tracing as a unit.

Writes are expected to be **structure-preserving**. Replacing the value with a PyTree of the same
shape and dtype is always safe. Changing the shape, dtype, or tree structure of a state's value
is, in general, incompatible with the fact that the state may already have been traced into a
compiled function — the compiled code was specialized to the old shapes. BrainState treats the
first value a state holds as defining its abstract signature; later writes are checked against it.
The practical rule is simple: a state is a fixed-shape slot you overwrite, not a variable you
reshape.

## The type is a label, not a mechanism

`State` has a family of subclasses — {class}`~brainstate.ParamState`,
{class}`~brainstate.ShortTermState`, {class}`~brainstate.LongTermState`,
{class}`~brainstate.HiddenState`, {class}`~brainstate.BatchState`, and others. They share the same
read/write machinery. What differs is *meaning*, and that meaning is used as a **filter key**.

| Type | Conventional role |
|---|---|
| `ParamState` | Trainable parameters — what an optimizer updates and `grad` differentiates. |
| `HiddenState` | Dynamical/recurrent state — membrane voltages, hidden activations, anything that evolves over time. |
| `ShortTermState` | Transient values that live for a single step, such as synaptic currents. |
| `LongTermState` | Persistent buffers that accumulate across steps, such as running normalization statistics. |
| `BatchState` | Values whose leading axis is a batch dimension. |

Selecting states by type is the central idiom of the framework. `model.states(ParamState)`
returns just the trainable parameters; differentiating with respect to that collection is how a
training step expresses "optimize the weights, leave the buffers alone." The type system is
therefore not bookkeeping — it is how you tell transformations *which* states to act on.

```python
params = model.states(brainstate.ParamState)      # trainable weights only
hidden = model.states(brainstate.HiddenState)      # dynamical state only
```

## Tracing and trace levels

A state created inside one JAX trace must not be written from a different trace context. If it
could, a value computed under one `jit` (an abstract tracer) might leak into another, producing
results that are silently wrong. To prevent this, each state records the trace context it belongs
to and rejects cross-context writes. This is the same class of protection JAX provides against
"leaked tracers," lifted to the level of the `State` abstraction. You will only encounter it if a
state escapes the scope it was meant for; in normal use it is invisible.

A consequence worth knowing: when a state is decomposed for a transformation and later
reconstructed (see {doc}`the_graph_model`), the rebuilt state is given a *fresh* trace identity, so
it is immediately usable again. The round trip is transparent.

## The state lifecycle and hooks

Every state participates in a small set of lifecycle operations: it is **created** (`init`),
**read**, **written** (with a `write_before` point just before the value changes and a
`write_after` point just after), and optionally **restored** from a checkpoint. BrainState exposes
these as *hooks* — callbacks you can register against any of those operations, globally or for a
single state.

Hooks make cross-cutting behavior possible without editing model code: logging every change,
validating that a write stays in range, enforcing an invariant by rewriting or rejecting a value.
A `write_before` hook can transform the incoming value or cancel the write entirely; the other
hook points are read-only observers. Because hooks are ordinary Python callbacks, they belong to
eager execution and debugging — checks that must run *inside* compiled code use the error-handling
tools instead.

## See also

- {doc}`the_parameter_model` — how `ParamState` is wrapped to add constraints and regularization.
- {doc}`the_graph_model` — how states are discovered, filtered, split, and merged.
- {doc}`../how_to/state_hooks` — registering and managing lifecycle hooks.
- {doc}`../tutorials/core/01_state_and_pytrees` — `State` and PyTrees in practice.
