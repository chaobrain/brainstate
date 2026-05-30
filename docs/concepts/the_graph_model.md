# The Graph Model

A BrainState model is an ordinary Python object graph: modules hold sub-modules in attributes,
lists, and dicts, and {class}`~brainstate.State` objects sit at the leaves. The graph model is the
set of operations that let BrainState walk this structure, pull the state out as plain data,
transform it, and put it back — turning a live, mutable object into something JAX can consume and
then reconstructing the object from the result.

## Static structure versus dynamic state

The central operation is {func}`~brainstate.graph.treefy_split`, which decomposes a model into two
parts:

```python
graphdef, states = brainstate.graph.treefy_split(model)
```

- **`graphdef`** captures the *static* structure: the classes, the attribute names, how nodes are
  nested, and which leaves are states. It contains no array data.
- **`states`** is a mapping from each state's path in the graph to its value, as plain nested
  data.

{func}`~brainstate.graph.treefy_merge` is the exact inverse: given a `graphdef` and a matching
state mapping, it rebuilds a live model.

```python
model = brainstate.graph.treefy_merge(graphdef, states)
```

This split is the bridge between two worlds. On one side is the object-oriented model you write
and read. On the other is the pure, PyTree-shaped data that JAX transformations require. Split to
cross into JAX's world, transform the state as a value, merge to come back. A model can be taken
apart and reassembled any number of times; the round trip is lossless and the reconstructed
states are immediately usable.

## Why a graph and not a tree

A naive flatten would treat the model as a tree and copy every object it reaches. Models are not
trees. Two layers may share the same parameter; a module may hold a reference to another module
that also appears elsewhere. If shared references were duplicated on split and not re-shared on
merge, the rebuilt model would silently have two independent copies where the original had one,
and an update to one would not be seen by the other.

The graph model tracks object identity. A node reached by two paths is recorded once and restored
as a single shared object, so aliasing and (where present) cycles survive the round trip. This
fidelity is what makes it safe to split a model, optimize its parameters as a flat collection, and
merge the result back into the same shared structure.

## Selecting and filtering

Most of the time you do not need the full split — you need a subset of states chosen by type or
location. {func}`~brainstate.graph.states` (and the `Module.states` method) returns the states of
a model, optionally filtered:

```python
params = brainstate.graph.states(model, brainstate.ParamState)
```

Filters compose the same way throughout the framework, because a state collection is just data:
nested dictionaries keyed by path, which you can inspect, partition, and recombine with ordinary
operations. This is the foundation that {doc}`transformation_semantics` builds on — every
state-aware transform is, underneath, a `treefy_split`, a pure transformation of the state data,
and a `treefy_merge`.

## See also

- {doc}`the_state_model` — what lives at the leaves of the graph.
- {doc}`transformation_semantics` — how split/merge powers `jit`, `grad`, and `vmap`.
- {doc}`../how_to/inspect_and_edit_state_graph` — splitting, filtering, and editing a model graph.
- {doc}`../tutorials/core/02_modules_and_graph` — building models as graphs of modules.
