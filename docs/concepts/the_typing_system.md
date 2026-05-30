# The Typing System

Scientific JAX code moves a small set of recurring shapes around: arrays with expected
dimensions, nested PyTrees, random keys, dtypes, and the *filters* used to select states from a
model. {mod}`brainstate.typing` gives these a shared vocabulary of type aliases. They are
annotations, not runtime checks — they cost nothing at execution time and exist to communicate
intent to readers and to static type checkers.

## Arrays, shapes, and axes

`Size`, `Shape`, and `Axes` annotate the dimensional arguments that pervade array APIs — a target
shape, the axes to reduce over. They are thin aliases around Python sequences, but a parameter
typed `along: Axes` says far more about intent than `along: int | tuple[int, ...]`.

For the arrays themselves, two aliases divide the space by *direction of flow*. `ArrayLike`
describes an **input** the function will accept: anything convertible to a JAX array — a Python
scalar, a NumPy array, a list, and notably a unit-carrying `brainunit.Quantity`. `Array` describes
an **output** or an internal array the function produces. The convention is to accept `ArrayLike`
and convert once at the boundary with `jnp.asarray`, keeping signatures permissive without losing
clarity about what is produced:

```python
def sum_energy(signal: ArrayLike) -> Array:
    arr = jnp.asarray(signal)
    return jnp.sum(arr ** 2)
```

`Array` also supports symbolic shape annotations (`Array["rows, cols"]`) for documenting the shape
contract of a function — purely informational, but a useful form of executable-looking comment.

## PyTrees

`PyTree` behaves like `typing.Any` for the type checker, but `PyTree[jax.Array]` documents the
expected *leaf* type. Utilities that operate on nested containers — anything that flattens a tree,
maps over leaves, or reduces across them — read far better when their signature states that the
leaves are arrays rather than leaving it to the reader to infer.

## Dtypes and random keys

`DType`, `DTypeLike`, and `SupportsDType` annotate dtype arguments, mirroring the array/array-like
split at the dtype level. For randomness, `SeedOrKey` enumerates the accepted sources of
entropy — a plain `int`, a JAX PRNG key, or a NumPy key — so a function can advertise that it will
normalize whatever form of seed it is given. `Key` annotates a single path component or key.

## Filters: the link to the graph model

The most BrainState-specific aliases are the ones that describe *selection*. `Filter`,
`Predicate`, and `PathParts` are the types of the arguments you pass when choosing states from a
model — by type, by path, or by an arbitrary predicate. A filter is exactly what `model.states(...)`
consumes:

```python
params = model.states(brainstate.ParamState)   # ParamState is a Filter
```

Typing these arguments connects the type system to the graph model ({doc}`the_graph_model`): state
selection is a first-class, well-typed operation, not an ad-hoc convention. `Missing` rounds out
the set — a sentinel for "no value supplied" in the cases where `None` is itself a meaningful
argument.

## See also

- {doc}`the_graph_model` — where filters are used to select states.
- The {doc}`typing API reference <../apis/typing>`.
