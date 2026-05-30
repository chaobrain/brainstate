# Thinking in BrainState

If you have written JAX or PyTorch, most of BrainState will feel familiar — but the pieces fit
together in a particular way. Three nouns carry the whole framework: **State**, **Module**, and
**Transform**. Hold these three in mind and the rest follows.

## State — a mutable cell

A {class}`~brainstate.State` wraps a value you intend to change. You read it with `.value` and
write it by assigning:

```python
s = brainstate.State(jnp.zeros(3))
s.value = s.value + 1.0
```

This is the one place mutation happens. Everything that needs to change over the life of a
program — parameters, hidden activations, optimizer buffers, running statistics — lives inside a
`State`. The *kind* of state is expressed by its class: `ParamState` for trainable parameters,
`HiddenState` for dynamical state, and so on. That class is not decoration; it is how you later
tell a transformation *which* states to act on.

## Module — a tree of state

A {class}`~brainstate.nn.Module` is an object that holds states and sub-modules as attributes. A
model is therefore a tree: modules nesting modules, with `State` objects at the leaves. You do not
register parameters by hand — assigning a `State` (or a sub-module that contains states) as an
attribute is enough for BrainState to find it.

The key operation on a module is *selecting* its states by type:

```python
params = model.states(brainstate.ParamState)   # just the trainable parameters
```

This returns a flat collection keyed by each state's path in the tree. Selection by type is the
idiom you will use constantly — it is how "optimize the weights, leave the buffers alone" is
expressed.

## Transform — state-aware jit, grad, vmap

BrainState's transformations mirror JAX's, but they understand `State`. Hand a model to
{func}`~brainstate.transform.jit` and its state reads and writes are threaded through the compiled
function automatically — no manual plumbing, and no silently-discarded updates.
{func}`~brainstate.transform.grad` differentiates with respect to a *collection of states* rather
than positional arguments, returning gradients keyed the same way as the states you passed.
{func}`~brainstate.transform.vmap` adds a batch axis, sharing or mapping each state as appropriate.

This is the rule worth internalizing: **write ordinary code that reads and writes `.value`, then
wrap it in a BrainState transform.** Reaching for raw `jax.jit` on stateful code is the common
first mistake — it traces the mutation once and throws it away.

## The shape of every program

Almost every BrainState training program is the same five lines, repeated:

```python
model = MyModule(...)                                  # 1. build a tree of state
params = model.states(brainstate.ParamState)           # 2. select what to train

@brainstate.transform.jit                              # 5. compile the step
def train_step(x, y):
    grads = brainstate.transform.grad(loss_fn, params)(x, y)   # 3. differentiate w.r.t. params
    for key in params:                                          # 4. update in place
        params[key].value = jax.tree.map(lambda p, g: p - lr * g, params[key].value, grads[key])
    return loss_fn(x, y)
```

Build state, select it, differentiate, update, compile. Once this loop is automatic, the rest of
the library is variations on it.

## Going deeper

- [Why state-based?](../concepts/why_state_based.md) — the rationale for this design.
- [The state model](../concepts/the_state_model.md) and [transformation semantics](../concepts/transformation_semantics.md) — how it works underneath.
- [Core tutorials](../tutorials/core/index.rst) — each piece, hands-on and in order.
