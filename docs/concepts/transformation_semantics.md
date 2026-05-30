# Transformation Semantics

BrainState's transformations — {func}`~brainstate.transform.jit`,
{func}`~brainstate.transform.grad`, {func}`~brainstate.transform.vmap`, and the rest — share one
underlying recipe. Understanding it once explains all of them.

## The unifying recipe

A JAX transformation needs a pure function of PyTrees. A BrainState computation is an impure
function that reads and writes `State` objects. Every state-aware transform closes that gap the
same way:

1. **Split.** Before running, identify the states the computation will touch and read their
   current values out as plain data (a `treefy_split`, conceptually — see {doc}`the_graph_model`).
2. **Transform a pure core.** Build a pure function whose inputs include those state values and
   whose outputs include their new values, and hand *that* to the underlying JAX transform.
3. **Merge.** Write the returned state values back into the live `State` objects.

So a state-aware transform is the JAX transform sandwiched between a split and a merge. The
mutable-looking model on the outside and the pure function on the inside are two views of the same
computation.

## Reads and writes are tracked separately

To build the pure core correctly, a transform must distinguish the states a computation *reads*
(which become inputs) from those it *writes* (which become outputs). A state that is only read is
threaded in but not out; a state that is written is threaded in *and* out so the new value
survives. BrainState discovers this read/write set by tracing the computation once and recording
which states are accessed and how. This analysis is why a mutation inside a compiled step is not
lost: the written state was promoted to an output of the pure function and merged back afterward.

This is the precise mechanism behind the contrast in {doc}`why_state_based`. Raw `jax.jit` sees no
state outputs, so the write vanishes; BrainState's `jit` sees the write, makes it an output, and
the value persists.

## How each transform specializes the recipe

**`jit`** compiles the pure core with XLA and caches it. Read states enter as inputs, written
states leave as outputs and are merged back. The model's mutations therefore behave identically
whether or not the step is compiled — the only difference is speed.

**`grad`** differentiates with respect to a *collection of states* rather than positional
arguments. You pass the states to differentiate (typically `model.states(ParamState)`), and `grad`
returns a mapping from each state's path to its gradient. The states are the differentiation
variables; the function itself can take ordinary arguments too. This is why a training step reads
as "differentiate the loss with respect to these parameters" rather than "differentiate with
respect to argument three."

**`vmap`** adds a batch axis. Because it is state-aware, it can treat each state in one of two
ways: *broadcast* it, so a single shared copy serves the whole batch (the usual choice for
parameters), or *map* over it, so each batch element gets its own copy (the choice for per-example
state, or for running an ensemble of models in parallel). The `in_states` / `out_states` controls
select which states are mapped and which are shared.

## Composition

Because each transform consumes and produces the same kind of state-aware computation, transforms
compose. The backbone of essentially every training loop is `jit(grad(...))`: differentiate the
loss with respect to the parameters, then compile the entire gradient-and-update step into one
fused kernel.

```python
@brainstate.transform.jit
def train_step(x, y):
    grads = brainstate.transform.grad(loss_fn, params)(x, y)
    for key in params:
        params[key].value -= lr * grads[key]
    return loss_fn(x, y)
```

The split/merge happens at the boundary of each transform, automatically, so the code you write
stays at the level of the model.

## See also

- {doc}`the_graph_model` — the split/merge machinery transforms are built on.
- {doc}`why_state_based` — why state-aware transforms are necessary at all.
- {doc}`../tutorials/core/06_transformations_essentials` — `jit`, `grad`, and `vmap` hands-on.
- {doc}`../tutorials/transformations/index` — each transformation in depth.
