# Why State-Based?

JAX is built on *pure functions*. A transformation like `jit`, `grad`, or `vmap` assumes that a
function's output depends only on its explicit inputs and that calling it produces no side
effects. This assumption is what lets JAX trace a function once, reason about it as a
mathematical object, and rewrite it freely — fuse it, differentiate it, vectorize it, or shard it
across devices.

Real models violate that assumption constantly. A network has parameters that must be updated. A
recurrent layer carries hidden state across time steps. A spiking neuron accumulates membrane
voltage and refractory counters. An optimizer maintains momentum buffers. All of this is *mutable
state*, and it sits awkwardly on top of a functional substrate.

## The two unsatisfying options

Faced with this tension, most frameworks pick one of two strategies.

The first is to **thread every piece of state through function arguments by hand**. This is the
honest, fully-functional approach, and it is what raw JAX encourages. It also scales badly: a
training step for a modest model becomes a function with a dozen parameters in and a dozen out,
the order matters, and a single misplaced return value produces a silent correctness bug rather
than an error. The mechanics of plumbing state crowd out the model you are actually trying to
express.

The second is to **hide state inside an object system** and mutate it eagerly, the way PyTorch
does. This reads naturally — `self.weight += update` — but it forfeits the transformations that
make JAX valuable. The moment you wrap an eagerly-mutating object in `jax.jit`, the mutation
happens during tracing and is then silently discarded on every subsequent call.

## BrainState's answer

BrainState keeps both halves: **state is explicit and mutable, and transformations are
state-aware.**

A {class}`~brainstate.State` is an explicit container holding a mutable value. You read it with
`state.value` and write it with `state.value = ...`, so model code reads like ordinary imperative
code. But because the state is a first-class object rather than a hidden attribute, BrainState's
transformations — {func}`~brainstate.transform.jit`, {func}`~brainstate.transform.grad`,
{func}`~brainstate.transform.vmap` — can discover which states a computation reads and writes, and
thread them through the underlying pure-functional JAX machinery automatically.

The result is code that *feels* mutable but *compiles* as if you had threaded everything by hand:

```python
class Counter(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = brainstate.State(jnp.array(0))

    def __call__(self):
        self.n.value += 1
        return self.n.value

counter = brainstate.transform.jit(Counter())
[int(counter()) for _ in range(4)]   # 1, 2, 3, 4 — the write persists
```

The same model handed to raw `jax.jit` returns `1, 1, 1, 1`: the increment is traced once and
then discarded. The difference is not the model — it is that BrainState's `jit` understands
`State`.

## Why *explicit* state

Making state an explicit object, rather than an implicit global or a hidden attribute, is a
deliberate design choice with three consequences:

- **Traceability.** Every transformation can enumerate exactly which states a function touches.
  This is what makes `grad` differentiate "with respect to these parameters" precise, and what
  lets `vmap` decide which states to batch and which to share.
- **Composability.** Because states are values in a graph (see {doc}`the_graph_model`), they can
  be filtered, split, merged, checkpointed, and partitioned with ordinary data operations rather
  than framework-specific magic.
- **Correctness under transformation.** State reads and writes are captured at well-defined
  points, so a mutation inside a `jit`-compiled step behaves the same as one outside it.

State-based programming is therefore not a convenience layer bolted onto JAX. It is the bridge
that lets a single program be both an imperative model you can read and a pure function JAX can
transform.

## See also

- {doc}`the_state_model` — the kinds of state and their value semantics.
- {doc}`transformation_semantics` — how state is threaded through `jit`, `grad`, and `vmap`.
- {doc}`../tutorials/core/01_state_and_pytrees` — a hands-on introduction to `State`.
