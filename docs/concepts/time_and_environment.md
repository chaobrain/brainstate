# Time and Environment

Some quantities are not properties of any one layer but of the *run*: the simulation time step,
whether the model is training or evaluating, the numerical precision, the target platform. Passing
these through every function call would drown the model in plumbing. BrainState collects them in a
single scoped context, {mod}`brainstate.environ`.

## A context, not a global

`environ` is a stack of settings with two ways to populate it. {func}`~brainstate.environ.set`
installs a global default; {func}`~brainstate.environ.context` pushes settings for the duration of
a `with` block and pops them on exit. Inside a computation, code reads what it needs with
{func}`~brainstate.environ.get` (or a typed accessor like
{func}`~brainstate.environ.get_dt`).

```python
with brainstate.environ.context(dt=0.1 * u.ms, fit=True):
    run_one_epoch(model)
```

Scoping matters: a nested context overrides an outer one only within its block, so a training loop
can set `fit=True` around the update and `fit=False` around evaluation without any global mutation
leaking between them. A key that has neither a global default nor a surrounding context raises
rather than guessing — unset state is an error, not a silent zero.

## Time and the discrete step

Dynamical models — spiking neurons, synapses, rate equations — advance in discrete steps of
duration $\Delta t$. That step appears in essentially every update rule (an Euler step is
$x \leftarrow x + \Delta t \, f(x)$), so it is the canonical example of something that belongs in
the environment rather than in every signature. A neuron's update reads it with
`environ.get_dt()`; a simulation sets it once with `context(dt=...)`. The well-known key is
`environ.DT`.

Time steps carry physical units (`0.1 * u.ms`), which is how BrainState keeps simulations
dimensionally consistent: a rate in $\text{mV}/\text{ms}$ multiplied by a $\text{ms}$ step yields a
$\text{mV}$ increment, and a unit mismatch surfaces as an error instead of a wrong number. The
step index of a running simulation is available under `environ.I`.

## Training mode

Layers such as dropout and batch normalization behave differently while training than while
evaluating. Rather than carry a `training` flag through the model, they read the `fit` setting
(`environ.FIT`) from the environment: under `fit=True` dropout drops and batch-norm updates its
running statistics; under `fit=False` both pass through deterministically. The training loop sets
the mode once, and every layer observes it consistently.

## Precision and platform

The environment also records numerical precision (`environ.get_precision`, which determines the
default floating-point width) and the active platform (`environ.get_platform`). Centralizing these
means a single setting governs the whole program rather than scattered per-array dtype choices.
Custom keys can be given defaults with {func}`~brainstate.environ.register_default_behavior`, so a
subsystem can extend the environment with its own configuration in the same uniform way.

## See also

- {doc}`../tutorials/brain_dynamics/index` — simulations that drive `dt` and the step index.
- The {doc}`environ API reference <../apis/environ>`.
