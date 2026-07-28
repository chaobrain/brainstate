# The parameter model

This page explains *why* the parameter system is shaped the way it is. Related documents cover
different ground:

| Document | What it gives you |
| --- | --- |
| {doc}`../tutorials/core/05_parameters_transforms_regularization` | The guided tour, with three worked models |
| {doc}`../how_to/choose_parameter_transforms` | The transform catalog, by constrained domain |
| {doc}`../how_to/constrain_and_regularize_parameters` | Short task recipes |
| {doc}`../apis/nn/parameters` | Exact signatures |
| {doc}`the_state_model` | `ParamState` and the broader state type system |

A {class}`~brainstate.ParamState` is a bare trainable container: an array an optimizer is free to
move anywhere in $\mathbb{R}^n$. That is often not what a model means. A rate must be positive, a
mixing coefficient must lie in $[0, 1]$, a categorical distribution must sum to one, and a weight
matrix may need a prior that discourages large values. {class}`~brainstate.nn.Param` exists to
express these intentions declaratively, layering two orthogonal concerns on top of `ParamState`: a
**constraint transform** and a **regularization prior**.

## 1. Constraints without fighting the optimizer

The naive way to keep a parameter positive is to clip it after every update. This works against
the optimizer: at the boundary the gradient is discarded, momentum is corrupted, and the
parameter sticks to the wall. The damage grows with the number of constrained parameters.

The principled alternative is a **change of variables**. Keep an unconstrained parameter
$\theta \in \mathbb{R}$, choose a smooth invertible map $T$ onto the valid domain, and have the
model use $T(\theta)$. The optimizer works in the unconstrained space, where there are no walls
and the loss surface is well conditioned; the constraint is satisfied by construction because
$T(\theta)$ can never leave its range. Gradients flow through $T$ by the chain rule, so learning
is unobstructed.

`Param` implements exactly this. The value you read back, `param.value()`, is the *constrained*
value $T(\theta)$ — use it in the forward pass. The underlying unconstrained parameter,
`param.val`, is the `ParamState` the optimizer updates. The two are kept in correspondence by the
transform:

```python
rate = brainstate.nn.Param(jnp.array(0.5), t=brainstate.nn.SoftplusT(lower=0.0))
rate.value()          # always > 0, whatever the optimizer does to rate.val
```

Reading `.value()` is a method call, not an attribute, precisely because it *computes* the
forward transform each time rather than storing a constrained copy that could drift out of sync.

The transform catalogue covers the common domains, and transforms compose:


| Transform                         | Maps$\mathbb{R}$ (or $\mathbb{R}^n$) onto            |
| --------------------------------- | ---------------------------------------------------- |
| `SoftplusT(lower)`, `ExpT(lower)` | $(\text{lower}, \infty)$ — positive quantities      |
| `SigmoidT(lower, upper)`          | $(\text{lower}, \text{upper})$ — bounded scalars    |
| `SimplexT()`                      | the probability simplex — non-negative, sums to one |
| `AffineT(scale, shift)`           | a linear reparameterization                          |
| `ChainT(t1, t2, ...)`             | the composition$t_1 \circ t_2 \circ \cdots$          |

## 2. Regularization as a prior

A regularization term expresses a *preference* over parameter values — a prior, in the Bayesian
reading, whose log-density is added to the data loss. Minimizing data loss plus penalty is then
maximum-a-posteriori estimation. The two classical choices correspond to familiar priors: an
$L_2$ penalty is a zero-mean Gaussian prior (weight decay), and an $L_1$ penalty is a Laplace
prior that drives parameters exactly to zero (sparsity).

`Param` attaches a prior through `reg=`, and `param.reg_loss()` returns that parameter's scalar
contribution. You sum these into the objective alongside the data term:

```python
w = brainstate.nn.Param(weights, reg=brainstate.nn.L2Reg(weight=1e-3))
loss = data_loss + w.reg_loss()
```

Keeping the penalty attached to the parameter, rather than recomputed in the loss function, means
the prior travels with the parameter it constrains — a layer that owns a regularized weight needs
no cooperation from the training loop beyond summing `reg_loss()`.

## 3. Constants

Not every value in a computation should be learned. {class}`~brainstate.nn.Const` wraps a value
that participates in the forward pass but is deliberately excluded from the `ParamState`
collection, so optimizers and `grad` never see it. It is the right tool for a fixed scale, a
lookup table, or any quantity you want frozen — clearer than a parameter you must remember not to
update.

## 4. Parameters form a tree

Everything above concerns *one* parameter: it carries its own transform, its own prior, and it
knows how to report its own penalty. A real model has dozens of them, nested inside submodules —
an excitatory population inside a network, a projection inside that.

The nesting is not incidental. `Param` **is a** {class}`~brainstate.nn.Module`, so a parameter is
a node in the same graph as everything else. Traversal is therefore not a bolted-on utility; it is
the natural way to ask a model about its parameters. Three questions arise as soon as there is
more than one parameter, and each has a one-call answer.

**What parameters does this model have?** {meth}`~brainstate.nn.Module.named_param_modules` walks
the graph and yields each parameter together with its dotted path — `exc.tau`, `syn_ee.w`.
Identity comes from structure, so there is no registry to keep in sync, and the path is already a
usable checkpoint key. {meth}`~brainstate.nn.Module.param_modules` gives the same walk without the
names. Both traverse `Const` as well, since it is a `Param` subclass; consult `.fit` when
trainability is what you mean. Neither takes a filter argument — narrow the result with an
ordinary list comprehension.

**What is the total penalty?** Because each prior travels with the parameter it constrains — the
point of the previous section — the model-level penalty is just their sum over the graph.
{meth}`~brainstate.nn.Module.reg_loss` performs that sum in one call, at any nesting depth:

```python
loss = data_loss + model.reg_loss()
```

Doing it by hand is possible but carries a hidden assumption: you have to know that the traversal
recurses. Restricting it to the top level looks reasonable and silently returns a smaller number,
with no error raised.

**How do I avoid recomputing transforms in a loop?** Here an earlier design decision comes due.
`value()` recomputes the forward transform on every call rather than storing a constrained copy —
which invites the obvious objection that a simulation reading the same parameter at every timestep
pays for it repeatedly.

The obvious fix — caching inside `value()` — would be wrong. Under `jit` the first call happens
during tracing, so the cached value would be a tracer, and it would leak into later calls where it
means something different. The cache is therefore opt-in, and
{meth}`~brainstate.nn.Module.param_precompute` gives it a **scope**: computed on entry, reused
throughout the block, discarded on exit, including when the block exits through an exception.
Efficiency inside the block, with no stale value escaping it. That is why it is a context manager
rather than a flag.

For runnable before-and-after comparisons of all four methods, see sections 4 and 5 of
{doc}`../tutorials/core/05_parameters_transforms_regularization`.
