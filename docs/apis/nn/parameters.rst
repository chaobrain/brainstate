Parameter containers
====================

.. currentmodule:: brainstate.nn

This page lists exact signatures. Four related documents cover different ground:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Document
     - What it gives you
   * - :doc:`../../tutorials/core/05_parameters_transforms_regularization`
     - The guided tour, with three worked models
   * - :doc:`../../how_to/choose_parameter_transforms`
     - The transform catalog, by constrained domain
   * - :doc:`../../how_to/constrain_and_regularize_parameters`
     - Short task recipes
   * - :doc:`../../concepts/the_parameter_model`
     - The *why* behind the design

1. Parameter containers
-----------------------

``Param`` wraps a trainable array with an optional bijective transform for constrained
optimization and an optional regularization penalty. The value passed to the constructor is
interpreted in *constrained* space; the inverse transform derives the unconstrained array that
is actually stored and updated.

``Const`` is the non-trainable counterpart: it participates in the forward pass but is excluded
from ``ParamState`` collection, so optimizers and ``grad`` never see it.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Param
   Const

2. Model-level parameter API
----------------------------

These methods live on :class:`Module` and operate across the whole module graph: discovering
parameters, aggregating their penalties, and scoping their caches.

.. important::

   All four methods accept **only** ``allowed_hierarchy=(min_depth, max_depth)`` — a closed
   interval, where a ``Param`` attached directly to the root counts as depth 1. **They do not
   accept a filter function.** Filter the result with an ordinary list comprehension::

       regularized = [(n, p) for n, p in model.named_param_modules() if p.reg is not None]
       trainable   = [p for p in model.param_modules() if p.fit]

   Traversal includes ``Const``, since it is a ``Param`` subclass; check ``.fit`` when
   trainability is what you mean.

.. autosummary::
   :nosignatures:

   Module.param_modules
   Module.named_param_modules
   Module.reg_loss
   Module.param_precompute

The per-parameter counterparts on ``Param`` itself:

.. autosummary::
   :nosignatures:

   Param.value
   Param.set_value
   Param.reg_loss
   Param.cache
   Param.clear_cache
   Param.cache_stats
   Param.clip
   Param.reset_to_prior
   Param.init

.. note::

   ``Param.value()`` does not populate the cache. The cache is opt-in: only ``Param.cache()`` or
   ``Module.param_precompute()`` warms it. This prevents a value traced inside ``jit`` from being
   cached and reused outside that trace.

3. Parameter transforms
-----------------------

Bijections between an unconstrained space and a constrained one, letting an optimizer work
without walls while the constraint holds by construction. All implement ``forward()``,
``inverse()``, and optionally ``log_abs_det_jacobian()`` for probabilistic applications. Pass one
to ``Param`` as ``t=``.

Base class and the no-op default:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Transform
   IdentityT

Positive and negative half-lines:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   SoftplusT
   ExpT
   PositiveT
   ReluT
   NegSoftplusT
   NegativeT
   LogT

Bounded intervals:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   SigmoidT
   ScaledSigmoidT
   ClipT
   TanhT
   SoftsignT

Structured domains:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   SimplexT
   UnitVectorT
   OrderedT

Reparameterizations and composition:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   AffineT
   PowerT
   ChainT
   MaskedT

4. Standard regularizations
---------------------------

Classical penalties encouraging sparsity, smoothness, or structural properties such as
orthogonality and bounded spectral norm. Pass one to ``Param`` as ``reg=``;
``Module.reg_loss()`` sums them across the model.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   Regularization
   L1Reg
   L2Reg
   ElasticNetReg
   HuberReg
   GroupLassoReg
   TotalVariationReg
   MaxNormReg
   EntropyReg
   OrthogonalReg
   SpectralNormReg
   ChainedReg

5. Prior distribution-based regularizations
-------------------------------------------

Probabilistic penalties derived from a prior distribution, for Bayesian-inspired parameter
estimation: variational inference, maximum a posteriori estimation, and uncertainty
quantification. Each contributes the negative log-density of the parameter under its prior, and
implements ``loss()``, ``sample_init()``, and ``reset_value()`` for prior-based initialization.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: classtemplate.rst

   GaussianReg
   StudentTReg
   CauchyReg
   UniformReg
   BetaReg
   LogNormalReg
   ExponentialReg
   GammaReg
   InverseGammaReg
   LogUniformReg
   HorseshoeReg
   SpikeAndSlabReg
   DirichletReg

6. Use cases
------------

Common patterns, each a minimal snippet you can lift into a model. All follow the same rule:
``.value()`` returns the parameter in its **constrained** domain (use it in the forward pass),
while ``.val`` is the underlying ``ParamState`` the optimizer updates in **unconstrained** space.

**A positive-only rate or time constant.** ``SoftplusT`` keeps the value above a floor no matter
where the optimizer lands, so a membrane time constant or a firing rate never goes negative::

    tau = Param(jnp.array(20.0), t=SoftplusT(lower=1.0))   # ms, stays > 1.0
    dv = (-v + i_input) / tau.value()

**A mixing weight or gate bounded to [0, 1].** ``SigmoidT`` maps the whole real line into an open
interval, so a convex-combination coefficient cannot leave its range::

    alpha = Param(jnp.array(0.5), t=SigmoidT(lower=0.0, upper=1.0))
    blended = alpha.value() * fast + (1.0 - alpha.value()) * slow

**A learned categorical distribution.** ``SimplexT`` guarantees ``.value()`` is a valid
probability vector (non-negative, sums to one) for any unconstrained input — handy for a learned
prior over components or a soft attention/routing weight::

    weights = Param(jnp.zeros(4), t=SimplexT())
    mixture = jnp.tensordot(weights.value(), components, axes=1)

**Weight decay and sparsity.** Attach a penalty with ``reg=`` and add ``.reg_loss()`` to the data
loss. ``L2Reg`` shrinks weights smoothly; ``L1Reg`` drives them to exact zeros::

    dense = Param(random.randn(din, dout) * 0.1, reg=L2Reg(weight=1e-4))
    gate  = Param(random.randn(dout), reg=L1Reg(weight=1e-3))

**A Bayesian / MAP prior on a parameter.** A prior-based regularization contributes the negative
log-density of the parameter under a prior, turning training into maximum-a-posteriori
estimation::

    mu = Param(jnp.zeros(dout), reg=GaussianReg(mean=0.0, std=1.0))

**Aggregating penalties across a whole model.** ``Module.reg_loss()`` sums every penalty in the
module tree in one call, so the training step never has to enumerate parameters by hand::

    def loss_fn(batch):
        data_loss = mse(model(batch.x), batch.y)
        return data_loss + model.reg_loss()

**Warming caches for constrained parameters.** When many parameters share expensive transforms,
call ``Module.param_precompute()`` once outside the hot loop; ``.value()`` then reads the cached
constrained array instead of recomputing the bijection on every forward pass::

    model.param_precompute()      # warm every Param cache in the tree
    for batch in loader:
        predictions = model(batch.x)
