# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Shared, internal utilities for the ``brainstate.transform`` IR (jaxpr) modules.

This module is intentionally **not** part of the public API. It consolidates
logic that was previously duplicated across ``_ir_optim`` and ``_ir_tocode``
(identity-keyed collections and the constant-folding engine) and provides
common validation helpers and an exception hierarchy used throughout the IR
utilities.
"""

import inspect
import itertools
from collections.abc import MutableSet, MutableMapping

import numpy as np

from brainstate._compatible_import import Literal, Var, Jaxpr, ClosedJaxpr

__all__ = []  # internal module; nothing here is part of the public API


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class IRError(Exception):
    """Base class for all brainstate IR utility errors."""
    __module__ = 'brainstate.transform'


class IRValidationError(IRError, ValueError):
    """Raised when an IR input is malformed.

    Subclasses :class:`ValueError` so existing ``except ValueError`` callers
    continue to work unchanged while allowing callers to catch the narrower
    :class:`IRError`.
    """
    __module__ = 'brainstate.transform'


class UnsupportedPrimitiveError(IRError):
    """Raised when a primitive or control-flow construct cannot be handled."""
    __module__ = 'brainstate.transform'


# ---------------------------------------------------------------------------
# Identity-keyed collections
# ---------------------------------------------------------------------------

class IdentitySet(MutableSet):
    """A set that compares elements by identity (``id()``) rather than equality.

    Useful for storing JAX ``Var`` objects, which compare by identity and may
    be unhashable in some configurations.

    Examples
    --------
    >>> s = IdentitySet()
    >>> a = [1, 2, 3]
    >>> b = [1, 2, 3]
    >>> s.add(a)
    >>> a in s
    True
    >>> b in s  # Different object, even though equal
    False
    """
    __module__ = 'brainstate.transform'

    def __init__(self, iterable=None):
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, value):
        return id(value) in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[id(value)] = value

    def discard(self, value):
        self._data.pop(id(value), None)

    def update(self, iterable):
        """Add all elements from ``iterable`` to the set."""
        for item in iterable:
            self.add(item)

    def __repr__(self):
        return f"IdentitySet({[repr(x) for x in self._data.values()]})"

    def __str__(self):
        return f"IdentitySet({[str(x) for x in self._data.values()]})"


class IdentityMap(MutableMapping):
    """A mapping keyed by the identity (``id()``) of the key object.

    Iteration yields keys (standard mapping semantics).
    """
    __module__ = 'brainstate.transform'

    def __init__(self):
        self._data = {}  # id(key) -> (key, value)

    def __getitem__(self, key):
        return self._data[id(key)][1]

    def __setitem__(self, key, value):
        self._data[id(key)] = (key, value)

    def __delitem__(self, key):
        del self._data[id(key)]

    def __contains__(self, key):
        return id(key) in self._data

    def __iter__(self):
        return (k for (k, _v) in self._data.values())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        body = {repr(k): repr(v) for (k, v) in self._data.values()}
        return f"IdentityMap({body})"


# ---------------------------------------------------------------------------
# Fresh-variable generation (version-robust)
# ---------------------------------------------------------------------------

def make_var_factory():
    """Return a callable ``factory(aval) -> Var`` that builds fresh variables.

    JAX's ``Var`` constructor signature has changed across versions:

    * modern (>= ~0.7): ``Var(aval, initial_qdd=None, final_qdd=None)``
    * ~0.6:             ``Var(suffix, aval)``
    * older:            ``Var(count, suffix, aval)``

    The most likely form is selected from the constructor signature, and the
    actual call falls back across the alternatives on ``TypeError`` (caching
    the first that succeeds) so a signature we did not anticipate still works.
    """
    params = [p for p in inspect.signature(Var.__init__).parameters if p != 'self']
    _counter = itertools.count()

    def _modern(aval):
        return Var(aval)

    def _suffix_aval(aval):
        return Var('', aval)

    def _count_suffix_aval(aval):
        return Var(next(_counter), '', aval)

    if params[:2] == ['suffix', 'aval']:
        order = [_suffix_aval, _count_suffix_aval, _modern]
    elif 'count' in params and 'suffix' in params:
        order = [_count_suffix_aval, _suffix_aval, _modern]
    else:
        order = [_modern, _suffix_aval, _count_suffix_aval]

    state = {'fn': None}

    def factory(aval):
        if state['fn'] is not None:
            return state['fn'](aval)
        last_exc = None
        for fn in order:
            try:
                var = fn(aval)
            except TypeError as exc:
                last_exc = exc
                continue
            state['fn'] = fn
            return var
        raise last_exc

    return factory


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def ensure_jaxpr(obj):
    """Validate and normalize a ``Jaxpr``/``ClosedJaxpr`` input.

    Returns a tuple ``(jaxpr, consts, was_closed)`` where ``jaxpr`` is a bare
    ``Jaxpr``, ``consts`` is a list (empty for a bare ``Jaxpr``), and
    ``was_closed`` indicates whether the input was a ``ClosedJaxpr``.

    Raises
    ------
    IRValidationError
        If ``obj`` is neither a ``Jaxpr`` nor a ``ClosedJaxpr``.
    """
    if isinstance(obj, ClosedJaxpr):
        return obj.jaxpr, list(obj.consts), True
    if isinstance(obj, Jaxpr):
        return obj, [], False
    raise IRValidationError(
        f"Expected a Jaxpr or ClosedJaxpr, got {type(obj).__name__}."
    )


def check_all_vars(varseq, name):
    """Raise :class:`IRValidationError` unless every item in ``varseq`` is a ``Var``."""
    for v in varseq:
        if not isinstance(v, Var):
            raise IRValidationError(
                f"{name} must contain only Var instances; got {type(v).__name__}."
            )


def is_scalar_literal_value(var, value) -> bool:
    """True iff ``var`` is a scalar ``Literal`` numerically equal to ``value``."""
    if not isinstance(var, Literal):
        return False
    val = var.val
    try:
        if isinstance(val, (int, float, complex)):
            return val == value
        arr = np.asarray(val)
        return arr.shape == () and arr.item() == value
    except Exception:
        return False


def literal_with_dtype(value, aval) -> Literal:
    """Create a ``Literal`` whose value matches ``aval.dtype`` (shape ``()``)."""
    dtype = getattr(aval, 'dtype', None)
    if dtype is None:
        return Literal(value, aval)
    return Literal(np.asarray(value, dtype=dtype), aval)


# ---------------------------------------------------------------------------
# Constant-folding engine (shared by _ir_optim and _ir_tocode)
# ---------------------------------------------------------------------------

# Primitives that must never be eagerly executed during constant folding:
#  - broadcast(_in_dim): folding them materializes potentially large constants.
#  - randomness / IO / callbacks: executing them at trace time is incorrect.
#  - control flow: their semantics are not a plain ``bind`` of concrete values.
CONSTANT_FOLD_BLACKLIST = {
    'broadcast_in_dim', 'broadcast',
    'rng_uniform', 'rng_bit_generator', 'random_seed', 'random_bits',
    'random_fold_in', 'random_split', 'random_wrap', 'random_unwrap',
    'random_gamma_grad',
    'while', 'scan', 'cond',
    'debug_callback', 'io_callback', 'pure_callback',
    'debug_print',
}


def _eval_eqn(eqn, vals):
    """Evaluate a single equation whose inputs are all concrete."""
    if eqn.primitive.name == "closed_call":
        return partial_eval_jaxpr(
            eqn.params['call_jaxpr'].jaxpr,
            {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)},
        )
    return eqn.primitive.bind(*vals, **eqn.params)


def partial_eval_jaxpr(jaxpr, env):
    """Evaluate all-constant equations at trace time, inlining their results.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to partially evaluate.
    env : dict
        Maps ``Var`` -> concrete value for already-known constants.

    Returns
    -------
    Jaxpr
        A new ``Jaxpr`` with constant equations folded and now-unused invars
        dropped. Blacklisted / control-flow primitives are passed through
        untouched.
    """
    env = dict(env)
    new_eqns = []

    def read(var):
        if isinstance(var, Literal):
            return var.val
        return env.get(var, None)

    def read_or_self(var):
        out = read(var)
        if out is None:
            return var
        if isinstance(out, Var):
            return out
        if isinstance(out, Literal):
            return Literal(out.val, var.aval)
        return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(v) for v in eqn.invars]
        if eqn.primitive.name in CONSTANT_FOLD_BLACKLIST:
            new_eqns.append(eqn)
        elif vals and all(v is not None for v in vals):
            out = _eval_eqn(eqn, vals)
            if isinstance(out, Jaxpr):
                new_eqns.extend(out.eqns)
                out = out.outvars
            elif not isinstance(out, (tuple, list)):
                out = (out,)
            for var, val in zip(eqn.outvars, out):
                env[var] = val.val if isinstance(val, Literal) else val
        else:
            new_eqns.append(eqn)

    out_eqns = [e.replace(invars=tuple(read_or_self(v) for v in e.invars)) for e in new_eqns]

    invars_still_used = IdentitySet()
    for e in out_eqns:
        for v in e.invars:
            if not isinstance(v, Literal):
                invars_still_used.add(v)

    invars = tuple(v for v in jaxpr.invars if v in invars_still_used)
    outvars = tuple(read_or_self(v) for v in jaxpr.outvars)
    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars)
