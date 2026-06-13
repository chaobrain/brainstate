# Modified from: https://github.com/dlwh/jax_sourceror
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

from __future__ import annotations

import ast
import enum
import keyword
import threading
import warnings
from collections.abc import MutableMapping, MutableSet
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.sharding_impls import UNSPECIFIED

from brainstate._compatible_import import Literal, Var, Jaxpr
from brainstate.transform._ir_utils import IRError, UnsupportedPrimitiveError


__all__ = [
    'fn_to_python_code',
    'jaxpr_to_python_code',
    'register_prim_handler',
]


class IdentitySet(MutableSet):
    """Set that compares objects by identity.

    This is a set that compares objects by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable set, but it does not support the ``__hash__`` method and
    therefore cannot be used as a dictionary key or as an element of another set.
    """

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

    def __repr__(self):
        return f"IdentitySet({list(repr(x) for x in self._data.values())})"

    def __str__(self):
        return f"IdentitySet({list(str(x) for x in self._data.values())})"


class IdentityMap(MutableMapping):
    """Map that compares keys by identity.

    This is a map that compares keys by identity instead of equality. It is
    useful for storing objects that are not hashable or that should be compared
    by identity.

    This is a mutable mapping, but it does not support the ``__hash__`` method
    and therefore cannot be used as a dictionary key or as an element of another
    set.
    """

    def __init__(self, iterable=None):
        # Map id(key) -> (key, value). The original key object is retained so
        # that ``__iter__`` can yield keys (the Mapping contract), which the
        # mixin methods ``keys``/``values``/``items``/``__eq__`` and ``dict(m)``
        # all rely on. Storing only ``id(key) -> value`` made ``__iter__`` yield
        # values, silently breaking every one of those.
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, key):
        return id(key) in self._data

    def __getitem__(self, key):
        return self._data[id(key)][1]

    def __setitem__(self, key, value):
        self._data[id(key)] = (key, value)

    def __delitem__(self, key):
        del self._data[id(key)]

    def __iter__(self):
        return iter(key for key, _ in self._data.values())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"IdentityMap({dict((repr(k), repr(v)) for k, v in self._data.values())})"

    def __str__(self):
        return f"IdentityMap({dict((str(k), str(v)) for k, v in self._data.values())})"


@dataclass
class SourcerorState:
    """State for the auto-minimizer. Basically just in charge of naming variables."""
    _var_names: IdentityMap[Var, str] = field(default_factory=IdentityMap)
    _skolem_count: int = 0

    def name(self, var, ctx=ast.Load()) -> ast.Name:
        return ast.Name(id=self.str_name(var), ctx=ctx)

    def str_name(self, var: Var):
        # Names things in a way vaguely compatible with
        # JAX's naming scheme, which is 'a'-'z' followed
        # by 'aa'-'az' etc.
        if var in self._var_names:
            return self._var_names[var]
        else:
            cur_count = len(self._var_names)
            name = ""
            while cur_count >= 26:
                name += chr(ord('a') + cur_count % 26)
                cur_count //= 26

            name += chr(ord('a') + cur_count)

            name = name[::-1]

            # The base-26 scheme can land on Python (soft) keywords -- e.g.
            # index 213 -> 'if', 221 -> 'in', 2137 -> 'def' -- which would emit
            # invalid source such as ``if = ie + 1.0``. Suffix an underscore to
            # disambiguate. This is collision-safe: no generated name contains
            # '_', so ``name + '_'`` cannot clash with another generated name.
            # The adjusted value is what gets cached, so the fast-path above
            # stays stable.
            if keyword.iskeyword(name) or keyword.issoftkeyword(name):
                name = name + '_'

            self._var_names[var] = name

            return name

    def skolem(self, prefix: str):
        self._skolem_count += 1
        return f"{prefix}_{self._skolem_count}"


class _ThreadLocalImports(threading.local):
    """Thread-local accumulator for the ``import`` lines a generation needs.

    ``prefix_imports`` used to be a single module-global ``set``. Because both
    public entry points clear it on enter/exit and handlers mutate it during
    generation, concurrent code generation in two threads could corrupt each
    other's output (one thread clearing the set mid-generation in another,
    dropping a required import and producing source with a ``NameError``).

    Subclassing ``threading.local`` (matching the codebase's established
    ``ThreadLocalStack`` idiom) gives each thread its own backing ``set`` while
    keeping the set-like API (``add``/``clear``/``in``/``len``/iteration) that
    the rest of the module -- and existing callers/tests -- rely on.
    """

    def __init__(self):
        # ``threading.local.__init__`` runs once per thread the first time the
        # instance is touched from that thread, giving each thread a fresh set.
        self._data = set()

    def add(self, value):
        self._data.add(value)

    def discard(self, value):
        self._data.discard(value)

    def clear(self):
        self._data.clear()

    def __contains__(self, value):
        return value in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


prefix_imports = _ThreadLocalImports()


@contextmanager
def catch_imports():
    try:
        prefix_imports.clear()
        yield
    finally:
        prefix_imports.clear()


def fn_to_python_code(fn: Callable, *args, **kwargs) -> str:
    """
    Given a function which is defined by jax primitives and the function arguments,
    return the Python code that would be generated by JAX for that function.

    Parameters
    ----------
    fn : Callable
        The function to generate code for
    args
        The positional arguments to the function
    kwargs
        The keyword arguments to the function

    Returns
    -------
    The Python code that would be generated by JAX for that function
    """
    closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    jaxpr = constant_fold_jaxpr(closed_jaxpr.jaxpr)
    state = SourcerorState()
    try:
        name = fn.__name__
    except AttributeError:
        name = "generated_function"
    # ``<lambda>`` and other non-identifier names cannot be used as a ``def``
    # name in the generated source.
    if not isinstance(name, str) or not name.isidentifier():
        name = "generated_function"
    with catch_imports():
        node = jaxpr_to_py_ast(state, jaxpr, fn_name=name)
        node = _maybe_wrap_fn_for_leaves(node, fn, len(args) + len(kwargs))
        ast.fix_missing_locations(node)
        source = ast.unparse(node)
        if len(prefix_imports):
            source = "\n".join(prefix_imports) + "\n\n" + source
    return source


def jaxpr_to_python_code(jaxpr: Jaxpr,
                         fn_name: str = "generated_function") -> str:
    """
    Given a JAX jaxpr, return the Python code that would be generated by JAX for that jaxpr.

    Parameters
    ----------
    jaxpr : Jaxpr
        The jaxpr to generate code.
    fn_name : str
        The name of the function to generate code.

    Returns
    -------
    The Python code that would be generated by JAX for that jaxpr
    """
    jaxpr = constant_fold_jaxpr(jaxpr)
    state = SourcerorState()
    with catch_imports():
        node = jaxpr_to_py_ast(state, jaxpr, fn_name=fn_name)
        ast.fix_missing_locations(node)
        source = ast.unparse(node)
        if len(prefix_imports):
            source = "\n".join(prefix_imports) + "\n\n" + source
    return source


def register_prim_handler(prim_name: str, handler: Callable) -> None:
    """
    Register a handler for a primitive for automin

    Parameters
    ----------
    prim_name : str
        Name of the primitive.
    handler : Callable
        Handler for the primitive.
    """
    if prim_name in prim_to_python:
        warnings.warn(f"Overwriting handler for primitive {prim_name}")
    prim_to_python[prim_name] = handler


def register_prim_as(prim_name):
    """
    Decorator to register a handler for a primitive.

    Parameters
    ----------
    prim_name
        Name of the primitive.
    """

    def decorator(fn):
        register_prim_handler(prim_name, fn)
        return fn

    return decorator


def _assign_stmt(call_expr: Callable):
    """
    Create a handler for a primitive that is a simple assignment.

    Parameters
    ----------
    call_expr : Callable
        Callable that builds the assignment's right-hand-side expression.
    """

    def binop_fn(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(
            outvars,
            call_expr(
                *invars,
                **{k: _astify_value(v) for k, v in eqn.params.items()}
            )
        )

    return binop_fn


def _binop_fn(op: ast.operator):
    # Binary arithmetic primitives ignore their params (e.g. ``out_dtype`` in
    # recent JAX) -- the result is simply ``x <op> y``.
    def fn(state, eqn):
        x, y = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(outvars, ast.BinOp(left=x, op=op, right=y))

    return fn


def _cmpop_fn(op: ast.cmpop):
    def fn(state, eqn):
        x, y = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(outvars, ast.Compare(left=x, ops=[op], comparators=[y]))

    return fn


def normal_fn(fn_name):
    """
    Create a handler for a normal function call.

    Parameters
    ----------
    fn_name
        Name of the function to call.
    """
    return _assign_stmt(
        lambda *args, **kwargs: ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=list(args),
            keywords=[ast.keyword(arg=k, value=v) for k, v in kwargs.items()]
        )
    )


def _reduce_fn(fn_name: str):
    def reduce_fn_inner(state: SourcerorState, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        # Only the reduction axes are forwarded; other params (e.g.
        # ``out_sharding`` in recent JAX) are not valid numpy kwargs.
        keywords = []
        if 'axes' in eqn.params and eqn.params['axes'] is not None:
            keywords.append(ast.keyword(arg='axis', value=_astify_value(tuple(eqn.params['axes']))))
        call_op = ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=keywords,
        )
        return ast.Assign(outvars, call_op)

    return reduce_fn_inner


prim_to_python = dict()

register_prim_handler('add', _binop_fn(ast.Add()))
register_prim_handler('sub', _binop_fn(ast.Sub()))
register_prim_handler('mul', _binop_fn(ast.Mult()))
register_prim_handler('div', _binop_fn(ast.Div()))
register_prim_handler('neg', normal_fn('jax.lax.neg'))
register_prim_handler('lt', _cmpop_fn(ast.Lt()))
register_prim_handler('gt', _cmpop_fn(ast.Gt()))
register_prim_handler('le', _cmpop_fn(ast.LtE()))
register_prim_handler('ge', _cmpop_fn(ast.GtE()))
register_prim_handler('eq', _cmpop_fn(ast.Eq()))
register_prim_handler('ne', _cmpop_fn(ast.NotEq()))
register_prim_handler('min', normal_fn('jax.lax.min'))
register_prim_handler('max', normal_fn('jax.lax.max'))
register_prim_handler('select_n', normal_fn('jax.lax.select_n'))
register_prim_handler('squeeze', normal_fn('jax.lax.squeeze'))
register_prim_handler('broadcast', normal_fn('jax.lax.broadcast'))
register_prim_handler('reduce_sum', _reduce_fn('jax.numpy.sum'))
register_prim_handler('transpose', normal_fn('jax.lax.transpose'))


def _call_noparams(fn_name: str):
    """Handler that emits ``fn_name(*invars)``, ignoring equation params.

    Suitable for elementwise primitives whose params (e.g. ``accuracy`` on
    transcendental ops in recent JAX) are not valid call arguments.
    """
    def fn(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        return ast.Assign(
            outvars,
            ast.Call(func=ast.Name(id=fn_name, ctx=ast.Load()), args=invars, keywords=[]),
        )

    return fn


# --- Elementwise unary / binary primitives (params ignored) ---
for _name in (
    'sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'asin', 'acos', 'atan',
    'exp', 'exp2', 'log', 'log1p', 'expm1', 'sqrt', 'rsqrt', 'cbrt',
    'abs', 'sign', 'floor', 'ceil', 'erf', 'erfc', 'erf_inv',
    'is_finite', 'logistic', 'square', 'reciprocal', 'not',
    'pow', 'rem', 'atan2', 'nextafter', 'and', 'or', 'xor',
    'shift_left', 'shift_right_logical', 'shift_right_arithmetic',
):
    if _name not in prim_to_python:
        register_prim_handler(_name, _call_noparams(f'jax.lax.{_name}'))


def _round_handler(state, eqn):
    """Handle the ``round`` primitive, forwarding its ``rounding_method``.

    ``jax.lax.round`` defaults to ``RoundingMethod.AWAY_FROM_ZERO`` while
    ``jnp.round``/``jnp.around`` lower with ``RoundingMethod.TO_NEAREST_EVEN``
    (banker's rounding). Dropping the param (as the generic ``_call_noparams``
    loop did) silently changes results on every half-integer. We emit the
    method as a fully-qualified, namespace-resolvable
    ``jax.lax.RoundingMethod(<int>)`` so it works in the generated namespace
    (which only has jax/jnp/np) and is robust across JAX versions (reconstruct
    by value rather than by member name).
    """
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    rm = eqn.params['rounding_method']  # a jax.lax.RoundingMethod enum
    method_ast = ast.Call(
        func=ast.Name(id='jax.lax.RoundingMethod', ctx=ast.Load()),
        args=[ast.Constant(value=int(rm))],
        keywords=[],
    )
    return ast.Assign(
        outvars,
        ast.Call(
            func=ast.Name(id='jax.lax.round', ctx=ast.Load()),
            args=invars + [method_ast],
            keywords=[],
        ),
    )


register_prim_handler('round', _round_handler)


def _integer_pow_handler(state, eqn):
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    return ast.Assign(
        outvars,
        ast.Call(
            func=ast.Name(id='jax.lax.integer_pow', ctx=ast.Load()),
            args=invars + [_astify_value(eqn.params['y'])],
            keywords=[],
        ),
    )


register_prim_handler('integer_pow', _integer_pow_handler)

# --- Reductions (axes -> axis) ---
register_prim_handler('reduce_max', _reduce_fn('jax.numpy.max'))
register_prim_handler('reduce_min', _reduce_fn('jax.numpy.min'))
register_prim_handler('reduce_prod', _reduce_fn('jax.numpy.prod'))
register_prim_handler('reduce_and', _reduce_fn('jax.numpy.all'))
register_prim_handler('reduce_or', _reduce_fn('jax.numpy.any'))


def _argreduce_fn(fn_name: str):
    def handler(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        axis = eqn.params['axes'][0]
        return ast.Assign(
            outvars,
            ast.Call(
                func=ast.Name(id=fn_name, ctx=ast.Load()),
                args=invars,
                keywords=[ast.keyword(arg='axis', value=_astify_value(axis))],
            ),
        )

    return handler


register_prim_handler('argmax', _argreduce_fn('jax.numpy.argmax'))
register_prim_handler('argmin', _argreduce_fn('jax.numpy.argmin'))


def _concatenate_handler(state, eqn):
    elts = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    return ast.Assign(
        outvars,
        ast.Call(
            func=ast.Name(id='jax.numpy.concatenate', ctx=ast.Load()),
            args=[ast.List(elts=elts, ctx=ast.Load())],
            keywords=[ast.keyword(arg='axis', value=_astify_value(eqn.params['dimension']))],
        ),
    )


register_prim_handler('concatenate', _concatenate_handler)


def _expand_dims_handler(state, eqn):
    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)
    return ast.Assign(
        outvars,
        ast.Call(
            func=ast.Name(id='jax.lax.expand_dims', ctx=ast.Load()),
            args=invars + [_astify_value(tuple(eqn.params['dimensions']))],
            keywords=[],
        ),
    )


register_prim_handler('expand_dims', _expand_dims_handler)


def _cumulative_fn(fn_name: str):
    def handler(state, eqn):
        invars = [_astify_atom(state, v) for v in eqn.invars]
        outvars = _astify_outvars(state, eqn.outvars)
        keywords = []
        if 'axis' in eqn.params:
            keywords.append(ast.keyword(arg='axis', value=_astify_value(eqn.params['axis'])))
        if 'reverse' in eqn.params:
            keywords.append(ast.keyword(arg='reverse', value=_astify_value(eqn.params['reverse'])))
        return ast.Assign(
            outvars,
            ast.Call(func=ast.Name(id=fn_name, ctx=ast.Load()), args=invars, keywords=keywords),
        )

    return handler


register_prim_handler('cumsum', _cumulative_fn('jax.lax.cumsum'))
register_prim_handler('cumprod', _cumulative_fn('jax.lax.cumprod'))
register_prim_handler('cummax', _cumulative_fn('jax.lax.cummax'))
register_prim_handler('cummin', _cumulative_fn('jax.lax.cummin'))


def _maybe_wrap_fn_for_leaves(node, f, num_args):
    if len(node.args.args) == num_args:
        return node

    # Reuse the inner def's already-sanitized name (``node.name``) rather than
    # re-reading the raw ``f.__name__``: a lambda would otherwise yield the
    # invalid ``def <lambda>(*args, **kwargs):``, and callables lacking
    # ``__name__`` (e.g. functools.partial) would raise AttributeError. The
    # caller (fn_to_python_code) already sanitized this name to a valid
    # identifier (falling back to 'generated_function').
    wrapped_node = ast.FunctionDef(
        name=node.name,
        args=ast.arguments(
            args=[],
            vararg=ast.arg(arg="args", annotation=None),
            kwarg=ast.arg(arg="kwargs", annotation=None),
            kwonlyargs=[], kw_defaults=[], defaults=[],
            posonlyargs=[]
        ),
        body=[
            node,
            ast.Return(
                ast.Call(
                    func=ast.Name(id=node.name, ctx=ast.Load()),
                    args=[
                        ast.Starred(
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.Attribute(value=ast.Name(id="jax", ctx=ast.Load()),
                                                        attr="tree_util",
                                                        ctx=ast.Load()),
                                    attr="tree_leaves",
                                    ctx=ast.Load()),
                                args=[ast.Tuple(elts=[ast.Name(id="args", ctx=ast.Load()),
                                                      ast.Name(id="kwargs", ctx=ast.Load())],
                                                ctx=ast.Load())],
                                keywords=[]
                            )
                        )
                    ],
                    keywords=[]
                )
            ),
        ],
        decorator_list=[]
    )

    return wrapped_node


def jaxpr_to_py_ast(state: SourcerorState,
                    jaxpr: Jaxpr,
                    fn_name: str = "function"):
    # Generate argument declarations
    ast_args = [ast.arg(arg=state.str_name(var), annotation=None)
                for var in jaxpr.invars]
    ast_args = ast.arguments(args=ast_args,
                             vararg=None,
                             kwonlyargs=[],
                             kw_defaults=[],
                             kwarg=None,
                             defaults=[],
                             posonlyargs=[])

    stmts = []

    # Generate body of the function
    for eqn in jaxpr.eqns:
        prim = str(eqn.primitive)
        handler = prim_to_python.get(prim)
        if handler is None:
            raise UnsupportedPrimitiveError(
                f"No code-generation handler for primitive '{prim}'. "
                f"Register one with register_prim_handler('{prim}', handler)."
            )
        eqn_stmts = handler(state, eqn)

        if isinstance(eqn_stmts, list):
            stmts.extend(eqn_stmts)
        else:
            stmts.append(eqn_stmts)

    # Generate return statement
    if len(jaxpr.outvars) == 1:
        returns = state.name(jaxpr.outvars[0])
    else:
        returns = ast.Tuple(elts=[state.name(var) for var in jaxpr.outvars], ctx=ast.Load())
    stmts.append(ast.Return(value=returns))

    return ast.FunctionDef(name=fn_name, args=ast_args, body=stmts, decorator_list=[])


def constant_fold_jaxpr(jaxpr: Jaxpr):
    """
    Given a jaxpr, return a new jaxpr with all constant folding done.
    """
    return partial_eval_jaxpr(jaxpr, {})


def partial_eval_jaxpr(jaxpr, env):
    env = env.copy()
    new_eqns = []

    def read(var):
        if isinstance(var, Literal):
            return var.val
        else:
            return env.get(var, None)

    def read_or_self(var):
        out = read(var)
        if out is None:
            return var
        elif isinstance(out, Var):
            return out
        elif isinstance(out, Literal):
            return Literal(out.val, var.aval)
        else:
            if isinstance(out, Jaxpr):
                raise IRError(
                    "Unexpected nested Jaxpr produced during constant folding "
                    "while generating Python code."
                )
            return Literal(out, var.aval)

    for eqn in jaxpr.eqns:
        vals = [read(var) for var in eqn.invars]
        if eqn.primitive.name in constant_fold_blacklist:
            new_eqns.append(eqn)
        elif all(val is not None for val in vals):
            # go ahead and eval it
            out = _eval_eqn(eqn, vals)

            # two options: either it's a jaxpr result (partial eval) or it's a value or a list of values
            if isinstance(out, Jaxpr):
                # we need to inline this
                new_eqns.extend(out.eqns)
                out = out.outvars
            elif not isinstance(out, tuple) and not isinstance(out, list):
                out = (out,)

            for var, val in zip(eqn.outvars, out):
                if isinstance(val, Jaxpr):
                    raise IRError(
                        "Unexpected nested Jaxpr produced during constant folding "
                        "while generating Python code."
                    )
                if isinstance(val, Literal):
                    env[var] = val.val
                else:
                    env[var] = val
        else:
            new_eqns.append(eqn)

    # now that we've evaled everything, inline all the constants
    out_eqns = []
    for eqn in new_eqns:
        eqn = eqn.replace(invars=tuple(read_or_self(var) for var in eqn.invars))
        out_eqns.append(eqn)

    # sub in any constants for outvars
    outvars = tuple(read_or_self(var) for var in jaxpr.outvars)

    invars_still_used = IdentitySet()
    for eqn in out_eqns:
        for var in eqn.invars:
            invars_still_used.add(var)
    # Pass-through invars (forwarded directly to an output without appearing in
    # any equation, e.g. the identity function) must be retained, otherwise the
    # generated function loses parameters it still needs to return.
    for var in outvars:
        invars_still_used.add(var)

    invars = tuple(var for var in jaxpr.invars if var in invars_still_used)

    return jaxpr.replace(eqns=out_eqns, outvars=outvars, invars=invars, debug_info=None)


def _eval_eqn(eqn, vals) -> Union[Jaxpr, tuple, list, jax.Array]:
    if eqn.primitive.name == "closed_call":
        out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
                                 {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    else:
        out = eqn.primitive.bind(*vals, **eqn.params)
    return out


@register_prim_as('dot_general')
def _astify_dot_general(state, eqn):
    x, y = eqn.invars
    d = eqn.params['dimension_numbers']
    precision = eqn.params['precision']
    preferred_element_type = eqn.params['preferred_element_type']

    has_dtype = preferred_element_type is None or x.aval.dtype == y.aval.dtype == preferred_element_type

    # recognize simple matmul case
    if d == (((1,), (0,)), ((), ())) and precision == None:
        invars = [_astify_atom(state, x), _astify_atom(state, y)]
        outvars = _astify_outvars(state, eqn.outvars)
        out = ast.Assign(targets=outvars, value=ast.Call(
            func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='matmul', ctx=ast.Load()),
            args=invars,
            keywords=[]))
        if not has_dtype:
            out = ast.Assign(targets=outvars,
                             value=ast.Call(func=ast.Attribute(value=out.value, attr='astype', ctx=ast.Load()),
                                            args=[_astify_value(preferred_element_type)], keywords=[]))

        return out

    # TODO: convert to einsum?

    invars = [_astify_atom(state, x),
              _astify_atom(state, y),
              _astify_value(d),
              _astify_value(precision),
              _astify_value(preferred_element_type)]
    outvars = _astify_outvars(state, eqn.outvars)
    return ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Attribute(value=ast.Name(id='jax.lax', ctx=ast.Load()), attr='dot_general', ctx=ast.Load()),
            args=invars,
            keywords=[]
        )
    )


@register_prim_as('dynamic_slice')
def _sourcify_dynamic_slice(state, eqn):
    sliced = eqn.invars[0]
    invars = ast.Tuple(elts=[_astify_atom(state, var) for var in eqn.invars[1:]], ctx=ast.Load())
    outvars = _astify_outvars(state, eqn.outvars)
    params = [ast.keyword(arg=k, value=_astify_value(v)) for k, v in eqn.params.items()]
    return ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='jax.lax', ctx=ast.Load()),
                attr='dynamic_slice',
                ctx=ast.Load()
            ),
            args=[_astify_atom(state, sliced), invars],
            keywords=params
        )
    )


@register_prim_as('slice')
def _sourcify_slice(state, eqn):
    sliced = eqn.invars[0]
    # invars = ast.Tuple(elts=[_astify_atom(state, var) for var in eqn.invars[1:]], ctx=ast.Load())
    outvars = _astify_outvars(state, eqn.outvars)
    start_indices = eqn.params['start_indices']
    limit_indices = eqn.params['limit_indices']
    strides = eqn.params['strides']
    if strides is None:
        strides = (None,) * len(start_indices)
    indices = [_astify_value(slice(s, e, stride))
               for s, e, stride in zip(start_indices, limit_indices, strides)]
    # params = [ast.keyword(arg=k, value=_astify_value(v)) for k, v in eqn.params.items()]
    return ast.Assign(
        targets=outvars,
        value=ast.Subscript(
            value=_astify_atom(state, sliced),
            slice=ast.Tuple(elts=indices, ctx=ast.Load()),
            ctx=ast.Load()
        )
    )


@register_prim_as('dynamic_update_slice')
def _sourcify_dynamic_update_slice(state, eqn):
    sliced = eqn.invars[0]
    # the first two arguments are the sliced array and the update array
    # the remaining are start indices and should be packaged into a tuple
    target = _astify_atom(state, eqn.invars[0])
    update = _astify_atom(state, eqn.invars[1])
    # ``jax.lax.dynamic_update_slice`` requires ``start_indices`` to be a
    # sequence, so always emit a tuple (even for a single index) — matching
    # ``_sourcify_dynamic_slice``.
    start_indices = ast.Tuple(
        elts=[_astify_atom(state, var) for var in eqn.invars[2:]],
        ctx=ast.Load(),
    )
    outvars = _astify_outvars(state, eqn.outvars)

    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jax.lax', ctx=ast.Load()),
            attr='dynamic_update_slice',
            ctx=ast.Load()
        ),
        args=[target, update, start_indices],
        keywords=[]
    ))


@register_prim_as('convert_element_type')
def _astify_convert_element_type(state, eqn):
    # now we use ast
    outvars = _astify_outvars(state, eqn.outvars)
    assert len(eqn.invars) == 1
    invar = _astify_atom(state, eqn.invars[0])
    dtype = _astify_value(eqn.params['new_dtype'])
    return ast.Assign(targets=outvars, value=ast.Call(
        func=ast.Attribute(
            value=invar,
            attr='astype',
            ctx=ast.Load()
        ),
        args=[dtype],
        keywords=[]
    ))


def is_array(arr):
    if isinstance(arr, (np.ndarray, np.generic, jax.Array)):
        return True
    # Some JAX versions box jaxpr literals in wrapper types (e.g.
    # ``jax._src.literals.TypedNdArray``) that are array-like - they carry a
    # ``dtype`` and ``shape`` and convert via ``numpy`` - but are *not*
    # subclasses of ``jax.Array``. Recognise them by duck-typing so their
    # values are emitted as real arrays instead of falling through to the
    # "unknown value" path (which produced syntactically invalid code).
    return (
        hasattr(arr, 'dtype')
        and hasattr(arr, 'shape')
        and not isinstance(arr, np.dtype)
    )


def _coerce_to_numpy(value):
    """Realise an array-like literal to a concrete ``numpy`` array.

    Native ``numpy`` scalars/arrays are returned unchanged so the existing
    ``np.int64`` / 0-d special cases in :func:`_astify_array` keep working.
    Everything else (``jax.Array`` and boxed literal wrappers) is converted
    via ``numpy``, with attribute-based fallbacks for wrappers that do not
    implement the array protocol directly.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return value
    # Boxed jaxpr literals (e.g. ``TypedNdArray``) expose the underlying
    # ndarray via ``.val``; prefer it so the exact dtype is preserved (the
    # wrapper's ``__array__`` is documented to misbehave on NumPy < 2.3).
    inner = getattr(value, 'val', None)
    if isinstance(inner, (np.ndarray, np.generic)):
        return inner
    try:
        return np.asarray(value)
    except Exception:
        for attr in ('_value', 'array', 'value'):
            inner = getattr(value, attr, None)
            if inner is not None and inner is not value:
                try:
                    return np.asarray(inner)
                except Exception:
                    continue
        raise


def _astify_array(value):
    assert is_array(value)
    value = _coerce_to_numpy(value)
    if isinstance(value, np.int64):
        return ast.Constant(value=int(value))

    if value.ndim == 0 and value.dtype in (jnp.float32, jnp.int32, jnp.bool_, jnp.int64):
        return ast.Constant(value=value.item())

    if value.ndim == 0:
        # ``_astify_value(dtype)`` returns a ``jax.numpy.dtype('...')`` instance
        # for dtypes outside the named set (uint8/16/32/64, int8/16, complex*),
        # and a DType instance is NOT callable -- ``dtype('uint16')(3)`` raises
        # ``TypeError`` at runtime. Emit ``jax.numpy.array(value, dtype=...)``
        # instead (mirroring the multi-element branch below); this preserves
        # ndim==0 and the exact dtype and works for every dtype.
        dtype_value = _astify_value(value.dtype)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                attr='array',
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=value.item())],
            keywords=[ast.keyword(arg='dtype', value=dtype_value)],
        )

    values = value.tolist()

    def rec_astify_list(values):
        if isinstance(values, list):
            return ast.List(elts=[rec_astify_list(val) for val in values], ctx=ast.Load())
        else:
            return ast.Constant(value=values)

    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='jax.numpy', ctx=ast.Load()),
            attr='array',
            ctx=ast.Load()
        ),
        args=[rec_astify_list(values)],
        keywords=[ast.keyword(arg='dtype',
                              value=_astify_value(value.dtype))]
    )


def _astify_atom(state: SourcerorState, var: Union[Literal, Var]):
    if isinstance(var, Literal):
        return _astify_value(var.val)
    elif isinstance(var, Var):
        return state.name(var)
    else:
        raise NotImplementedError()


def _astify_value(value):
    assert not isinstance(value, (Literal, Var))

    if is_array(value):
        return _astify_array(value)
    elif isinstance(value, bool):
        return ast.Constant(value=bool(value))
    elif isinstance(value, int):
        # Coerce subclasses (e.g. JAX-boxed ``TypedInt`` literals) to a plain
        # ``int`` so ``ast.unparse`` emits a literal instead of the object repr.
        return ast.Constant(value=int(value))
    elif isinstance(value, float):
        # Coerce subclasses (e.g. JAX-boxed ``TypedFloat`` literals) to a plain
        # ``float`` for the same reason.
        return ast.Constant(value=float(value))
    elif isinstance(value, (str, type(None))):
        return ast.Constant(value=value)
    elif isinstance(value, (tuple, list)):
        return ast.Tuple(elts=[_astify_value(v) for v in value], ctx=ast.Load())
    elif isinstance(value, jnp.dtype):
        # return ast.Call(func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()), attr='dtype', ctx=ast.Load()), args=[ast.Constant(value=str(value))], keywords=[])
        if value.name in ('float32', 'float64', 'int32', 'int64', 'bfloat16', 'float16'):
            # return ast.Constant(value=getattr(jnp, value.name))
            return ast.Attribute(
                value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                attr=value.name,
                ctx=ast.Load()
            )
        elif value.name == 'bool':
            return ast.Attribute(
                value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                attr='bool_',
                ctx=ast.Load()
            )
        else:
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                                   attr='dtype',
                                   ctx=ast.Load()),
                args=[ast.Constant(value=str(value))],
                keywords=[]
            )
    elif value is UNSPECIFIED:
        prefix_imports.add('from jax._src.sharding_impls import UNSPECIFIED')
        return ast.Name(id='UNSPECIFIED', ctx=ast.Load())
    elif isinstance(value, enum.Enum):
        # Emit a fully-qualified, importable reference and register the import,
        # mirroring the UNSPECIFIED branch. A bare ``ClassName.MEMBER`` has no
        # binding in the generated namespace (only jax/jnp/np) and raises
        # ``NameError``. Build ``module.qualname.MEMBER`` as a chained
        # Attribute, which also handles nested enums whose ``__qualname__``
        # contains dots (``from m import Outer.Inner`` is not valid syntax).
        cls = value.__class__
        module = cls.__module__
        qualname = cls.__qualname__  # may contain dots for nested enums
        prefix_imports.add(f'import {module}')
        node = ast.Name(id=module.split('.')[0], ctx=ast.Load())
        for part in module.split('.')[1:] + qualname.split('.') + [value.name]:
            node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
        return node

    else:
        warnings.warn(f"Unknown value type {type(value)}")
        return ast.parse(repr(value)).body[0]


def _astify_outvars(state, outvars):
    out = [state.name(v, ctx=ast.Store()) for v in outvars]
    if len(out) == 1:
        return out
    else:
        return [ast.Tuple(elts=out, ctx=ast.Store())]


def maybe_tuple_vars(vars):
    if len(vars) == 1:
        return vars[0]
    else:
        return ast.Tuple(elts=vars, ctx=ast.Load())


def maybe_untuple_vars(var, is_tuple):
    if is_tuple:
        return ast.Starred(value=var, ctx=ast.Load())
    else:
        return var


@register_prim_as('scan')
def _astify_scan(state, eqn):
    assert eqn.primitive.name == 'scan'

    # the args to scan are [constants, carry, xs]
    # constants aren't exposed in the Python API, so we need to handle them specially (we use a lambda)
    num_consts = eqn.params['num_consts']
    num_carry = eqn.params['num_carry']

    # TODO: bring back map
    # if num_carry == 0:
    # this is a map
    # return _astify_map(eqn)

    constant_args = eqn.invars[:num_consts]
    carries = eqn.invars[num_consts:num_consts + num_carry]
    xs = eqn.invars[num_consts + num_carry:]

    jaxpr = eqn.params['jaxpr']

    if num_consts != 0:
        # we want to construct an environment where we partial eval the function using the constants as the env
        env = dict(zip(jaxpr.jaxpr.invars, constant_args))
        jaxpr = partial_eval_jaxpr(jaxpr.jaxpr, env)
    else:
        jaxpr = constant_fold_jaxpr(jaxpr.jaxpr)

    fn_name = state.skolem('fn')
    fn_ast = jaxpr_to_py_ast(state, jaxpr, fn_name)

    length = _astify_value(eqn.params['length'])
    unroll = _astify_value(eqn.params['unroll'])
    reverse = _astify_value(eqn.params['reverse'])

    stmts = []

    if num_carry != 1 or len(jaxpr.invars) != 2:
        # what we want is something like:
        # fn_name = lambda carry, xs: fn_name(*carry, *xs)
        # jax.lax.scan(fn_name, (carries...), (xs...))

        # The wrapper's parameter/result names must not collide with the
        # variable namer's output. ``str_name`` emits base-26 names (``a``..``z``,
        # ``ba``..) which CAN land on ``carry``/``x``/``ys`` (e.g. ``x`` is index
        # 23, ``ys`` index 642): a hardcoded ``carry``/``x`` lambda arg then
        # shadows a same-named carry var, and a hardcoded ``final_carry``/``ys``
        # output clobbers a same-named outer var. ``skolem`` names contain an
        # ``_<count>`` suffix that no base-26 name has, so they are collision-free.
        carry_name = state.skolem('carry')
        x_name = state.skolem('x')

        modified_signature = ast.arguments(
            args=[ast.arg(arg=carry_name), ast.arg(arg=x_name)],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[]
        )

        initial_assign = ast.Assign(
            targets=[ast.Tuple(elts=[ast.Name(a.arg) for a in fn_ast.args.args],
                               ctx=ast.Store())],
            value=ast.Tuple(
                elts=[maybe_untuple_vars(ast.Name(id=carry_name, ctx=ast.Load()), num_carry != 1),
                      maybe_untuple_vars(ast.Name(id=x_name, ctx=ast.Load()), len(xs) != 1)]
            )
        )

        fn_return = fn_ast.body[-1]
        assert isinstance(fn_return, ast.Return)

        fn_return_value = fn_return.value

        if isinstance(fn_return_value, ast.Tuple):
            fn_return_value = fn_return_value.elts
            ret_carries = maybe_tuple_vars(fn_return_value[:num_carry])
            ret_ys = maybe_tuple_vars(fn_return_value[num_carry:])
        elif num_carry == 0:
            ret_carries = _astify_value(())
            ret_ys = fn_return_value
        else:
            ret_carries = fn_return_value
            ret_ys = _astify_value(())

        scan_return = ast.Return(
            value=ast.Tuple(elts=[ret_carries, ret_ys], ctx=ast.Load())
        )

        new_body = [initial_assign] + list(fn_ast.body[:-1]) + [scan_return]

        fn_ast = ast.FunctionDef(
            name=fn_name,
            args=modified_signature,
            body=new_body,
            decorator_list=[]
        )

        stmts.append(fn_ast)

        # Collision-free temporaries for the scan's (final_carry, ys) result
        # (see the note above on ``carry``/``x``).
        final_carry_name = state.skolem('final_carry')
        ys_name = state.skolem('ys')

        scan_call = ast.Assign(
            # targets=_astify_outvars(eqn.outvars),
            targets=[
                ast.Tuple(
                    elts=[ast.Name(id=final_carry_name, ctx=ast.Store()),
                          ast.Name(id=ys_name, ctx=ast.Store())],
                    ctx=ast.Store()
                )
            ],
            value=ast.Call(
                func=ast.Name(id='jax.lax.scan', ctx=ast.Load()),
                args=[ast.Name(id=fn_name, ctx=ast.Load()),
                      maybe_tuple_vars([_astify_atom(state, v) for v in carries]),
                      maybe_tuple_vars([_astify_atom(state, v) for v in xs])],
                keywords=[ast.keyword(arg='length', value=length),
                          ast.keyword(arg='unroll', value=unroll),
                          ast.keyword(arg='reverse', value=reverse)]
            )
        )
        stmts.append(scan_call)

        if num_carry > 0:
            assign_carry = ast.Assign(
                targets=_astify_outvars(state, eqn.outvars[:num_carry]),
                value=ast.Name(id=final_carry_name, ctx=ast.Load())
            )

            stmts.append(assign_carry)

        if num_carry < len(eqn.outvars):
            assign_ys = ast.Assign(
                targets=_astify_outvars(state, eqn.outvars[num_carry:]),
                value=ast.Name(id=ys_name, ctx=ast.Load())
            )

            stmts.append(assign_ys)
    else:
        stmts.append(fn_ast)

        scan_call = ast.Assign(
            targets=_astify_outvars(state, eqn.outvars),
            value=ast.Call(
                func=ast.Name(id='jax.lax.scan', ctx=ast.Load()),
                args=[ast.Name(id=fn_name, ctx=ast.Load())] + [_astify_atom(state, v) for v in eqn.invars],
                keywords=[ast.keyword(arg='length', value=length),
                          ast.keyword(arg='unroll', value=unroll),
                          ast.keyword(arg='reverse', value=reverse)]
            )
        )

        stmts.append(scan_call)

    return stmts


def _astify_map(state, eqn):
    assert eqn.primitive.name == 'scan'
    assert eqn.params['num_carry'] == 0

    jaxpr = eqn.params['jaxpr']
    jaxpr = constant_fold_jaxpr(jaxpr.jaxpr)

    fn_name = state.skolem('fn')
    fn_ast = jaxpr_to_py_ast(state, jaxpr, fn_name)

    # map is a bit funny, because the jaxpr takes K args, but the jax.lax.map function takes a single tuple arg
    # so we need to use a lambda to redirect the call
    lam = ast.parse(f"lambda args: {fn_name}(*args)").body[0]

    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=ast.Name(id='jax.lax.map', ctx=ast.Load()),
            args=[lam,
                  ast.Tuple(elts=[_astify_atom(state, v) for v in eqn.invars],
                            ctx=ast.Load())],
            keywords=[]
        )
    )

    return [fn_ast, assign]


@register_prim_as('closed_call')
def _astify_closed_call(state, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    raw_jaxpr = eqn.params['call_jaxpr'].jaxpr
    literal_args = {k: v.val
                    for k, v in zip(raw_jaxpr.invars, eqn.invars)
                    if isinstance(v, Literal)}
    call_japr = partial_eval_jaxpr(raw_jaxpr, literal_args)
    fn_name = state.skolem('fn')

    fn_ast = jaxpr_to_py_ast(state, call_japr, fn_name)

    invars = [_astify_atom(state, v)
              for v in eqn.invars
              if not isinstance(v, Literal)]
    outvars = _astify_outvars(state, eqn.outvars)

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=fn_name, ctx=ast.Load()),
            args=invars,
            keywords=[]
        )
    )

    return [fn_ast, assign]


def _astify_pjit(state, eqn):
    # The jit/pjit primitive carries a nested jaxpr plus sharding/donation
    # metadata whose param names vary across JAX versions. Rather than
    # reconstructing the sharding arguments (fragile and version-specific), emit
    # a nested function wrapped in a plain ``jax.jit`` and call it. This is
    # numerically equivalent for the common (unsharded) case.
    jaxpr = eqn.params.get('jaxpr')
    if jaxpr is None:
        jaxpr = eqn.params.get('call_jaxpr')

    inner = constant_fold_jaxpr(jaxpr.jaxpr)
    name = eqn.params.get('name', 'jitted')
    if not isinstance(name, str) or not name.isidentifier():
        name = 'jitted'
    fn_name = state.skolem(name)
    fn_ast = jaxpr_to_py_ast(state, inner, fn_name)

    jitted_fn = ast.Call(
        func=ast.Attribute(ast.Name(id='jax', ctx=ast.Load()), attr='jit', ctx=ast.Load()),
        args=[ast.Name(id=fn_name, ctx=ast.Load())],
        keywords=[],
    )
    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=jitted_fn,
            args=[_astify_atom(state, v) for v in eqn.invars],
            keywords=[],
        ),
    )
    return [fn_ast, assign]


# The jit primitive is named 'pjit' in older JAX and 'jit' in newer releases.
register_prim_handler('pjit', _astify_pjit)
register_prim_handler('jit', _astify_pjit)


@register_prim_as('remat2')
def _astify_remat(state: SourcerorState, eqn):
    # out = partial_eval_jaxpr(eqn.params['call_jaxpr'].jaxpr,
    #                          {var: val for var, val in zip(eqn.params['call_jaxpr'].jaxpr.invars, vals)})
    call_japr = constant_fold_jaxpr(eqn.params['jaxpr'])
    fn_name = state.skolem('fn')

    fn_ast = jaxpr_to_py_ast(state, call_japr, fn_name)

    invars = [_astify_atom(state, v) for v in eqn.invars]
    outvars = _astify_outvars(state, eqn.outvars)

    lam = ast.Assign(
        targets=[ast.Name(id=f"ckpt_{fn_name}", ctx=ast.Store())],
        # value=ast.parse(f"jax.checkpoint({fn_name})").body[0]
        value=ast.Call(
            func=ast.Name(id='jax.checkpoint', ctx=ast.Load()),
            args=[ast.Name(id=fn_name, ctx=ast.Load())],
            keywords=[])
    )

    assign = ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id=f"ckpt_{fn_name}"),
            args=invars,
            keywords=[]
        ))

    return [fn_ast, lam, assign]


@register_prim_as('reshape')
def _astify_reshape(state, eqn):
    # the lax reshape is a bit different, because it can combine a transpose and reshape into one.
    # np.reshape(np.transpose(operand, dimensions), new_sizes)
    dimensions = eqn.params['dimensions']
    new_sizes = eqn.params['new_sizes']

    source = _astify_atom(state, eqn.invars[0])

    if dimensions is not None:
        source = ast.Call(
            func=ast.Name(id='jax.numpy.transpose', ctx=ast.Load()),
            args=[source, _astify_value(dimensions)],
            keywords=[]
        )

    assign = ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=ast.Call(
            func=ast.Name(id='jax.numpy.reshape', ctx=ast.Load()),
            args=[source, _astify_value(new_sizes)],
            keywords=[]
        ))

    return [assign]


@register_prim_as('add_any')
def _astify_add_any(state, eqn):
    # add_any is a weird undocumented jax primitive. best guess is it adds?
    return _binop_fn(ast.Add())(state, eqn)


def _broadcast_in_dim_general(state, eqn):
    # Emit jax.lax.broadcast_in_dim(operand, shape, broadcast_dimensions),
    # forwarding only the arguments the public function accepts (recent JAX adds
    # a ``sharding`` param that must not be passed through).
    invar = _astify_atom(state, eqn.invars[0])
    outvars = _astify_outvars(state, eqn.outvars)
    return ast.Assign(
        targets=outvars,
        value=ast.Call(
            func=ast.Name(id='jax.lax.broadcast_in_dim', ctx=ast.Load()),
            args=[invar,
                  _astify_value(eqn.params['shape']),
                  _astify_value(eqn.params['broadcast_dimensions'])],
            keywords=[],
        ),
    )


@register_prim_as('broadcast_in_dim')
def _astify_broadcast_in_dim(state, eqn):
    # broadcast_in_dim is how zeros, ones, full, etc are implemented,
    # so we prefer to use those where possible
    assert len(eqn.invars) == 1
    value = eqn.invars[0]
    shape = eqn.params['shape']
    broadcast_dimensions = eqn.params['broadcast_dimensions']

    if not isinstance(value, Literal) or broadcast_dimensions != ():
        return _broadcast_in_dim_general(state, eqn)

    if not isinstance(value.val, np.ndarray) or value.val.ndim != 0:
        return _broadcast_in_dim_general(state, eqn)
    else:
        constant_value = value.val.item()
        if constant_value == 0:
            call = ast.Call(
                ast.Attribute(
                    value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                    attr='zeros',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape),
                      _astify_value(value.val.dtype)],
                keywords=[]
            )
        elif constant_value == 1:
            call = ast.Call(
                ast.Attribute(
                    value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                    attr='ones',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape),
                      _astify_value(value.val.dtype)],
                keywords=[]
            )
        else:
            call = ast.Call(
                ast.Attribute(
                    value=ast.Name(id='jax.numpy', ctx=ast.Load()),
                    attr='full',
                    ctx=ast.Load()
                ),
                args=[_astify_value(shape),
                      _astify_value(constant_value),
                      _astify_value(value.val.dtype)],
                keywords=[]
            )

        return [ast.Assign(
            targets=_astify_outvars(state, eqn.outvars),
            value=call
        )]


@register_prim_as('random_wrap')
def _astify_random_wrap(state, eqn):
    # we treat this as a noop
    return ast.Assign(
        targets=_astify_outvars(state, eqn.outvars),
        value=_astify_atom(state, eqn.invars[0])
    )


constant_fold_blacklist = {
    'broadcast_in_dim',
    'broadcast',
    # Control flow: preserve the structure so dedicated handlers emit code for
    # it rather than eagerly executing it during constant folding.
    'scan',
    'cond',
    'while',
    # Randomness / callbacks: never eagerly execute at code-generation time.
    'rng_uniform',
    'rng_bit_generator',
    'random_seed',
    'random_bits',
    'random_fold_in',
    'random_split',
    'random_wrap',
    'random_unwrap',
    'debug_callback',
    'io_callback',
    'pure_callback',
}
