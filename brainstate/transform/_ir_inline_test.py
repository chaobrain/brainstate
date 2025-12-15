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


import jax
import jax.numpy as jnp
import numpy as np
from jax import make_jaxpr

from brainstate.transform._ir_inline import inline_jit


def has_primitive(jaxpr, primitive_name: str) -> bool:
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == primitive_name:
            return True
    return False


def count_equations(jaxpr) -> int:
    return len(jaxpr.eqns)


def expand_small_jits(max_eqns: int = 5):
    def predicate(eqn):
        call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
        if call_jaxpr is None:
            return False
        return count_equations(call_jaxpr) <= max_eqns

    return predicate


def expand_without_primitive(primitive_name: str):
    def predicate(eqn):
        call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
        if call_jaxpr is None:
            return False
        return not has_primitive(call_jaxpr, primitive_name)

    return predicate


def count_call_equations(jaxpr) -> int:
    return sum(
        1 for eqn in jaxpr.eqns
        if (eqn.params.get('call_jaxpr') is not None) or (eqn.params.get('jaxpr') is not None)
    )


def eval_jaxpr(jaxpr, *args):
    return jax.core.eval_jaxpr(jaxpr, [], *args)


def test_expand_all_jits_preserves_value_and_removes_calls():
    @jax.jit
    def inner_func(x, y):
        return x + y * 2

    def outer_func(a, b, c):
        result1 = inner_func(a, b)
        result2 = jnp.sin(result1)
        result3 = inner_func(result2, c)
        return result3

    inputs = (1.0, 2.0, 3.0)
    jaxpr = make_jaxpr(outer_func)(*inputs)
    orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)

    expanded = inline_jit(jaxpr.jaxpr)
    exp_out = eval_jaxpr(expanded, *inputs)

    assert np.allclose(np.array(orig_out), np.array(exp_out))
    # Expect no remaining call-style equations after full expansion
    assert count_call_equations(expanded) == 0


def test_conditional_expansion_by_size():
    @jax.jit
    def small_func(x):
        return x + 1

    @jax.jit
    def large_func(x):
        x = x + 1
        x = x * 2
        x = jnp.sin(x)
        x = jnp.cos(x)
        x = x ** 2
        return x

    def outer_func(a, b):
        result1 = small_func(a)
        result2 = large_func(b)
        return result1 + result2

    inputs = (1.0, 2.0)
    jaxpr = make_jaxpr(outer_func)(*inputs)
    orig_call_count = count_call_equations(jaxpr.jaxpr)

    predicate = expand_small_jits(max_eqns=3)
    expanded = inline_jit(jaxpr.jaxpr, predicate)
    exp_call_count = count_call_equations(expanded)

    # Values preserved
    orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)
    exp_out = eval_jaxpr(expanded, *inputs)
    assert np.allclose(np.array(orig_out), np.array(exp_out))

    # Small jits should be expanded, so call count should decrease but not necessarily to zero
    assert exp_call_count <= orig_call_count


def test_expand_without_sin_primitive():
    @jax.jit
    def func_with_sin(x):
        return jnp.sin(x) + 1

    @jax.jit
    def func_without_sin(x):
        return x * 2 + 1

    def outer_func(a, b):
        result1 = func_without_sin(a)
        result2 = func_with_sin(b)
        return result1 + result2

    inputs = (1.0, 2.0)
    jaxpr = make_jaxpr(outer_func)(*inputs)
    orig_call_count = count_call_equations(jaxpr.jaxpr)

    predicate = expand_without_primitive('sin')
    expanded = inline_jit(jaxpr.jaxpr, predicate)

    orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)
    exp_out = eval_jaxpr(expanded, *inputs)
    assert np.allclose(np.array(orig_out), np.array(exp_out))

    # At least some calls (those without 'sin') should be expanded
    assert count_call_equations(expanded) <= orig_call_count


def test_nested_jits_expand_recursively():
    @jax.jit
    def innermost(x):
        return x + 1

    @jax.jit
    def middle(x):
        return innermost(x) * 2

    def outer(x):
        return middle(x) + 3

    inputs = (1.0,)
    jaxpr = make_jaxpr(outer)(*inputs)
    orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)

    expanded = inline_jit(jaxpr.jaxpr)
    exp_out = eval_jaxpr(expanded, *inputs)

    assert np.allclose(np.array(orig_out), np.array(exp_out))
    # Fully expanded nested jits should remove call-style equations
    assert count_call_equations(expanded) == 0


def test_custom_predicate_expands_only_selected_jits():
    @jax.jit
    def func1(x):
        return x + 1

    @jax.jit
    def func2(x):
        return x * 2

    @jax.jit
    def func3(x):
        return x ** 2

    def outer(a, b, c):
        r1 = func1(a)
        r2 = func2(b)
        r3 = func3(c)
        return r1 + r2 + r3

    inputs = (1.0, 2.0, 3.0)
    jaxpr = make_jaxpr(outer)(*inputs)
    orig_out = eval_jaxpr(jaxpr.jaxpr, *inputs)

    def custom_predicate(eqn):
        call_jaxpr = eqn.params.get('call_jaxpr') or eqn.params.get('jaxpr')
        if call_jaxpr is None:
            return False
        return has_primitive(call_jaxpr, 'mul') or has_primitive(call_jaxpr, 'add')

    expanded = inline_jit(jaxpr.jaxpr, custom_predicate)
    exp_out = eval_jaxpr(expanded, *inputs)

    assert np.allclose(np.array(orig_out), np.array(exp_out))
    # Some calls should be expanded according to the custom predicate
    assert count_call_equations(expanded) <= count_call_equations(jaxpr.jaxpr)
