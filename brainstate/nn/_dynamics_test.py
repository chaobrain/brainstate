# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import unittest

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
from brainstate import _testing
from brainstate.nn._dynamics import (
    Dynamics,
    Prefetch,
    PrefetchDelay,
    PrefetchDelayAt,
    OutputDelayAt,
    init_maybe_prefetch,
    receive_update_output,
    not_receive_update_output,
    receive_update_input,
    not_receive_update_input,
    _get_prefetch_item,
    _get_prefetch_item_delay,
    _get_prefetch_delay_key,
    _get_output_delay_key,
)


class _Neuron(Dynamics):
    """Minimal concrete ``Dynamics`` subclass holding a single voltage state.

    Used across the dynamics tests to exercise state init, prefetch, and delay
    helpers without depending on a full neuron model implementation.
    """

    def init_state(self, *args, **kwargs):
        """Allocate the voltage hidden state with the module's variable shape."""
        self.V = brainstate.HiddenState(jnp.zeros(self.varshape))

    def update(self, x=0.0):
        """Accumulate the input into the voltage and return the new value."""
        self.V.value = self.V.value + x
        return self.V.value


class TestModuleGroup(unittest.TestCase):
    def test_initialization(self):
        group = brainstate.nn.DynamicsGroup()
        self.assertIsInstance(group, brainstate.nn.DynamicsGroup)


class TestDynamics(unittest.TestCase):
    def test_initialization(self):
        dyn = brainstate.nn.Dynamics(in_size=10)
        self.assertIsInstance(dyn, brainstate.nn.Dynamics)
        self.assertEqual(dyn.in_size, (10,))
        self.assertEqual(dyn.out_size, (10,))

    def test_size_validation(self):
        with self.assertRaises(ValueError):
            brainstate.nn.Dynamics(in_size=[])
        with self.assertRaises(ValueError):
            brainstate.nn.Dynamics(in_size="invalid")

    def test_varshape(self):
        dyn = brainstate.nn.Dynamics(in_size=(2, 3))
        self.assertEqual(dyn.varshape, (2, 3))
        dyn = brainstate.nn.Dynamics(in_size=(2, 3))
        self.assertEqual(dyn.varshape, (2, 3))


class TestDynamicsSizeNormalization(unittest.TestCase):
    """Verify ``in_size``/``out_size``/``varshape`` normalization and validation."""

    def test_int_is_wrapped_in_tuple(self):
        """A scalar ``in_size`` is normalized to a 1-tuple."""
        dyn = Dynamics(in_size=7)
        self.assertEqual(dyn.in_size, (7,))
        self.assertEqual(dyn.out_size, (7,))
        self.assertEqual(dyn.varshape, (7,))

    def test_numpy_integer_is_accepted(self):
        """A numpy integer scalar is accepted as ``in_size``."""
        dyn = Dynamics(in_size=np.int64(5))
        self.assertEqual(dyn.in_size, (5,))

    def test_list_is_converted_to_tuple(self):
        """A list ``in_size`` is converted to a tuple and preserved."""
        dyn = Dynamics(in_size=[2, 3, 4])
        self.assertEqual(dyn.in_size, (2, 3, 4))
        self.assertEqual(dyn.varshape, (2, 3, 4))

    def test_empty_sequence_raises(self):
        """An empty sequence ``in_size`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            Dynamics(in_size=())

    def test_non_integer_first_element_raises(self):
        """A sequence whose first element is not an int raises ``ValueError``."""
        with self.assertRaises(ValueError):
            Dynamics(in_size=(1.5, 2))

    def test_non_integer_scalar_raises(self):
        """A non-int, non-sequence ``in_size`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            Dynamics(in_size=3.5)

    def test_name_is_recorded(self):
        """The optional ``name`` argument is exposed read-only."""
        dyn = Dynamics(in_size=4, name='my_dyn')
        self.assertEqual(dyn.name, 'my_dyn')


class TestDynamicsUpdateHooks(unittest.TestCase):
    """Cover registration, lookup, and dispatch of before/after update hooks."""

    def test_before_updates_starts_none(self):
        """Before/after update registries start as ``None``."""
        dyn = Dynamics(3)
        self.assertIsNone(dyn.before_updates)
        self.assertIsNone(dyn.after_updates)

    def test_add_and_get_before_update(self):
        """A registered before-update is retrievable and reported as present."""
        dyn = Dynamics(3)
        fn = lambda: None
        dyn.add_before_update('k', fn)
        self.assertTrue(dyn.has_before_update('k'))
        self.assertIs(dyn.get_before_update('k'), fn)

    def test_add_and_get_after_update(self):
        """A registered after-update is retrievable and reported as present."""
        dyn = Dynamics(3)
        fn = lambda ret: None
        dyn.add_after_update('k', fn)
        self.assertTrue(dyn.has_after_update('k'))
        self.assertIs(dyn.get_after_update('k'), fn)

    def test_duplicate_before_key_raises(self):
        """Re-registering a before-update key raises ``KeyError``."""
        dyn = Dynamics(3)
        dyn.add_before_update('k', lambda: None)
        with self.assertRaises(KeyError):
            dyn.add_before_update('k', lambda: None)

    def test_duplicate_after_key_raises(self):
        """Re-registering an after-update key raises ``KeyError``."""
        dyn = Dynamics(3)
        dyn.add_after_update('k', lambda r: None)
        with self.assertRaises(KeyError):
            dyn.add_after_update('k', lambda r: None)

    def test_get_missing_before_when_none_raises(self):
        """Getting a before-update when none are registered raises ``KeyError``."""
        dyn = Dynamics(3)
        with self.assertRaises(KeyError):
            dyn.get_before_update('missing')

    def test_get_missing_before_when_present_raises(self):
        """Getting an unknown before-update key raises ``KeyError``."""
        dyn = Dynamics(3)
        dyn.add_before_update('present', lambda: None)
        with self.assertRaises(KeyError):
            dyn.get_before_update('missing')

    def test_get_missing_after_when_none_raises(self):
        """Getting an after-update when none are registered raises ``KeyError``."""
        dyn = Dynamics(3)
        with self.assertRaises(KeyError):
            dyn.get_after_update('missing')

    def test_get_missing_after_when_present_raises(self):
        """Getting an unknown after-update key raises ``KeyError``."""
        dyn = Dynamics(3)
        dyn.add_after_update('present', lambda r: None)
        with self.assertRaises(KeyError):
            dyn.get_after_update('missing')

    def test_has_update_false_when_none(self):
        """``has_*_update`` returns False when no registry exists."""
        dyn = Dynamics(3)
        self.assertFalse(dyn.has_before_update('x'))
        self.assertFalse(dyn.has_after_update('x'))

    def test_call_dispatches_hooks_in_order(self):
        """``__call__`` runs before hooks, ``update``, then after hooks."""

        class _N(Dynamics):
            """Tiny dynamics returning input plus one."""

            def update(self, x=0.0):
                """Return ``x + 1``."""
                return x + 1

        dyn = _N(2)
        log = []

        class _BefInput:
            """Before hook that receives the update input."""

            def __call__(self, *a, **k):
                log.append(('bef_in', a))

        receive_update_input(_BefInput)
        dyn.add_before_update('b_in', _BefInput())

        class _BefNoInput:
            """Before hook that takes no input."""

            def __call__(self):
                log.append(('bef_noin',))

        dyn.add_before_update('b_noin', _BefNoInput())

        class _AftOutput:
            """After hook that receives the update output."""

            def __call__(self, ret):
                log.append(('aft_out', ret))

        dyn.add_after_update('a_out', _AftOutput())

        class _AftNoOutput:
            """After hook that ignores the update output."""

            def __call__(self):
                log.append(('aft_noout',))

        not_receive_update_output(_AftNoOutput)
        dyn.add_after_update('a_noout', _AftNoOutput())

        result = dyn(5)
        self.assertEqual(result, 6)
        self.assertEqual(
            log,
            [('bef_in', (5,)), ('bef_noin',), ('aft_out', 6), ('aft_noout',)],
        )


class TestUpdateDecorators(unittest.TestCase):
    """Cover the receive/not-receive input/output marker decorators."""

    def test_receive_then_not_receive_input(self):
        """``receive_update_input`` sets a marker that ``not_receive`` removes."""

        class _C:
            """Marker test target."""

        receive_update_input(_C)
        self.assertTrue(hasattr(_C, '_receive_update_input'))
        not_receive_update_input(_C)
        self.assertFalse(hasattr(_C, '_receive_update_input'))

    def test_not_receive_input_when_absent_is_noop(self):
        """``not_receive_update_input`` is a no-op when the marker is absent."""

        class _C:
            """Marker test target."""

        not_receive_update_input(_C)
        self.assertFalse(hasattr(_C, '_receive_update_input'))

    def test_not_receive_then_receive_output(self):
        """``not_receive_update_output`` sets a marker that ``receive`` removes."""

        class _C:
            """Marker test target."""

        not_receive_update_output(_C)
        self.assertTrue(hasattr(_C, '_not_receive_update_output'))
        receive_update_output(_C)
        self.assertFalse(hasattr(_C, '_not_receive_update_output'))

    def test_receive_output_when_absent_is_noop(self):
        """``receive_update_output`` is a no-op when the marker is absent."""

        class _C:
            """Marker test target."""

        receive_update_output(_C)
        self.assertFalse(hasattr(_C, '_not_receive_update_output'))


class TestPrefetch(unittest.TestCase):
    """Cover the ``Prefetch`` reference object and its accessors."""

    def test_prefetch_returns_prefetch(self):
        """``Dynamics.prefetch`` returns a ``Prefetch`` bound to the module."""
        dyn = _Neuron(3)
        dyn.init_all_states()
        pf = dyn.prefetch('V')
        self.assertIsInstance(pf, Prefetch)
        self.assertIs(pf.module, dyn)
        self.assertEqual(pf.item, 'V')

    def test_call_returns_state_value(self):
        """Calling a ``Prefetch`` on a ``State`` returns its value array."""
        dyn = _Neuron(3)
        dyn.init_all_states()
        pf = dyn.prefetch('V')
        _testing.assert_allclose(pf(), jnp.zeros(3))

    def test_get_item_value_matches_call(self):
        """``get_item_value`` returns the same value as ``__call__``."""
        dyn = _Neuron(3)
        dyn.init_all_states()
        pf = dyn.prefetch('V')
        _testing.assert_allclose(pf.get_item_value(), pf())

    def test_get_item_returns_state_object(self):
        """``get_item`` returns the underlying ``State`` object, not its value."""
        dyn = _Neuron(3)
        dyn.init_all_states()
        pf = dyn.prefetch('V')
        self.assertIsInstance(pf.get_item(), brainstate.State)

    def test_call_returns_plain_attribute(self):
        """A prefetch of a non-State attribute returns the attribute itself."""
        dyn = _Neuron(3)
        dyn.plain = 42
        pf = dyn.prefetch('plain')
        self.assertEqual(pf(), 42)

    def test_missing_attribute_raises(self):
        """Prefetching a missing attribute raises ``AttributeError`` on access."""
        dyn = _Neuron(3)
        pf = dyn.prefetch('does_not_exist')
        with self.assertRaises(AttributeError):
            pf()

    def test_delay_property_returns_prefetch_delay(self):
        """The ``.delay`` property returns a ``PrefetchDelay``."""
        dyn = _Neuron(3)
        pd = dyn.prefetch('V').delay
        self.assertIsInstance(pd, PrefetchDelay)
        self.assertEqual(pd.item, 'V')


class TestPrefetchDelay(unittest.TestCase):
    """Cover the delayed-access prefetch chain (``PrefetchDelay``/``PrefetchDelayAt``)."""

    def test_delay_at_returns_prefetch_delay_at(self):
        """``PrefetchDelay.at`` returns a ``PrefetchDelayAt``."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            at = dyn.prefetch('V').delay.at(0.5 * u.ms)
            self.assertIsInstance(at, PrefetchDelayAt)

    def test_prefetch_delay_registers_after_update(self):
        """``prefetch_delay`` registers a ``StateWithDelay`` after-update hook."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            dyn.prefetch_delay('V', 1.0 * u.ms)
            key = _get_prefetch_delay_key('V', None)
            self.assertTrue(dyn.has_after_update(key))

    def test_prefetch_delay_retrieves_past_value(self):
        """A registered prefetch delay retrieves the state value from the past."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            delayed = dyn.prefetch_delay('V', 1.0 * u.ms)
            dyn.init_all_states()
            for i in range(20):
                with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
                    dyn(jnp.ones(3))
            # 10 steps of delay -> value 10 steps ago (current is 20).
            _testing.assert_allclose(delayed(), jnp.ones(3) * 10.0)

    def test_prefetch_delay_at_empty_time_returns_current(self):
        """A ``PrefetchDelayAt`` built with no delay time returns the current value."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            dyn.init_all_states()
            at = PrefetchDelayAt(dyn, 'V')
            self.assertEqual(len(at.delay_time), 0)
            _testing.assert_allclose(at(), jnp.zeros(3))

    def test_prefetch_delay_at_unpacks_single_sequence(self):
        """A single tuple of (time,) argument is unpacked to scalar delay time."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            at = PrefetchDelayAt(dyn, 'V', (1.0 * u.ms,))
            self.assertEqual(len(at.delay_time), 1)

    def test_prefetch_delay_at_requires_dynamics(self):
        """Constructing a ``PrefetchDelayAt`` with a non-Dynamics module fails."""
        with self.assertRaises(AssertionError):
            PrefetchDelayAt(object(), 'V')


class TestOutputDelay(unittest.TestCase):
    """Cover the module output delay helper ``OutputDelayAt``."""

    def test_output_delay_registers_after_update(self):
        """``output_delay`` registers an output-delay after-update hook."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            dyn.output_delay(1.0 * u.ms)
            self.assertTrue(dyn.has_after_update(_get_output_delay_key(None)))

    def test_output_delay_retrieves_past_output(self):
        """A registered output delay retrieves the module output from the past."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            delayed = dyn.output_delay(1.0 * u.ms)
            dyn.init_all_states()
            for i in range(20):
                with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
                    dyn(jnp.ones(3))
            _testing.assert_allclose(delayed(), jnp.ones(3) * 10.0)

    def test_output_delay_requires_dynamics(self):
        """Constructing an ``OutputDelayAt`` with a non-Dynamics module fails."""
        with self.assertRaises(AssertionError):
            OutputDelayAt(object(), 1.0 * u.ms)


class TestInitMaybePrefetch(unittest.TestCase):
    """Cover the ``init_maybe_prefetch`` dispatcher over prefetch object types."""

    def test_init_prefetch(self):
        """Initializing a ``Prefetch`` resolves the referenced item."""
        dyn = _Neuron(3)
        dyn.init_all_states()
        init_maybe_prefetch(dyn.prefetch('V'))  # should not raise

    def test_init_prefetch_missing_attribute_raises(self):
        """Initializing a ``Prefetch`` of a missing attribute raises ``AttributeError``."""
        dyn = _Neuron(3)
        with self.assertRaises(AttributeError):
            init_maybe_prefetch(Prefetch(dyn, 'missing'))

    def test_init_prefetch_delay_at_is_noop(self):
        """Initializing a ``PrefetchDelayAt`` performs no action."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            at = dyn.prefetch_delay('V', 1.0 * u.ms)
            init_maybe_prefetch(at)  # PrefetchDelayAt branch: pass

    def test_init_prefetch_delay_hits_key_arity_bug(self):
        """Initializing a ``PrefetchDelay`` hits the key-arity bug (documents BUG).

        The intended behavior is to resolve the registered delay handler, but
        ``_get_prefetch_item_delay`` calls ``_get_prefetch_delay_key(item)`` with
        a single argument while that helper now requires ``(item, update_every)``,
        so a ``TypeError`` is raised before the lookup can complete.
        """
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            dyn.prefetch_delay('V', 1.0 * u.ms)
            with self.assertRaises(TypeError):
                init_maybe_prefetch(dyn.prefetch('V').delay)


class TestPrefetchItemDelayHelper(unittest.TestCase):
    """Cover reachable lines of the ``_get_prefetch_item_delay`` helper."""

    def test_non_dynamics_module_asserts(self):
        """The helper asserts that the target module is a ``Dynamics``."""

        class _Fake:
            """Stand-in target whose module is not a Dynamics."""

            module = object()
            item = 'V'

        with self.assertRaises(AssertionError):
            _get_prefetch_item_delay(_Fake())

    def test_key_helper_arity_bug_raises_type_error(self):
        """The helper hits the key-arity bug and raises ``TypeError`` (documents BUG)."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            dyn.prefetch_delay('V', 1.0 * u.ms)
            with self.assertRaises(TypeError):
                _get_prefetch_item_delay(dyn.prefetch('V').delay)


class TestDelayRegistrationReuse(unittest.TestCase):
    """Cover the reuse path when a delay hook is already registered."""

    def test_second_prefetch_delay_reuses_hook(self):
        """A second ``prefetch_delay`` on the same state reuses the after-update hook."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            first = dyn.prefetch_delay('V', 1.0 * u.ms)
            second = dyn.prefetch_delay('V', 2.0 * u.ms)
            # Both share the same StateWithDelay hook instance.
            self.assertIs(first.state_delay, second.state_delay)

    def test_second_output_delay_reuses_hook(self):
        """A second ``output_delay`` reuses the registered output-delay hook."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            dyn = _Neuron(3)
            first = dyn.output_delay(1.0 * u.ms)
            second = dyn.output_delay(2.0 * u.ms)
            self.assertIs(first.out_delay, second.out_delay)


class TestDelayKeyHelpers(unittest.TestCase):
    """Cover the internal delay-key formatting helpers."""

    def test_prefetch_delay_key_format(self):
        """``_get_prefetch_delay_key`` embeds the item and update_every."""
        self.assertEqual(_get_prefetch_delay_key('V', None), 'V-prefetch-delay-None')
        self.assertEqual(_get_prefetch_delay_key('V', 5.0), 'V-prefetch-delay-5.0')

    def test_output_delay_key_format(self):
        """``_get_output_delay_key`` embeds the update_every value."""
        self.assertEqual(_get_output_delay_key(None), 'output-delay-None')


if __name__ == '__main__':
    unittest.main()
