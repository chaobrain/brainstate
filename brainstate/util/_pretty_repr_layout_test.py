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

"""Tests for the width-aware layout, options, and formatting of the pretty-repr engine."""

import dataclasses
import os
import threading
import unittest
from unittest import mock

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate.util import (
    PrettyAttr,
    PrettyRepr,
    PrettyType,
    ReprOptions,
    ReprSummary,
    format_array_like,
    format_bytes,
    get_repr_options,
    pretty_repr_object,
    pretty_format,
    PrettyDict,
    PrettyList,
    repr_options,
    set_repr_options,
)
from brainstate.util._pretty_repr import CONTEXT


class Obj(PrettyRepr):
    """A configurable PrettyRepr used to drive the renderer."""

    def __init__(self, name='Obj', **attrs):
        self._name = name
        self._attrs = attrs

    def __pretty_repr__(self):
        yield PrettyType(self._name)
        for k, v in self._attrs.items():
            yield PrettyAttr(k, v)


class Summarised(Obj):
    """A PrettyRepr exposing a summary hook, like ``graph.Node``."""

    def __init__(self, summary, **attrs):
        super().__init__('Summarised', **attrs)
        self._summary = summary

    def __pretty_repr_summary__(self):
        return self._summary


class TestFlatVersusBroken(unittest.TestCase):

    def test_fits_on_one_line(self):
        self.assertEqual(repr(Obj(a=1, b=[1, 2])), 'Obj(a=1, b=[1, 2])')

    def test_width_boundary(self):
        obj = Obj(a=1, b=2)  # 'Obj(a=1, b=2)' is 13 columns
        with repr_options(width=13):
            self.assertEqual(repr(obj), 'Obj(a=1, b=2)')
        with repr_options(width=12):
            self.assertEqual(repr(obj), 'Obj(\n  a=1,\n  b=2,\n)')

    def test_nested_only_outer_breaks(self):
        obj = Obj(inner=Obj('In', x=1), long='x' * 30)
        with repr_options(width=40):
            self.assertEqual(
                repr(obj),
                'Obj(\n  inner=In(x=1),\n  long=' + 'x' * 30 + ',\n)',
            )

    def test_nested_inner_breaks_too(self):
        obj = Obj(inner=Obj('In', a='a' * 20, b='b' * 20))
        with repr_options(width=30):
            self.assertEqual(
                repr(obj),
                'Obj(\n  inner=In(\n    a=' + 'a' * 20 + ',\n    b=' + 'b' * 20 + ',\n  ),\n)',
            )

    def test_empty_object(self):
        self.assertEqual(repr(Obj()), 'Obj()')

    def test_multiline_leaf_forces_break_and_reindents(self):
        class Multi:
            def __repr__(self):
                return 'line1\nline2'

        self.assertEqual(
            repr(Obj(m=Multi(), k=1)),
            'Obj(\n  m=line1\n  line2,\n  k=1,\n)',
        )

    def test_positional_items(self):
        class Pos(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType('Pos')
                yield PrettyAttr('', '(3,) -> (4,)')
                yield PrettyAttr('k', 1)

        self.assertEqual(repr(Pos()), 'Pos((3,) -> (4,), k=1)')

    def test_attr_start_end_are_kept(self):
        class Wrapped(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType('W')
                yield PrettyAttr('k', 1, start='<', end='>')

        self.assertEqual(repr(Wrapped()), 'W(<k=1>)')

    def test_explicit_comment_only_when_broken(self):
        class Commented(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType('C')
                yield PrettyAttr('k', 1, comment='note')

        self.assertEqual(repr(Commented()), 'C(k=1)')
        with repr_options(width=4):
            self.assertEqual(repr(Commented()), 'C(\n  k=1,  # note\n)')

    def test_plain_containers(self):
        obj = Obj(t=(1,), t2=(1, 2), l=[], d={'a': 'x'}, s='str')
        self.assertEqual(repr(obj), "Obj(t=(1,), t2=(1, 2), l=[], d={'a': 'x'}, s=str)")

    def test_single_tuple_broken_has_no_double_comma(self):
        obj = Obj(t=('a' * 30,))
        with repr_options(width=20):
            self.assertEqual(repr(obj), "Obj(\n  t=(\n    '" + 'a' * 30 + "',\n  ),\n)")

    def test_types_render_by_name(self):
        self.assertEqual(repr(Obj(t=int, d=np.float32)), 'Obj(t=int, d=float32)')

    def test_generated_dataclass_repr_is_laid_out(self):
        @dataclasses.dataclass
        class DC:
            a: int
            b: str
            hidden: int = dataclasses.field(default=0, repr=False)

        self.assertEqual(repr(Obj(dc=DC(1, 'x'))), "Obj(dc=DC(a=1, b='x'))")

    def test_handwritten_dataclass_repr_is_respected(self):
        @dataclasses.dataclass
        class DC:
            a: int

            def __repr__(self):
                return 'custom'

        self.assertEqual(repr(Obj(dc=DC(1))), 'Obj(dc=custom)')

    def test_invalid_item_raises(self):
        class Bad(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType('Bad')
                yield 'oops'

        with self.assertRaises(TypeError):
            repr(Bad())


class TestPrettyFormat(unittest.TestCase):

    def test_plain_values(self):
        self.assertEqual(pretty_format([np.zeros(2, np.float32), 'a', {'k': (1,)}]), "[f32[2], 'a', {'k': (1,)}]")
        self.assertEqual(pretty_format(3), '3')

    def test_pretty_objects_and_width(self):
        self.assertEqual(pretty_format(Obj(a=1)), 'Obj(a=1)')
        with repr_options(width=6):
            self.assertEqual(pretty_format([10, 20]), '[\n  10,\n  20,\n]')

    def test_color(self):
        self.assertIn('\x1b[', pretty_format(Obj(a=1), color=True))


class TestReferences(unittest.TestCase):

    def test_shared_reference(self):
        child = Obj('Child', x=1)
        self.assertEqual(repr(Obj(a=child, b=child)), 'Obj(a=Child(x=1), b=<shared: .a>)')

    def test_shared_in_list_and_dict_paths(self):
        child = Obj('Child')
        self.assertEqual(
            repr(Obj(l=[child], d={'k': child})),
            "Obj(l=[Child()], d={'k': <shared: .l[0]>})",
        )

    def test_cycle_to_root(self):
        obj = Obj(x=1)
        obj._attrs['me'] = obj
        self.assertEqual(repr(obj), 'Obj(x=1, me=<cycle: .>)')

    def test_cycle_to_inner(self):
        inner = Obj('In')
        inner._attrs['parent'] = inner
        self.assertEqual(repr(Obj(a=inner)), 'Obj(a=In(parent=<cycle: .a>))')

    def test_self_referencing_plain_containers(self):
        lst = [1]
        lst.append(lst)
        dct = {'a': 1}
        dct['self'] = dct
        self.assertEqual(repr(Obj(l=lst, d=dct)), "Obj(l=[1, <cycle: .l>], d={'a': 1, 'self': <cycle: .d>})")

    def test_self_referencing_dataclass(self):
        @dataclasses.dataclass
        class Link:
            nxt: object = None

        link = Link()
        link.nxt = link
        self.assertEqual(repr(Obj(x=link)), 'Obj(x=Link(nxt=<cycle: .x>))')

    def test_repeated_plain_container_is_not_shared(self):
        shared = [1, 2]
        self.assertEqual(repr(Obj(a=shared, b=shared)), 'Obj(a=[1, 2], b=[1, 2])')

    def test_pretty_repr_returning_plain_iterable(self):
        class ListRepr(PrettyRepr):
            def __pretty_repr__(self):
                return [PrettyType('L'), PrettyAttr('k', 1)]

        class EmptyRepr(PrettyRepr):
            def __pretty_repr__(self):
                return iter(())

        self.assertEqual(repr(ListRepr()), 'L(k=1)')
        with repr_options(max_depth=0):
            self.assertEqual(repr(Obj(a=ListRepr())), 'Obj(a=L(...))')
        with self.assertRaises(TypeError):
            repr(EmptyRepr())

    def test_header_ref(self):
        class Ref(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType('Ref', ref='<marker>')

        self.assertEqual(repr(Obj(r=Ref())), 'Obj(r=<marker>)')

    def test_sequence_of_objects_gets_index_labels_when_broken(self):
        obj = Obj(layers=[Obj('A', v='a' * 30), Obj('B')])
        self.assertEqual(repr(obj), "Obj(layers=[A(v=" + 'a' * 30 + "), B()])")
        with repr_options(width=30):
            self.assertEqual(
                repr(obj),
                'Obj(\n  layers=[\n    [0] A(\n      v=' + 'a' * 30 + ',\n    ),\n    [1] B(),\n  ],\n)',
            )


class TestLimits(unittest.TestCase):

    def test_max_depth(self):
        obj = Obj(a=Obj('A', b=Obj('B', c=1)), l=[Obj('C')])
        with repr_options(max_depth=0):
            self.assertEqual(repr(obj), 'Obj(a=A(...), l=[C(...)])')
        with repr_options(max_depth=1):
            self.assertEqual(repr(obj), 'Obj(a=A(b=B(...)), l=[C()])')
        with repr_options(max_depth=None):
            self.assertEqual(repr(obj), 'Obj(a=A(b=B(c=1)), l=[C()])')

    def test_max_items_for_objects_and_containers(self):
        obj = Obj(**{f'k{i}': i for i in range(5)})
        with repr_options(max_items=2):
            self.assertEqual(repr(obj), 'Obj(k0=0, k1=1, ... (+3 more))')
            self.assertEqual(repr(Obj(l=list(range(5)))), 'Obj(l=[0, 1, ... (+3 more)])')
            self.assertEqual(repr(Obj(d={1: 1, 2: 2, 3: 3})), 'Obj(d={1: 1, 2: 2, ... (+1 more)})')


class TestOptions(unittest.TestCase):

    def test_defaults(self):
        options = ReprOptions()
        self.assertEqual(options.width, 88)
        self.assertIsNone(options.max_depth)
        self.assertEqual(options.max_items, 64)
        self.assertTrue(options.summary)
        self.assertTrue(options.short_dtypes)

    def test_validation(self):
        for kwargs in (
            dict(width=0), dict(width='88'), dict(max_depth=-1),
            dict(max_items=0), dict(max_items=1.5),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReprOptions(**kwargs)
        with self.assertRaises(TypeError):
            with repr_options(unknown=1):
                pass

    def test_context_nesting_and_restore_on_error(self):
        base = get_repr_options()
        with repr_options(width=50):
            with repr_options(max_depth=2):
                self.assertEqual(get_repr_options().width, 50)
                self.assertEqual(get_repr_options().max_depth, 2)
            self.assertIsNone(get_repr_options().max_depth)
            with self.assertRaises(RuntimeError):
                with repr_options(width=10):
                    raise RuntimeError
            self.assertEqual(get_repr_options().width, 50)
        self.assertEqual(get_repr_options(), base)

    def test_set_repr_options_global_and_restore(self):
        previous = set_repr_options(width=33)
        try:
            self.assertEqual(get_repr_options().width, 33)
            with repr_options(width=44):
                self.assertEqual(get_repr_options().width, 44)
            self.assertEqual(get_repr_options().width, 33)
        finally:
            set_repr_options(**dataclasses.asdict(previous))
        self.assertEqual(get_repr_options(), previous)

    def test_context_is_thread_local(self):
        seen = {}
        entered = threading.Event()
        release = threading.Event()

        def other():
            entered.wait()
            seen['width'] = get_repr_options().width
            release.set()

        thread = threading.Thread(target=other)
        thread.start()
        with repr_options(width=21):
            entered.set()
            release.wait(5)
        thread.join(5)
        self.assertEqual(seen['width'], get_repr_options().width)
        self.assertNotEqual(seen['width'], 21)

    def test_render_depth_restored_after_error(self):
        class Boom(PrettyRepr):
            def __pretty_repr__(self):
                yield PrettyType('Boom')
                raise RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            repr(Boom())
        self.assertEqual(CONTEXT.render_depth, 0)


class TestColor(unittest.TestCase):

    def test_repr_never_colored(self):
        with repr_options(color=True):
            self.assertNotIn('\x1b[', repr(Obj(a=np.zeros(2))))

    def test_str_colored_when_enabled(self):
        obj = Obj(a=np.zeros(2, dtype=np.float32), b=[Obj('C')])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('NO_COLOR', None)
            with repr_options(color=True):
                text = str(obj)
        self.assertIn('\x1b[1;36mObj\x1b[0m', text)
        self.assertIn('\x1b[34ma\x1b[0m=', text)
        self.assertIn('\x1b[32mf32[2]\x1b[0m', text)

    def test_str_plain_by_default_and_with_no_color(self):
        obj = Obj(a=1)
        self.assertEqual(str(obj), repr(obj))
        with mock.patch.dict(os.environ, {'NO_COLOR': '1'}):
            with repr_options(color=True):
                self.assertEqual(str(obj), 'Obj(a=1)')

    def test_str_respects_overridden_repr(self):
        class Custom(Obj):
            def __repr__(self):
                return 'custom'

        with repr_options(color=True):
            self.assertEqual(str(Custom()), 'custom')

    def test_str_of_containers_uses_engine(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('NO_COLOR', None)
            with repr_options(color=True):
                self.assertIn('\x1b[', str(PrettyList([Obj('C')])))
                self.assertIn('\x1b[', str(PrettyDict({'a': Obj('C')})))

    def test_colored_layout_uses_plain_widths(self):
        obj = Obj(a=1, b=2)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('NO_COLOR', None)
            with repr_options(color=True, width=13):
                text = str(obj)
        self.assertNotIn('\n', text)  # escape codes do not count toward width


class TestSummary(unittest.TestCase):

    def test_root_footer(self):
        obj = Summarised(ReprSummary(26, 104), x=1)
        self.assertEqual(repr(obj), 'Summarised(x=1)  # 26 params (104 B)')
        with repr_options(summary=False):
            self.assertEqual(repr(obj), 'Summarised(x=1)')

    def test_footer_with_state_values_and_empty_summary(self):
        obj = Summarised(ReprSummary(1, 4, 2, 8))
        self.assertEqual(repr(obj), 'Summarised()  # 1 param (4 B), 2 state values (8 B)')
        self.assertEqual(repr(Summarised(ReprSummary())), 'Summarised()')

    def test_child_comment_only_when_parent_broken(self):
        obj = Obj(a=Summarised(ReprSummary(16, 64), w='w' * 20), b=Summarised(ReprSummary(0, 0, 3, 12)))
        self.assertNotIn('#', repr(obj))
        with repr_options(width=40):
            self.assertEqual(
                repr(obj),
                'Obj(\n  a=Summarised(w=' + 'w' * 20 + '),  # 16 params\n'
                '  b=Summarised(),  # 3 state values\n)',
            )

    def test_singular_wording(self):
        self.assertEqual(ReprSummary(1, 4).comment(), '1 param')
        self.assertEqual(ReprSummary(0, 0, 1, 4).footer(), '1 state value (4 B)')
        self.assertEqual(ReprSummary(1234, 4936).footer(), '1,234 params (4.8 KiB)')

    def test_failing_summary_is_ignored(self):
        class Broken(Obj):
            def __pretty_repr_summary__(self):
                raise RuntimeError

        self.assertEqual(repr(Broken('B')), 'B()')

    def test_format_bytes(self):
        self.assertEqual(format_bytes(0), '0 B')
        self.assertEqual(format_bytes(1023), '1023 B')
        self.assertEqual(format_bytes(1024), '1.0 KiB')
        self.assertEqual(format_bytes(5 * 1024 ** 2), '5.0 MiB')
        self.assertEqual(format_bytes(2048 * 1024 ** 3), '2048.0 GiB')


class TestFormatArrayLike(unittest.TestCase):

    def test_dtype_short_names(self):
        cases = {
            np.float16: 'f16', np.float32: 'f32', np.float64: 'f64',
            np.int8: 'i8', np.int32: 'i32', np.uint8: 'u8',
            np.complex64: 'c64', np.bool_: 'bool',
        }
        for dtype, name in cases.items():
            with self.subTest(dtype=dtype):
                self.assertEqual(format_array_like(np.zeros((2, 3), dtype=dtype)), f'{name}[2,3]')
        self.assertEqual(format_array_like(jnp.zeros(2, dtype=jnp.bfloat16)), 'bf16[2]')

    def test_long_dtype_names(self):
        self.assertEqual(format_array_like(np.zeros(3, np.float32), short_dtypes=False), 'float32[3]')
        with repr_options(short_dtypes=False):
            self.assertEqual(repr(Obj(a=np.zeros(3, np.int32))), 'Obj(a=int32[3])')

    def test_scalars_and_abstract_values(self):
        self.assertEqual(format_array_like(jnp.asarray(0.5, dtype=jnp.float32)), 'f32(0.5)')
        self.assertEqual(format_array_like(np.bool_(True)), 'bool(True)')
        self.assertEqual(format_array_like(jax.ShapeDtypeStruct((), jnp.float32)), 'f32[]')
        self.assertEqual(format_array_like(jax.ShapeDtypeStruct((4, 5), jnp.int32)), 'i32[4,5]')
        aval = jax.api_util.shaped_abstractify(jnp.zeros((2,)))
        self.assertEqual(format_array_like(aval), 'f32[2]')

    def test_tracers_are_summarised(self):
        captured = []

        @jax.jit
        def f(x):
            captured.append(format_array_like(x))
            return x

        f(jnp.zeros((2, 2)))
        self.assertEqual(captured, ['f32[2,2]'])

    def test_quantities(self):
        self.assertEqual(format_array_like(jnp.ones(3) * u.mV), 'f32[3] mV')
        self.assertEqual(format_array_like(u.Quantity(jnp.ones(2))), 'f32[2]')
        self.assertEqual(format_array_like(5. * u.ms), '5. ms')
        self.assertEqual(format_array_like(jnp.asarray(2.0) * u.mV), '2. mV')

    def test_typed_prng_keys(self):
        self.assertEqual(format_array_like(jax.random.key(0)), 'key<fry>[]')
        self.assertEqual(format_array_like(jax.random.split(jax.random.key(0), 3)), 'key<fry>[3]')

    def test_non_arrays(self):
        for value in (1, 1.5, 'x', [1], None, object()):
            with self.subTest(value=value):
                self.assertIsNone(format_array_like(value))


if __name__ == '__main__':
    unittest.main()
