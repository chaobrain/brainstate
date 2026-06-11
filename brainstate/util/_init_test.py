"""Tests for the public ``brainstate.util`` package surface."""

import unittest


class TestUtilPackageExports(unittest.TestCase):
    """Validate the package-level export table."""

    def test_all_names_are_bound(self):
        """Every name listed in ``__all__`` is importable as a package attribute."""
        import brainstate.util as util

        for name in util.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(util, name), name)

    def test_star_import_includes_all_names(self):
        """Wildcard import should not fail due to a stale ``__all__`` entry."""
        namespace = {}

        exec("from brainstate.util import *", namespace)

        self.assertIn('BoundedCache', namespace)
        self.assertIn('breakpoint_if', namespace)


if __name__ == "__main__":
    unittest.main()
