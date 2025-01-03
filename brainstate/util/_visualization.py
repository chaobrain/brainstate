# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

__all__ = [
    'display',
]

import importlib.util

treescope_installed = importlib.util.find_spec('treescope') is not None
try:
    from IPython import get_ipython

    in_ipython = get_ipython() is not None
except ImportError:
    in_ipython = False


def display(*args):
    """Display the given objects using the Treescope pretty-printer.

    If treescope is not installed or the code is not running in IPython,
    ``display`` will print the objects instead.
    """
    if not treescope_installed or not in_ipython:
        for x in args:
            print(x)
        return

    import treescope  # type: ignore[import-not-found,import-untyped]

    for x in args:
        treescope.display(x, ignore_exceptions=True, autovisualize=True)
