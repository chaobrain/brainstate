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

"""Pytest configuration shared across the brainstate test suite.

Provides:

* a ``--run-slow`` command-line flag; tests marked ``@pytest.mark.slow`` are
  skipped unless it is supplied (keeps the default CI run fast),
* an autouse fixture that seeds ``brainstate.random`` before every test so
  randomized tests are deterministic and isolated from one another.
"""

import pytest

import brainstate


def pytest_addoption(parser):
    """Register the ``--run-slow`` flag."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``slow``-marked tests unless ``--run-slow`` is given."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed brainstate.random before each test for reproducibility."""
    brainstate.random.seed(0)
    yield
