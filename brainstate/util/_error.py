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


__all__ = [
    'BrainStateError',
    'TraceContextError',
]


class BrainStateError(Exception):
    """
    A custom exception class for BrainState-related errors.

    This exception is raised when a BrainState-specific error occurs during
    the execution of the program. It serves as a base class for more specific
    BrainState exceptions.

    Attributes:
        Inherits all attributes from the built-in Exception class.

    Usage::

        raise BrainStateError("A BrainState-specific error occurred.")
    """
    pass


class TraceContextError(BrainStateError):
    """
    A custom exception class for trace context-related errors in BrainState.

    This exception is raised when an error occurs specifically related to
    trace context operations or manipulations within the BrainState framework.

    Attributes:
        Inherits all attributes from the BrainStateError class.

    Usage::

        raise TraceContextError("An error occurred while handling trace context.")
    """
    pass
