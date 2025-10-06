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


__all__ = [
    'BrainStateError',
    'BatchAxisError',
]


class BrainStateError(Exception):
    """
    A custom exception class for BrainState-related errors.

    This exception is raised when a BrainState-specific error occurs during
    the execution of the program. It serves as a base class for more specific
    BrainState exceptions.
    """
    pass


class BatchAxisError(BrainStateError):
    """
    Exception raised for errors related to batch axis operations.

    This custom exception is used to indicate errors that occur during
    batch processing or vectorization operations, particularly in the
    context of state management in the BrainState framework.

    Inherits from:
        BrainStateError: The base error class for BrainState-related exceptions.
    """
    __module__ = 'brainstate.transform'
