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

from typing import Optional, Callable, Union

from brainstate import init
from brainstate._state import ParamState
from brainstate.typing import ArrayLike
from ._module import Module

__all__ = [
    'Embedding',
]


class Embedding(Module):
    r"""
    A simple lookup table that stores embeddings of a fixed size.

    Args:
      num_embeddings: Size of embedding dictionary. Must be non-negative.
      embedding_size: Size of each embedding vector. Must be non-negative.
      embedding_init: The initializer for the embedding lookup table, of shape `(num_embeddings, embedding_size)`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        embedding_init: Union[Callable, ArrayLike] = init.LecunUniform(),
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if num_embeddings < 0:
            raise ValueError("num_embeddings must not be negative.")
        if embedding_size < 0:
            raise ValueError("embedding_size must not be negative.")
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.out_size = (embedding_size,)

        weight = init.param(embedding_init, (self.num_embeddings, self.embedding_size))
        self.weight = ParamState(weight)

    def update(self, indices: ArrayLike):
        return self.weight.value[indices]
