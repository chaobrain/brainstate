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

# Import tracer utilities
from ._tracers import (
    StateJaxTracer,
)

# Import cache utilities
from ._cache import (
    BoundedCache,
)

# Import pretty representation utilities
from ._pretty_repr import (
    yield_unique_pretty_repr_items,
    PrettyType,
    PrettyAttr,
    PrettyRepr,
    PrettyMapping,
    MappingReprMixin,
)

# Import other utilities
from ._others import (
    split_total,
    clear_buffer_memory,
    not_instance_eval,
    is_instance_eval,
    DictManager,
    DotDict,
    get_unique_name,
    merge_dicts,
    flatten_dict,
    unflatten_dict,
)

# Import filter utilities
from .filter import (
    to_predicate,
    WithTag,
    PathContains,
    OfType,
    Any,
    All,
    Nothing,
    Not,
    Everything,
)

# Import struct utilities
from .struct import (
    field,
    is_dataclass,
    dataclass,
    PyTreeNode,
    FrozenDict,
    freeze,
    unfreeze,
    copy,
    pop,
    pretty_repr,
)

# Import pretty pytree utilities
from ._pretty_pytree import (
    PrettyDict,
    NestedDict,
    FlattedDict,
    flat_mapping,
    nest_mapping,
    PrettyList,
    PrettyObject,
)

__all__ = [
    # Tracer utilities
    'StateJaxTracer',

    # Cache utilities
    'BoundedCache',

    # Pretty representation utilities
    'yield_unique_pretty_repr_items',
    'PrettyType',
    'PrettyAttr',
    'PrettyRepr',
    'PrettyMapping',
    'MappingReprMixin',

    # Other utilities
    'split_total',
    'clear_buffer_memory',
    'not_instance_eval',
    'is_instance_eval',
    'DictManager',
    'DotDict',
    'get_unique_name',
    'merge_dicts',
    'flatten_dict',
    'unflatten_dict',

    # Filter utilities
    'to_predicate',
    'WithTag',
    'PathContains',
    'OfType',
    'Any',
    'All',
    'Nothing',
    'Not',
    'Everything',

    # Struct utilities
    'field',
    'is_dataclass',
    'dataclass',
    'PyTreeNode',
    'FrozenDict',
    'freeze',
    'unfreeze',
    'copy',
    'pop',
    'pretty_repr',

    # Pretty pytree utilities
    'PrettyDict',
    'NestedDict',
    'FlattedDict',
    'flat_mapping',
    'nest_mapping',
    'PrettyList',
    'PrettyObject',
]

