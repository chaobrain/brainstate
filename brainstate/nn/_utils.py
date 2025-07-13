# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Union, Tuple

from brainstate._state import ParamState
from brainstate.util import PrettyTable
from ._module import Module

__all__ = [
    "count_parameters",
]


def _format_parameter_count(num_params, precision=2):
    if num_params < 1000:
        return str(num_params)

    suffixes = ['', 'K', 'M', 'B', 'T', 'P', 'E']
    magnitude = 0
    while abs(num_params) >= 1000:
        magnitude += 1
        num_params /= 1000.0

    format_string = '{:.' + str(precision) + 'f}{}'
    formatted_value = format_string.format(num_params, suffixes[magnitude])

    # 检查是否接近 1000，如果是，尝试使用更大的基数
    if magnitude < len(suffixes) - 1 and num_params >= 1000 * (1 - 10 ** (-precision)):
        magnitude += 1
        num_params /= 1000.0
        formatted_value = format_string.format(num_params, suffixes[magnitude])

    return formatted_value


def count_parameters(
    module: Module,
    precision: int = 2,
    return_table: bool = False,
) -> Union[Tuple[PrettyTable, int], int]:
    """
    Count and display the number of trainable parameters in a neural network model.

    This function iterates through all the parameters of the given model,
    counts the number of parameters for each module, and displays them in a table.
    It also calculates and returns the total number of trainable parameters.

    Parameters:
    -----------
    model : brainstate.nn.Module
        The neural network model for which to count parameters.

    Returns:
    --------
    int
        The total number of trainable parameters in the model.

    Prints:
    -------
    A pretty-formatted table showing the number of parameters for each module,
    followed by the total number of trainable parameters.
    """
    assert isinstance(module, Module), "Input must be a neural network module"  # noqa: E501
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in module.states(ParamState).items():
        param = parameter.numel()
        table.add_row([name, _format_parameter_count(param, precision=precision)])
        total_params += param
    table.add_row(["Total", _format_parameter_count(total_params, precision=precision)])
    print(table)
    if return_table:
        return table, total_params
    return total_params
