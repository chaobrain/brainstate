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


import warnings
from typing import Callable, Optional

import brainunit as u
import jax
import numpy as np

import brainstate as bst
from .gdiist_bpu import analyze_jaxpr_group_connection

__all__ = [
    'ForLoop',
    'JIT',
]

from brainstate.compile import ProgressBar

registered_devices = {
    'cpu': None,
    'gpu': None,
    'tpu': None,
    'bpu': None,  # Brain Processing Unit support
}


class ForLoop:
    """
    Enhanced ForLoop with BPU (Brain Processing Unit) analysis support.
    
    This class provides a for-loop execution interface that can analyze
    BrainState modules when device='bpu' is specified, utilizing JAXpr
    analysis for optimization insights.
    """

    def __init__(
        self,
        fn: Callable,
        device: str = 'cpu',
        enable_analysis: bool = True,
        analysis_kwargs: Optional[dict] = None
    ):
        """
        Initialize ForLoop with device-specific optimizations.
        
        Args:
            fn: The function to execute in the loop
            device: Target device ('cpu', 'gpu', 'tpu', 'bpu')
            enable_analysis: Whether to enable JAXpr analysis for BPU device
            analysis_kwargs: Additional arguments for JAXpr analysis
        """
        self.fn = fn
        self.device = device
        self.enable_analysis = enable_analysis
        self.analysis_kwargs = analysis_kwargs or {}

        # Initialize device-specific setup
        if device == 'bpu':
            self._setup_bpu()
        elif device in registered_devices:
            self._setup_standard_device()
        else:
            raise ValueError(f"Unsupported device: {device}. Supported devices: {list(registered_devices.keys())}")

    def _setup_bpu(self):
        """Setup BPU-specific analysis and execution."""
        try:
            self._analysis_enabled = self.enable_analysis
            print(f"BPU mode initialized with JAXpr analysis {'enabled' if self._analysis_enabled else 'disabled'}")
        except ImportError as e:
            warnings.warn(f"BPU analysis not available: {e}. Falling back to CPU mode.")
            self.device = 'cpu'
            self._analysis_enabled = False

    def _setup_standard_device(self):
        """Setup standard JAX device."""
        self._analysis_enabled = False
        print(f"{self.device.upper()} mode initialized")

    def _analyze_group_connection(self, *xs):
        """Perform BPU analysis on the computation."""
        if not self._analysis_enabled:
            return

        print("\n=== BPU JAXpr Analysis ===")

        try:
            # Use the analyze_function_jaxpr that encapsulates make_jaxpr + parsing
            # unpack the first element of each xs
            xs_unpacked = [x[0] if isinstance(x, (jax.Array, np.ndarray, u.Quantity)) else x for x in xs]

            result = analyze_jaxpr_group_connection(
                self.fn,
                *xs_unpacked,
                **self.analysis_kwargs
            )

            # Print analysis summary - only core BPU info
            print(f"BPU Analysis Results:")
            print(f"  Found {len(result.groups)} groups and {len(result.connections)} connections")
            print()
            
            # Print groups with better formatting
            print("Groups:")
            for i, group in enumerate(result.groups, 1):
                print(str(group))
            
            print()
            print("Connections:")
            for i, conn in enumerate(result.connections, 1):
                print(str(conn))
            
            print()
            print("  Ready for BPU compilation")

            return result

        except Exception as e:
            print(f"BPU analysis failed: {e}")
            return None

    def __call__(
        self,
        *xs,
        length: Optional[int] = None,
        reverse: bool = False,
        unroll: int | bool = 1,
        pbar: Optional[ProgressBar | int] = None
    ):
        """
        Execute the for-loop with device-specific optimizations.
        
        Args:
            times: Time array for the loop iteration
            *args, **kwargs: Additional arguments
            
        Returns:
            Results from the loop execution
        """
        if self.device == 'bpu':
            return self._call_bpu(
                *xs,
                length=length,
                reverse=reverse,
                unroll=unroll,
                pbar=pbar
            )
        else:
            return self._call_standard(
                *xs,
                length=length,
                reverse=reverse,
                unroll=unroll,
                pbar=pbar
            )

    def _call_bpu(
        self,*xs,**kwargs
    ):
        """BPU-specific execution with analysis."""
        # Perform analysis
        analysis_result = self._analyze_group_connection(*xs)

        print(f"\n=== BPU Execution (Analysis Mode) ===")
        print(f"Note: Actual BPU hardware execution not available.")
        print(f"TODO: Implement BPU hardware interface")

        # For now, fall back to CPU execution with analysis insights
        print(f"Falling back to CPU execution with BPU optimizations...")

        # TODO: Here would be the actual BPU hardware execution
        # For now, we simulate with standard JAX execution
        return self._simulate_bpu_execution(
            *xs,
            **kwargs
        )

    def _simulate_bpu_execution(self, *xs, **kwargs):
        """Simulate BPU execution using standard JAX with optimization hints."""
        print(f"Simulating BPU execution...")

        # Use standard for_loop as fallback
        results = bst.compile.for_loop(
            self.fn,
            *xs,
            **kwargs
        )
        print(f"BPU simulation completed successfully!")
        return results

    def _call_standard(self, *xs, **kwargs):
        """Standard device execution."""
        results = bst.compile.for_loop(
            self.fn,
            *xs,
            **kwargs
        )
        return results


class JIT:
    def __init__(
        self,
        fn: Callable,
        device: str,
    ):
        self.fn = fn
        self.device = device

        if device not in registered_devices:
            raise ValueError(f"Device '{device}' is not registered.")
        self.device = device

    def __call__(self, *args, **kwargs):
        pass
