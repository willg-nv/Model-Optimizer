#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""ONNX Q/DQ Autotuning Command-Line Interface.

This module provides a command-line interface for automated Q/DQ (Quantize/Dequantize)
optimization of ONNX models. It uses pattern-based region analysis and TensorRT performance
measurement to find optimal Q/DQ insertion points that minimize inference latency.

**Usage Examples:**

    # Basic usage - automatic region discovery and optimization
    python -m modelopt.onnx.quantization.autotune --model model.onnx

    # INT8 vs FP8 quantization
    python -m modelopt.onnx.quantization.autotune --model model.onnx --quant_type fp8

    # Warm-start from pattern cache (transfer learning)
    python -m modelopt.onnx.quantization.autotune \\
        --model model.onnx \\
        --pattern_cache ./output/pattern_cache.yaml

    # Import patterns from pre-quantized baseline model
    python -m modelopt.onnx.quantization.autotune \\
        --model model.onnx \\
        --qdq_baseline quantized_baseline.onnx

    # Full example with all optimization options
    python -m modelopt.onnx.quantization.autotune \\
        --model model.onnx \\
        --schemes_per_region 50 \\
        --pattern_cache pattern_cache.yaml \\
        --qdq_baseline baseline.onnx \\
        --output ./results \\
        --quant_type int8 \\
        --verbose

    # Use custom TensorRT plugins for model-specific operations
    python -m modelopt.onnx.quantization.autotune \\
        --model model.onnx \\
        --plugin_libraries /path/to/plugin1.so /path/to/plugin2.so

**Output Files:**

    output_dir/
    ├── autotuner_state.yaml          # Checkpoint for resume capability
    ├── baseline.onnx                 # Unquantized baseline model
    ├── optimized_final.onnx          # Final optimized model with Q/DQ
    ├── logs/                         # TensorRT build logs per scheme
    │   ├── baseline.log
    │   ├── region_*_scheme_*.log
    │   └── final.log
    └── region_models/                # Best model per region
        └── region_*_level_*.onnx
"""

import sys

from modelopt.onnx.quantization.autotune.cli import get_autotune_parser, run_autotune


def main():
    """Command-line entry point for ONNX Q/DQ autotuning.

    Parses command-line arguments and executes the autotuning workflow.

    Returns:
        Exit code from run_autotune (0 for success, non-zero for errors)
    """
    parser = get_autotune_parser()
    args = parser.parse_args()

    # Run autotuning
    return run_autotune(args)


if __name__ == "__main__":
    sys.exit(main())
