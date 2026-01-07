# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ONNX Q/DQ Autotuning Workflows.

SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

This module provides high-level workflow functions for automated Q/DQ (Quantization/Dequantization)
optimization of ONNX models using pattern-based region analysis and TensorRT performance measurement.

**Core Capabilities:**

1. **Automated Region Discovery**: Discovers hierarchical regions in the computation graph
   - LEAF regions: Contain actual graph nodes
   - COMPOSITE regions: Contain child regions with hierarchical structure

2. **Pattern-Based Optimization**: Groups regions by structural pattern
   - Regions with identical patterns share optimization schemes
   - One optimization applies to all matching regions simultaneously

3. **TensorRT Benchmarking**: Measures actual inference performance
   - Builds TensorRT engines for each Q/DQ configuration
   - Measures median latency across multiple runs
   - Caches timing data for faster iteration

4. **Incremental State Management**: Supports crash recovery and resume
   - Saves state after each region profiling
   - Resumes from last checkpoint automatically
   - Preserves baseline and all measurements

5. **Pattern Cache Warm-Start**: Leverages previous optimization results
   - Loads known-good schemes from cache
   - Reduces exploration time for similar models
   - Transfers learned patterns across runs

**Key Functions:**

- **benchmark_onnx_model()**: Benchmark ONNX model inference latency using TensorRT
- **init_benchmark_instance()**: Initialize global TensorRT benchmark instance
- **region_pattern_autotuning_workflow()**: Complete end-to-end Q/DQ optimization workflow

**Workflow Overview:**

1. Initialize autotuner with automatic region discovery
2. Measure baseline performance (no Q/DQ)
3. For each region pattern:
   - Generate Q/DQ insertion schemes
   - Benchmark each scheme with TensorRT
   - Select best scheme for pattern
   - Apply to all regions with matching pattern
4. Export final optimized model

**Performance Optimization:**

- Pattern-based approach reduces redundant evaluation
- TensorRT timing cache speeds up engine builds
- Incremental state saves enable long-running optimizations
- Pattern cache enables cross-model learning
"""

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path

import onnx

from modelopt.onnx.quantization.autotune.autotuner import QDQAutotuner
from modelopt.onnx.quantization.autotune.benchmark import TensorRTPyBenchmark, TrtExecBenchmark
from modelopt.onnx.quantization.autotune.common import Config, PatternCache
from modelopt.onnx.quantization.qdq_utils import get_quantized_tensors

logger = logging.getLogger(__name__)

# Global benchmark instance - will be initialized with timing cache
_benchmark_instance = None


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_onnx_model(
    model_path: str | bytes, log_file: str | None = None, flush_timing_cache: bool = False
) -> float:
    """Benchmark ONNX model inference latency using TensorRT Python API.

    Uses the global TensorRTPyBenchmark instance to build a TensorRT engine
    and measure inference latency. The benchmark instance persists across calls
    for efficiency (reuses Builder, Runtime, Logger, and timing cache).

    **Process:**
    1. Loads ONNX model (from file path or bytes)
    2. Builds optimized TensorRT engine (uses timing cache for speed)
    3. Runs warmup iterations to stabilize performance
    4. Measures latency across multiple timing iterations
    5. Returns median latency

    Args:
        model_path: Path to ONNX model file, or bytes containing serialized model protobuf
        log_file: Optional path to save detailed TensorRT build and benchmark logs
                 (default: None, no logging)
        flush_timing_cache: If True, flushes TensorRT timing cache before building engine.
                           Useful for periodic cache refresh (default: False)

    Returns:
        Measured median inference latency in milliseconds.
        Returns float('inf') on failure (invalid model, build error, etc.)

    Raises:
        No exceptions raised - errors are caught and logged, returning float('inf')

    Note:
        Requires _benchmark_instance to be initialized via init_benchmark_instance()
        before calling this function. Otherwise returns float('inf').

    Example:
        >>> init_benchmark_instance("timing.cache", warmup_runs=5, timing_runs=20)
        >>> latency = benchmark_onnx_model("model.onnx", log_file="build.log")
        >>> print(f"Latency: {latency:.2f} ms")
    """
    global _benchmark_instance

    if _benchmark_instance is None:
        logger.error("Benchmark instance not initialized")
        return float("inf")

    try:
        # Run TensorRT benchmark
        latency = _benchmark_instance.run(
            model_path, log_file=log_file, flush_timing_cache=flush_timing_cache
        )

        if latency == float("inf"):
            if isinstance(model_path, bytes):
                logger.warning("Benchmark failed for model bytes")
            else:
                logger.warning(f"Benchmark failed: {model_path}")
            return float("inf")

        logger.debug(f"Benchmark result: {latency:.2f} ms")
        return latency

    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        return float("inf")


def init_benchmark_instance(
    use_trtexec: bool = False,
    plugin_libraries: list[str] | None = None,
    timing_cache_file: str | None = None,
    warmup_runs: int = 10,
    timing_runs: int = 20,
):
    """Initialize global TensorRT benchmark instance for model performance measurement.

    Creates and configures a TensorRTPyBenchmark instance that persists across
    multiple benchmark_onnx_model() calls for efficiency. The instance reuses
    TensorRT Builder, Runtime, Logger, and timing cache.

    **Benefits of Persistent Instance:**
    - Avoids repeated initialization overhead
    - Reuses timing cache across multiple models
    - Maintains consistent benchmark configuration


    Args:
        use_trtexec: Whether to use trtexec for benchmarking.
        plugin_libraries: List of paths to TensorRT plugin shared libraries (.so files).
                          These plugins will be loaded by trtexec or TensorRT Python API during engine building.
                          If None, no custom plugins are loaded.
        timing_cache_file: Path to TensorRT timing cache file for faster engine builds.
                          If None, uses default "trtexec_timing.cache" (default: None)
        warmup_runs: Number of warmup inference iterations before measurement.
                    Allows GPU to reach stable performance state (default: 5)
        timing_runs: Number of timed inference iterations for latency measurement.
                    Higher values give more stable median (default: 20)

    Returns:
        TensorRTPyBenchmark or TrtExecBenchmark instance if initialization succeeds, None on failure

    Example:
        >>> # Initialize with default settings using TensorRT Python API
        >>> benchmark = init_benchmark_instance(use_trtexec=False)
        >>> if benchmark:
        ...     latency = benchmark_onnx_model("model.onnx")
        ...     print(f"Latency: {latency:.2f} ms")

    See Also:
        benchmark_onnx_model(): Uses the initialized instance to benchmark models
    """
    global _benchmark_instance
    try:
        if use_trtexec:
            _benchmark_instance = TrtExecBenchmark(
                timing_cache_file=timing_cache_file,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
                plugin_libraries=plugin_libraries,
            )
            logger.info("Trtexec benchmark initialized")
        else:
            _benchmark_instance = TensorRTPyBenchmark(
                timing_cache_file=timing_cache_file,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
                plugin_libraries=plugin_libraries,
            )
            logger.info("TensorRT Python API benchmark initialized")
        logger.debug(
            f"Settings: warmup={warmup_runs}, timing={timing_runs}, "
            f"cache={timing_cache_file or 'trtexec_timing.cache'}, plugin_libraries={plugin_libraries}"
        )
        return _benchmark_instance
    except Exception as e:
        logger.error(f"TensorRT initialization failed: {e}", exc_info=True)
        return None


def _region_matches_filter(region, graph, filter_patterns: list[str]) -> bool:
    """Check if any node in the region matches any of the filter patterns.

    Args:
        region: Region object to check
        graph: ONNX graph (graphsurgeon) containing node information
        filter_patterns: List of wildcard patterns to match against node names

    Returns:
        True if at least one node in the region matches any pattern, False otherwise
    """
    if not filter_patterns:
        return True  # No filter means all regions pass

    # Get all node indices in this region (including children)
    node_indices = region.get_all_nodes_recursive()

    for node_idx in node_indices:
        if node_idx < len(graph.nodes):
            node_name = graph.nodes[node_idx].name
            for pattern in filter_patterns:
                if fnmatch.fnmatch(node_name, pattern):
                    return True

    return False


# =============================================================================
# Autotuning Workflow
# =============================================================================


def region_pattern_autotuning_workflow(
    model_path: str,
    output_dir: Path,
    num_schemes_per_region: int = 50,
    pattern_cache_file: str | None = None,
    state_file: str | None = None,
    quant_type: str = "int8",
    default_dq_dtype: str = "float32",
    qdq_baseline_model: str | None = None,
    node_filter_list: list[str] | None = None,
    verbose: bool = False,
) -> QDQAutotuner:
    """Run automated Q/DQ (Quantization/Dequantization) optimization on an ONNX model.

    This workflow uses pattern-based region optimization to efficiently find optimal
    Q/DQ insertion points. The key insight: regions with identical structural patterns
    can share the same Q/DQ scheme. When a best scheme is found for a pattern, it
    automatically applies to all regions matching that pattern, making optimization
    both efficient and consistent.

    Automatically discovers regions, generates and tests Q/DQ insertion schemes,
    and exports optimized model. Supports incremental state saving for crash recovery
    and pattern cache-based warm-start.

    **Workflow Steps:**
    1. Load model and initialize autotuner with automatic hierarchical region discovery
    2. Resume from checkpoint if state file exists (crash recovery)
    3. Load pattern cache if provided (warm-start with known-good schemes)
    4. Import Q/DQ patterns from baseline model if provided (transfer learning)
    5. Measure baseline performance without Q/DQ insertions
    6. For each discovered region pattern:
       a. Generate Q/DQ insertion schemes (pattern-relative)
       b. Build TensorRT engine and measure latency for each scheme
       c. Select best scheme for this pattern (applies to all matching regions)
       d. Save checkpoint and intermediate model
    7. Export final optimized model with best Q/DQ scheme for each pattern

    **State Management (Crash Recovery):**
    - Automatically saves checkpoint after profiling each region
    - Resume from interruption by running same command (auto-detects state file)
    - State file contains:
      * Baseline latency measurement
      * All profiled pattern schemes and their latencies
      * Best scheme selection for each pattern
      * Region discovery results and pattern assignments
    - Enables long-running optimizations with fault tolerance
    - Safe for cluster environments with preemption

    **Pattern Cache (Warm-Start Optimization):**
    - Pattern cache files (YAML format) contain top-performing schemes indexed by pattern
    - Stores results from previous optimization runs for reuse
    - Used to seed scheme generation (warm-start vs cold-start)
    - Benefits:
      * Reduces exploration time by prioritizing known-good configurations
      * Transfers learned schemes across similar models or model versions
      * Accumulates knowledge across multiple optimization sessions
      * Particularly effective for models with recurring pattern structures
    - Cache is pattern-specific, not model-specific (generalizes well)

    **QDQ Baseline Model (Transfer Learning):**
    - If provided, extracts Q/DQ insertion points from a pre-quantized model
    - Identifies which tensors are quantized in the baseline model
    - Maps these quantization points to region patterns in the current model
    - Updates pattern cache with learned insertion strategies
    - Enables warm-start from:
      * Expert-tuned quantized models (manually optimized)
      * Previous autotuning runs (transfer across model versions)
      * Reference implementations (e.g., from framework exporters)

    Args:
        model_path: Path to ONNX model file to optimize
        output_dir: Directory for output files (state, logs, models). Created if doesn't exist.
        num_schemes_per_region: Number of Q/DQ insertion schemes to test per region pattern.
                               Higher values explore more configurations but take longer (default: 50)
        pattern_cache_file: Optional path to pattern cache YAML file containing known-good schemes
                           from previous runs. Enables warm-start optimization (default: None)
        state_file: Optional path to state file for checkpoint/resume. If None, automatically
                   uses <output_dir>/autotuner_state.yaml (default: None)
        quant_type: Quantization data type - "int8" for INT8 quantization (default),
                   "fp8" for FP8 quantization
        qdq_baseline_model: Optional path to a pre-quantized ONNX model. If provided,
                           extracts Q/DQ insertion patterns and adds them to pattern cache
                           for warm-start (default: None)

    Returns:
        Configured QDQAutotuner instance containing:
        - All discovered regions and their patterns
        - Profiled Q/DQ insertion schemes for each pattern
        - Best scheme selections and performance measurements
        - Complete optimization state (can be saved/loaded)

        The returned autotuner can be used for:
        - Exporting optimized models with best Q/DQ schemes
        - Analyzing per-pattern optimization results
        - Further refinement or experimentation
        - Pattern cache generation for future runs

    Example:
        >>> # Initial run
        >>> autotuner = region_pattern_autotuning_workflow("model.onnx", Path("./output"))
        >>> # Resume from interruption
        >>> autotuner = region_pattern_autotuning_workflow("model.onnx", Path("./output"))
        >>> # With pattern cache warm-start
        >>> autotuner = region_pattern_autotuning_workflow(
        ...     "model.onnx", Path("./output"), pattern_cache_file="./pattern_cache.yaml"
        ... )
        >>> # With QDQ baseline model for pattern import
        >>> autotuner = region_pattern_autotuning_workflow(
        ...     "model.onnx", Path("./output"), qdq_baseline_model="quantized_baseline.onnx"
        ... )
    """
    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "region_models"
    models_dir.mkdir(exist_ok=True)

    # Determine state file path (auto-resume if exists)
    if state_file is None:
        state_file = str(output_dir / "autotuner_state.yaml")
        logger.debug(f"Using default state file: {state_file}")
    state_path = Path(state_file)

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = onnx.load(model_path)

    # Load pattern cache if provided
    pattern_cache = None
    if pattern_cache_file:
        pattern_cache_path = Path(pattern_cache_file)
        if pattern_cache_path.exists():
            pattern_cache = PatternCache.load(str(pattern_cache_path))
            logger.info(
                f"Loaded pattern cache: {pattern_cache.num_patterns} patterns, "
                f"{pattern_cache.total_schemes} schemes"
            )
        else:
            logger.warning(f"Pattern cache not found: {pattern_cache_file}")

    # Initialize autotuner with config
    logger.info(
        f"Initializing autotuner (quant_type={quant_type}, default_dq_dtype={default_dq_dtype})"
    )
    config = Config(
        default_quant_type=quant_type,
        default_dq_dtype=default_dq_dtype,
        performance_threshold=1.01,  # Accept ≥1% improvement
        verbose=verbose,
    )

    autotuner = QDQAutotuner(model)
    autotuner.initialize(config, pattern_cache)

    # Auto-resume: load previous state if exists
    if state_path.exists():
        logger.info(f"Found existing state file, resuming from checkpoint: {state_path}")
        autotuner.load_state(str(state_path))
    else:
        logger.info("No existing state file found, starting new autotuning session")

    # Import Q/DQ insertion points from baseline model if provided
    if qdq_baseline_model:
        qdq_baseline_path = Path(qdq_baseline_model)
        if qdq_baseline_path.exists():
            logger.info(f"Importing Q/DQ insertion points from: {qdq_baseline_model}")
            qdq_model = onnx.load(str(qdq_baseline_path))

            # Extract quantized tensors from baseline model
            quantized_tensors = get_quantized_tensors(qdq_model)
            logger.info(f"Found {len(quantized_tensors)} quantized tensors in PTQ model")

            # Import insertion points into autotuner for warm-start
            autotuner.import_insertion_points(quantized_tensors)
            logger.info("Q/DQ insertion points imported successfully")
        else:
            logger.warning(f"QDQ baseline model not found: {qdq_baseline_model}")

    # Get discovered regions
    regions = autotuner.regions
    logger.info(f"Ready to profile {len(regions)} regions")

    # Measure baseline (no Q/DQ) if not already measured
    if autotuner.baseline_latency_ms is None:
        logger.info("Measuring baseline (no Q/DQ)")
        baseline_path = output_dir / "baseline.onnx"
        autotuner.export_onnx(str(baseline_path), insert_qdq=False)
        baseline_log = logs_dir / "baseline.log"
        baseline_latency = benchmark_onnx_model(str(baseline_path), str(baseline_log))
        autotuner.submit(baseline_latency)
        logger.info(f"Baseline: {baseline_latency:.2f} ms")
    else:
        baseline_latency = autotuner.baseline_latency_ms
        logger.info(f"Using baseline from checkpoint: {baseline_latency:.2f} ms")

    # Profile regions
    logger.info(f"Starting region profiling ({num_schemes_per_region} schemes per region)")

    iteration_count = 0

    for region_idx, region in enumerate(regions):
        logger.info(
            f"Region {region_idx + 1}/{len(regions)} (ID={region.id}, level={region.get_level()})"
        )

        # Check if region matches node filter list
        if node_filter_list and not _region_matches_filter(
            region, autotuner.graph, node_filter_list
        ):
            logger.info("  Skipping (no nodes match filter patterns)")
            continue

        # Set as current profile region
        # Commit previous region's results (except for first region)
        commit = region_idx > 0
        autotuner.set_profile_region(region, commit=commit)

        # Check if already profiled (from loaded state)
        if autotuner.current_profile_pattern_schemes is None:
            logger.info("  Skipping (already profiled)")
            continue

        # Generate and test schemes for this region
        schemes_tested = 0
        for scheme_num in range(num_schemes_per_region):
            iteration_count += 1

            # Generate new scheme
            scheme_idx = autotuner.generate()

            if scheme_idx == -1:
                logger.debug(f"  Stopping at scheme {scheme_num + 1} (no more unique schemes)")
                break

            schemes_tested += 1

            # Export model with this scheme + best from previous regions
            model_bytes = autotuner.export_onnx(None, insert_qdq=True)

            # Benchmark with TensorRT
            test_log = logs_dir / f"region_{region.id}_scheme_{scheme_idx}.log"
            flush_timing_cache = (iteration_count % 10) == 0
            latency = benchmark_onnx_model(
                model_bytes, str(test_log), flush_timing_cache=flush_timing_cache
            )

            # Record result
            autotuner.submit(latency, success=(latency != float("inf")))

        # Display region summary
        ps = autotuner.current_profile_pattern_schemes
        if ps and ps.schemes:
            best_scheme = ps.best_scheme
            if best_scheme and best_scheme.latency_ms < float("inf") and baseline_latency > 0:
                speedup = baseline_latency / best_scheme.latency_ms
                status = "✓" if speedup >= autotuner.config.performance_threshold else "·"
                logger.info(
                    f"  {status} Tested {schemes_tested} schemes: "
                    f"best {best_scheme.latency_ms:.2f} ms ({speedup:.3f}x speedup)"
                )
            else:
                logger.info(f"  Tested {schemes_tested} schemes: no valid measurements")
        else:
            logger.info(f"  Tested {schemes_tested} schemes")

        # Save best model for this region (before committing to next region)
        region_model_path = models_dir / f"region_{region.id}_level_{region.get_level()}.onnx"
        autotuner.export_onnx(str(region_model_path), insert_qdq=True, best=True)
        logger.debug(f"  Saved best model: {region_model_path.name}")

        # Save state after each region (incremental, crash recovery)
        autotuner.save_state(str(state_path))
        logger.debug("  Checkpoint saved")

    # Commit final region
    autotuner.set_profile_region(None, commit=True)

    # Export and measure final optimized model
    logger.info("Exporting final optimized model")
    final_model_path = output_dir / "optimized_final.onnx"
    autotuner.export_onnx(str(final_model_path), insert_qdq=True)
    final_log = logs_dir / "final.log"
    final_latency = benchmark_onnx_model(str(final_model_path), str(final_log))

    # Display results
    if final_latency > 0 and final_latency != float("inf"):
        speedup = baseline_latency / final_latency
        logger.info(
            f"Results: {baseline_latency:.2f} ms → {final_latency:.2f} ms ({speedup:.3f}x speedup)"
        )
    else:
        logger.info(f"Results: {baseline_latency:.2f} ms → failed (invalid measurement)")

    # Save final state
    autotuner.save_state(str(state_path))

    logger.info("Autotuning complete")
    logger.info(f"  Final model: {final_model_path}")
    logger.info(f"  State: {state_path}")
    logger.debug(f"  Logs: {logs_dir}")
    logger.debug(f"  Region models: {models_dir}")

    return autotuner


# =============================================================================
# Quantize Integration Workflow
# =============================================================================


# Autotune mode presets with hyperparameters
# Each mode varies num_schemes_per_region and timing_runs
# warmup_runs is fixed at 10 across all modes
# Note: "none" mode is not included as autotune is disabled in that case
AUTOTUNE_MODE_PRESETS: dict[str, dict[str, int]] = {
    "fast": {
        "num_schemes_per_region": 20,
        "warmup_runs": 10,
        "timing_runs": 10,
    },
    "default": {
        "num_schemes_per_region": 50,
        "warmup_runs": 10,
        "timing_runs": 20,
    },
    "best": {
        "num_schemes_per_region": 100,
        "warmup_runs": 10,
        "timing_runs": 50,
    },
    "extreme": {
        "num_schemes_per_region": 200,
        "warmup_runs": 10,
        "timing_runs": 100,
    },
}


@dataclass
class QDQAutotunerWorkflow:
    """Autotuner workflow for quantize() integration.

    Collects relevant arguments from quantize() and provides the `run()` method
    to execute autotuning. This is distinct from the `Config` class in `common.py`
    which controls autotuner internal behavior, and `QDQAutotuner` which is the
    core autotuning engine.

    **Autotune Modes:**
    - 'none': Disabled, no autotuning
    - 'fast': Quick search (20 schemes/region, 10 timing runs)
    - 'default': Balanced search (50 schemes/region, 20 timing runs)
    - 'best': Thorough search (100 schemes/region, 50 timing runs)
    - 'extreme': Exhaustive search (200 schemes/region, 100 timing runs)

    **Execution Modes:**
    - autotune != 'none' and not direct_autotune: Run autotune after PTQ
    - direct_autotune=True: Skip PTQ and run autotune directly (uses autotune mode)

    Attributes:
        # Core paths
        input_model_path: Path to input ONNX model (high precision)
        output_path: Path to save final optimized model
        output_dir: Directory for autotune working files (state, logs, models)

        # Autotune mode
        autotune: Autotune mode ('none', 'fast', 'default', 'best', 'extreme')
        direct_autotune: If True, skip PTQ and run autotune directly

        # Quantization settings (from quantize())
        quantize_mode: Quantization mode ('int8', 'fp8', 'int4')
        high_precision_dtype: High precision dtype ('fp32', 'fp16', 'bf16')
        trt_plugins: List of TensorRT plugin library paths

        # Autotune-specific settings (derived from mode or overridden)
        num_schemes_per_region: Number of schemes to test per region
        pattern_cache_file: Path to pattern cache for warm-start
        state_file: Path to state file for checkpoint/resume
        warmup_runs: TensorRT benchmark warmup iterations
        timing_runs: TensorRT benchmark timing iterations
        performance_threshold: Minimum speedup ratio to accept a scheme
        use_trtexec: Use trtexec instead of TensorRT Python API (default: False)
    """

    # Core paths
    input_model_path: str
    output_path: str
    output_dir: str | None = None

    # Autotune mode
    autotune: str = "none"  # 'none', 'fast', 'default', 'best', 'extreme'
    direct_autotune: bool = False  # If True, skip PTQ and run autotune directly

    # Quantization settings
    quantize_mode: str = "int8"
    high_precision_dtype: str = "fp16"
    trt_plugins: list[str] | None = None

    # Autotune-specific settings (can be overridden or derived from mode)
    num_schemes_per_region: int | None = None
    pattern_cache_file: str | None = None
    state_file: str | None = None
    warmup_runs: int | None = None
    timing_runs: int | None = None
    performance_threshold: float | None = None
    use_trtexec: bool = False
    node_filter_list: list[str] | None = None  # Wildcard patterns to filter nodes
    verbose: bool = False  # Enable verbose DEBUG logging

    def __post_init__(self):
        """Apply mode presets for any unset hyperparameters."""
        mode = self.autotune
        if mode in AUTOTUNE_MODE_PRESETS:
            preset = AUTOTUNE_MODE_PRESETS[mode]
            if self.num_schemes_per_region is None:
                self.num_schemes_per_region = preset["num_schemes_per_region"]
            if self.warmup_runs is None:
                self.warmup_runs = preset["warmup_runs"]
            if self.timing_runs is None:
                self.timing_runs = preset["timing_runs"]
        else:
            # Fallback to default preset values
            if self.num_schemes_per_region is None:
                self.num_schemes_per_region = 50
            if self.warmup_runs is None:
                self.warmup_runs = 10
            if self.timing_runs is None:
                self.timing_runs = 20
        # performance_threshold is fixed at 1.02
        if self.performance_threshold is None:
            self.performance_threshold = 1.02

    @classmethod
    def from_quantize_args(
        cls,
        onnx_path: str,
        output_path: str | None = None,
        quantize_mode: str = "int8",
        high_precision_dtype: str = "fp16",
        trt_plugins: list[str] | None = None,
        autotune: str = "none",
        direct_autotune: bool = False,
        log_level: str = "INFO",
        **kwargs,
    ) -> "QDQAutotunerWorkflow":
        """Create QDQAutotunerWorkflow from quantize() arguments.

        Args:
            onnx_path: Path to input ONNX model
            output_path: Path to save quantized model
            quantize_mode: Quantization mode ('int8', 'fp8', 'int4')
            high_precision_dtype: High precision dtype
            trt_plugins: TensorRT plugin library paths
            autotune: Autotune mode ('none', 'fast', 'default', 'best', 'extreme')
            direct_autotune: If True, skip PTQ and run autotune directly
            log_level: Log level ('DEBUG', 'INFO', etc.). 'DEBUG' enables verbose autotuning.
            **kwargs: Additional autotune-specific arguments for overriding presets:
                - autotune_num_schemes: Override num_schemes_per_region
                - autotune_warmup_runs: Override warmup_runs
                - autotune_timing_runs: Override timing_runs
                - autotune_performance_threshold: Override performance_threshold
                - autotune_pattern_cache: Path to pattern cache file
                - autotune_state_file: Path to state file
                - autotune_use_trtexec: Use trtexec instead of Python API
                - autotune_node_filter: List of wildcard patterns to filter nodes

        Returns:
            Configured QDQAutotunerWorkflow instance with mode-appropriate hyperparameters
        """
        # Determine output directory
        if output_path:
            output_dir = str(Path(output_path).parent / "autotune")
        else:
            output_dir = str(Path(onnx_path).parent / "autotune")

        # Enable verbose mode if log_level is DEBUG
        verbose = log_level.upper() == "DEBUG"

        return cls(
            input_model_path=onnx_path,
            output_path=output_path or onnx_path.replace(".onnx", ".quant.onnx"),
            output_dir=output_dir,
            autotune=autotune,
            direct_autotune=direct_autotune,
            quantize_mode=quantize_mode,
            high_precision_dtype=high_precision_dtype,
            trt_plugins=trt_plugins,
            # These can override mode presets if provided
            num_schemes_per_region=kwargs.get("autotune_num_schemes"),
            pattern_cache_file=kwargs.get("autotune_pattern_cache"),
            state_file=kwargs.get("autotune_state_file"),
            warmup_runs=kwargs.get("autotune_warmup_runs"),
            timing_runs=kwargs.get("autotune_timing_runs"),
            performance_threshold=kwargs.get("autotune_performance_threshold"),
            use_trtexec=kwargs.get("autotune_use_trtexec", False),
            node_filter_list=kwargs.get("autotune_node_filter"),
            verbose=verbose,
        )

    @property
    def should_run_autotune(self) -> bool:
        """Check if autotune should be executed."""
        return self.autotune != "none"

    @property
    def quant_type(self) -> str:
        """Get quantization type for autotune (int8 or fp8)."""
        if self.quantize_mode in ["int8", "fp8"]:
            return self.quantize_mode
        # Default to int8 for other modes like int4
        return "int8"

    @property
    def default_dq_dtype(self) -> str:
        """Get default DQ output dtype based on high precision dtype."""
        dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        return dtype_map.get(self.high_precision_dtype, "float32")

    def run(self, quantized_model_path: str | None = None) -> str | None:
        """Run autotune workflow.

        This method integrates autotuning into the quantize() workflow. It supports
        two execution modes controlled by the autotune/direct_autotune settings:

        **Autotune Modes ('none', 'fast', 'default', 'best', 'extreme'):**
        - 'none': Disabled
        - 'fast': Quick search (20 schemes/region, 10 timing runs)
        - 'default': Balanced search (50 schemes/region, 20 timing runs)
        - 'best': Thorough search (100 schemes/region, 50 timing runs)
        - 'extreme': Exhaustive search (200 schemes/region, 100 timing runs)

        **Execution Mode 1: Post-PTQ Autotune (autotune != 'none', direct_autotune=False)**
        - Requires quantized_model_path from PTQ
        - Uses quantized model as baseline for pattern import
        - Optimizes Q/DQ placement starting from PTQ result

        **Execution Mode 2: Direct Autotune (direct_autotune=True)**
        - Runs autotune directly on high precision input model
        - Skips PTQ step entirely
        - Discovers optimal Q/DQ placement from scratch

        Args:
            quantized_model_path: Path to PTQ-quantized model (required if not direct_autotune)

        Returns:
            Path to the final optimized model, or None if autotune was skipped/failed

        Example:
            >>> # Post-PTQ autotune with default mode
            >>> workflow = QDQAutotunerWorkflow.from_quantize_args(
            ...     onnx_path="model.onnx", autotune="default", quantize_mode="int8"
            ... )
            >>> optimized_path = workflow.run(quantized_model_path="model.quant.onnx")

            >>> # Direct autotune with fast mode (skip PTQ)
            >>> workflow = QDQAutotunerWorkflow.from_quantize_args(
            ...     onnx_path="model.onnx",
            ...     autotune="fast",
            ...     direct_autotune=True,
            ...     quantize_mode="int8",
            ... )
            >>> optimized_path = workflow.run()
        """
        if not self.should_run_autotune:
            logger.debug("Autotune not enabled (mode='none'), skipping")
            return None

        # Validate parameters
        if not self.direct_autotune:
            if not quantized_model_path:
                logger.error(f"autotune='{self.autotune}' requires quantized_model_path from PTQ")
                return None

        # Log mode and hyperparameters
        mode = self.autotune
        logger.info(
            f"Autotune mode: '{mode}' "
            f"(schemes={self.num_schemes_per_region}, "
            f"warmup={self.warmup_runs}, timing={self.timing_runs})"
        )

        # Determine which model to use as input
        if self.direct_autotune:
            # Direct autotune: use high precision input model
            model_path = self.input_model_path
            qdq_baseline = None
            logger.info("Running direct autotune on high precision model (skipping PTQ)")
        else:
            # Post-PTQ autotune: use high precision model, import Q/DQ points from quantized
            model_path = self.input_model_path
            qdq_baseline = quantized_model_path
            logger.info(
                f"Running autotune after PTQ, importing Q/DQ insertion points from: "
                f"{quantized_model_path}"
            )

        # Setup output directory
        output_dir = (
            Path(self.output_dir) if self.output_dir else Path(model_path).parent / "autotune"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorRT benchmark
        timing_cache = str(output_dir / "trt_timing.cache")
        # These are guaranteed to be set by __post_init__, assert for mypy
        assert self.warmup_runs is not None
        assert self.timing_runs is not None
        assert self.num_schemes_per_region is not None
        benchmark = init_benchmark_instance(
            use_trtexec=self.use_trtexec,
            plugin_libraries=self.trt_plugins,
            timing_cache_file=timing_cache,
            warmup_runs=self.warmup_runs,
            timing_runs=self.timing_runs,
        )

        if benchmark is None:
            logger.error("Failed to initialize TensorRT benchmark, skipping autotune")
            return None

        try:
            # Run the autotuning workflow
            region_pattern_autotuning_workflow(
                model_path=model_path,
                output_dir=output_dir,
                num_schemes_per_region=self.num_schemes_per_region,
                pattern_cache_file=self.pattern_cache_file,
                state_file=self.state_file,
                quant_type=self.quant_type,
                default_dq_dtype=self.default_dq_dtype,
                qdq_baseline_model=qdq_baseline,
                node_filter_list=self.node_filter_list,
                verbose=self.verbose,
            )

            # Get final optimized model path
            final_model_path = output_dir / "optimized_final.onnx"

            if final_model_path.exists():
                # Determine destination path
                # For post-PTQ autotune, don't overwrite the PTQ model - use .autotune.onnx
                if self.output_path and not self.direct_autotune:
                    # Create autotune-specific output path (e.g., model.quant.onnx -> model.autotune.onnx)
                    dest_path = self.output_path.replace(".quant.onnx", ".autotune.onnx")
                    if dest_path == self.output_path:
                        # Fallback if no .quant.onnx suffix
                        dest_path = self.output_path.replace(".onnx", ".autotune.onnx")
                elif self.output_path:
                    # Direct autotune: use the specified output path
                    dest_path = self.output_path
                else:
                    # No output path specified, keep in autotune directory
                    dest_path = str(final_model_path)

                # Copy to destination if different from source
                if dest_path != str(final_model_path):
                    import shutil

                    shutil.copy(str(final_model_path), dest_path)

                mode_desc = "Direct autotune" if self.direct_autotune else "Post-PTQ autotune"
                logger.info(f"{mode_desc} completed successfully: {dest_path}")
                return dest_path
            else:
                logger.warning("Autotune completed but no final model found")
                return None

        except Exception as e:
            logger.error(f"Autotune workflow failed: {e}", exc_info=True)
            return None
