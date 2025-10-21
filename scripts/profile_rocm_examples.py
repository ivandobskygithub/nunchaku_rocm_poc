#!/usr/bin/env python3
"""Profile nunchaku example scripts in a ROCm environment.

This helper runs selected example scripts inside the current Python process so
that we can capture torch/ROCm metrics (runtime, peak GPU memory, HIP runtime
information). Optionally a baseline JSON file can be supplied to compare
against measurements from other hardware (e.g., NVIDIA).

Example usage::

    python scripts/profile_rocm_examples.py \
        --examples examples/v1/qwen-image-edit-2509.py examples/flux.1-dev.py \
        --output logs/rocm_profile.json --baseline logs/nvidia_profile.json
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _resolve_example_paths(paths: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw in paths:
        if raw.startswith("@"):
            list_path = Path(raw[1:])
            if not list_path.exists():
                raise FileNotFoundError(f"Example list file not found: {list_path}")
            for line in list_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                resolved.append(Path(line))
            continue
        resolved.append(Path(raw))
    return resolved


def _collect_rocm_context() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch_available": False}
    try:
        import torch
    except ImportError:
        info["error"] = "torch is not importable"
        return info

    info["torch_available"] = True
    info["torch_version"] = torch.__version__
    info["hip_version"] = getattr(torch.version, "hip", None)
    info["cuda_compiled_version"] = getattr(torch.version, "cuda", None)
    info["is_cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["current_device"] = torch.cuda.current_device()
    return info


def _run_example(example: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "example": str(example),
        "status": "pending",
    }
    if not example.exists():
        result["status"] = "missing"
        result["error"] = "file does not exist"
        return result

    rocm_info = _collect_rocm_context()
    result["rocm_info"] = rocm_info
    if not rocm_info.get("torch_available"):
        result["status"] = "skipped"
        result["error"] = rocm_info.get("error", "torch not available")
        return result

    import torch

    if not torch.cuda.is_available():
        result["status"] = "skipped"
        result["error"] = "torch.cuda is not available"
        return result

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    try:
        runpy.run_path(str(example), run_name="__main__")
        torch.cuda.synchronize()
    except SystemExit as exc:  # allow examples to call sys.exit
        result["exit_code"] = exc.code
        torch.cuda.synchronize()
        if exc.code not in (0, None):
            result["status"] = "error"
            result["error"] = f"example exited with code {exc.code}"
            return result
    except Exception as exc:  # noqa: BLE001 - diagnostics for profiling
        torch.cuda.synchronize()
        result["status"] = "error"
        result["error"] = f"{exc.__class__.__name__}: {exc}"
        return result
    finally:
        end_time = time.perf_counter()

    duration = end_time - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result.update(
        {
            "status": "success",
            "duration_s": duration,
            "peak_memory_mb": peak_mem,
            "hip_version": rocm_info.get("hip_version"),
            "device_name": rocm_info.get("device_name"),
        }
    )

    return result


def _compare_to_baseline(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    if "duration_s" in current and "duration_s" in baseline:
        comparison["duration_delta_s"] = current["duration_s"] - baseline["duration_s"]
        if baseline["duration_s"]:
            comparison["duration_delta_pct"] = (
                comparison["duration_delta_s"] / baseline["duration_s"] * 100.0
            )
    if "peak_memory_mb" in current and "peak_memory_mb" in baseline:
        comparison["peak_memory_delta_mb"] = current["peak_memory_mb"] - baseline["peak_memory_mb"]
    return comparison


def _load_baseline(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Baseline JSON must contain an object at the top level")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        nargs="+",
        required=True,
        help=(
            "Example scripts to run. Prefix a value with '@' to read additional paths "
            "from the referenced file (one per line)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the collected metrics as JSON.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional JSON file containing baseline metrics for comparison.",
    )

    args = parser.parse_args(argv)
    examples = _resolve_example_paths(args.examples)

    baseline_data: Dict[str, Dict[str, Any]] = {}
    if args.baseline:
        baseline_data = _load_baseline(args.baseline)

    results: Dict[str, Dict[str, Any]] = {}
    had_errors = False

    for example in examples:
        print(f"\n▶ Running {example}...")
        result = _run_example(example)
        results[str(example)] = result

        status = result.get("status")
        if status == "success":
            print(
                f"  ✓ duration: {result['duration_s']:.2f}s, peak memory: {result['peak_memory_mb']:.1f} MiB, "
                f"HIP {result.get('hip_version')}"
            )
        else:
            print(f"  ✗ status={status}: {result.get('error')}")
            had_errors = True

        if args.baseline and str(example) in baseline_data:
            comparison = _compare_to_baseline(result, baseline_data[str(example)])
            if comparison:
                print("  ↕ baseline deltas:")
                for key, value in comparison.items():
                    print(f"    {key}: {value:+.3f}")
                result["baseline_comparison"] = comparison

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nMetrics written to {args.output}")

    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
