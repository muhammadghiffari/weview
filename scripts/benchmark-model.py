#!/usr/bin/env python3
"""
WiFi-DensePose Model Benchmarking

Loads trained ONNX models, runs inference on test data, and reports
performance metrics: latency, throughput, PCK@0.2, model size, and
estimated FLOPs.

Can compare multiple models from a hyperparameter sweep.

Usage:
    # Benchmark a single model
    python scripts/benchmark-model.py --model checkpoints/best.onnx

    # Benchmark with recorded test data
    python scripts/benchmark-model.py --model best.onnx --test-data data/recordings/test.csi.jsonl

    # Compare models from a sweep
    python scripts/benchmark-model.py --sweep-dir training-results/wdp-train-a100-*/checkpoints/

    # Benchmark with synthetic data (no recordings needed)
    python scripts/benchmark-model.py --model best.onnx --synthetic --num-samples 200

    # Export results as JSON
    python scripts/benchmark-model.py --model best.onnx --output results.json

Prerequisites:
    pip install onnxruntime numpy
    Optional: pip install onnx  (for FLOPs estimation)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)


# ── Configuration ────────────────────────────────────────────────────────────

# Default model input shape (must match TrainingConfig defaults)
NUM_SUBCARRIERS = 56
NUM_ANTENNAS_TX = 3
NUM_ANTENNAS_RX = 3
WINDOW_FRAMES = 100
NUM_KEYPOINTS = 17
HEATMAP_SIZE = 56

# PCK threshold
PCK_THRESHOLD = 0.2


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    model_path: str
    model_size_mb: float
    num_parameters: Optional[int] = None
    estimated_flops: Optional[int] = None

    # Latency
    warmup_runs: int = 10
    benchmark_runs: int = 100
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_fps: float = 0.0

    # Accuracy (if ground truth available)
    pck_at_02: Optional[float] = None
    mean_per_joint_error: Optional[float] = None
    num_test_samples: int = 0

    # Input shape
    input_shape: list = field(default_factory=list)
    provider: str = ""


# ── ONNX model loading ──────────────────────────────────────────────────────

def load_model(model_path: str) -> ort.InferenceSession:
    """Load an ONNX model with the best available execution provider."""
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count() or 4

    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    return session


def get_model_info(model_path: str) -> dict:
    """Extract model metadata: size, parameter count, FLOPs estimate."""
    path = Path(model_path)
    size_mb = path.stat().st_size / (1024 * 1024)

    info = {
        "size_mb": round(size_mb, 2),
        "num_parameters": None,
        "estimated_flops": None,
    }

    # Try to count parameters via onnx
    try:
        import onnx
        model = onnx.load(model_path)
        total_params = 0
        for initializer in model.graph.initializer:
            shape = list(initializer.dims)
            if shape:
                total_params += int(np.prod(shape))
        info["num_parameters"] = total_params

        # Rough FLOPs estimate: ~2 * params (multiply-accumulate)
        info["estimated_flops"] = total_params * 2
    except ImportError:
        pass
    except Exception as e:
        print(f"  Warning: Could not extract parameter count: {e}")

    return info


# ── Synthetic data generation ────────────────────────────────────────────────

def generate_synthetic_input(
    batch_size: int = 1,
    num_subcarriers: int = NUM_SUBCARRIERS,
    num_tx: int = NUM_ANTENNAS_TX,
    num_rx: int = NUM_ANTENNAS_RX,
    window_frames: int = WINDOW_FRAMES,
) -> np.ndarray:
    """Generate synthetic CSI input tensor matching the model's expected shape.

    The WiFi-DensePose model expects input shape:
      [batch, channels, height, width]
    where channels = num_tx * num_rx, height = window_frames, width = num_subcarriers.
    """
    channels = num_tx * num_rx  # 3x3 = 9 MIMO streams
    # Simulate CSI amplitude data with realistic distribution
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=(batch_size, channels, window_frames, num_subcarriers))
    return data.astype(np.float32)


def generate_synthetic_keypoints(
    num_samples: int,
    num_keypoints: int = NUM_KEYPOINTS,
    heatmap_size: int = HEATMAP_SIZE,
) -> np.ndarray:
    """Generate synthetic ground truth keypoint coordinates for PCK evaluation."""
    rng = np.random.default_rng(123)
    # Keypoints as (x, y) in [0, heatmap_size) range
    return rng.uniform(0, heatmap_size, size=(num_samples, num_keypoints, 2)).astype(np.float32)


# ── Load test data from .csi.jsonl ──────────────────────────────────────────

def load_test_data(
    jsonl_path: str,
    window_frames: int = WINDOW_FRAMES,
    num_subcarriers: int = NUM_SUBCARRIERS,
    max_samples: int = 500,
) -> np.ndarray:
    """Load CSI frames from a .csi.jsonl file and window them into model inputs."""
    frames = []
    path = Path(jsonl_path)

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                subs = record.get("subcarriers", [])
                if len(subs) > 0:
                    frames.append(subs)
            except json.JSONDecodeError:
                continue

    if len(frames) < window_frames:
        print(f"  Warning: Only {len(frames)} frames, need {window_frames}. Padding with zeros.")
        while len(frames) < window_frames:
            frames.append([0.0] * num_subcarriers)

    # Normalize subcarrier count
    normalized = []
    for frame in frames:
        if len(frame) < num_subcarriers:
            frame = frame + [0.0] * (num_subcarriers - len(frame))
        elif len(frame) > num_subcarriers:
            # Downsample via linear interpolation
            indices = np.linspace(0, len(frame) - 1, num_subcarriers)
            frame = np.interp(indices, range(len(frame)), frame).tolist()
        normalized.append(frame)

    frames = normalized

    # Create sliding windows
    samples = []
    stride = max(1, window_frames // 2)
    for i in range(0, len(frames) - window_frames + 1, stride):
        window = frames[i : i + window_frames]
        # Shape: [channels=1, window_frames, num_subcarriers]
        # Expand single stream to 9 channels (repeat for MIMO)
        arr = np.array(window, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)  # [1, window_frames, num_subcarriers]
        arr = np.repeat(arr, NUM_ANTENNAS_TX * NUM_ANTENNAS_RX, axis=0)  # [9, window, subs]
        samples.append(arr)

        if len(samples) >= max_samples:
            break

    if not samples:
        return generate_synthetic_input(1)

    return np.stack(samples, axis=0)  # [N, 9, window_frames, num_subcarriers]


# ── Benchmarking ─────────────────────────────────────────────────────────────

def benchmark_latency(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    warmup: int = 10,
    runs: int = 100,
) -> dict:
    """Measure inference latency over multiple runs."""
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: input_data[:1]})

    # Timed runs
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data[:1]})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
    }


def compute_pck(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = PCK_THRESHOLD,
    normalize_by: float = HEATMAP_SIZE,
) -> float:
    """Compute Percentage of Correct Keypoints at a given threshold.

    PCK@t = fraction of predicted keypoints within t * normalize_by of ground truth.
    """
    if predictions.shape != ground_truth.shape:
        return 0.0

    # Euclidean distance per keypoint
    distances = np.linalg.norm(predictions - ground_truth, axis=-1)  # [N, K]
    threshold_pixels = threshold * normalize_by
    correct = (distances < threshold_pixels).astype(float)
    return float(np.mean(correct))


def extract_keypoints_from_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Convert heatmap outputs [N, K, H, W] to keypoint coordinates [N, K, 2]."""
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    max_idx = np.argmax(flat, axis=-1)  # [N, K]
    y = max_idx // w
    x = max_idx % w
    return np.stack([x, y], axis=-1).astype(np.float32)


def benchmark_model(
    model_path: str,
    test_data: Optional[np.ndarray] = None,
    gt_keypoints: Optional[np.ndarray] = None,
    warmup: int = 10,
    runs: int = 100,
) -> BenchmarkResult:
    """Run full benchmark on a single model."""
    print(f"\nBenchmarking: {model_path}")

    # Load model
    session = load_model(model_path)
    provider = session.get_providers()[0]
    print(f"  Provider: {provider}")

    # Model info
    model_info = get_model_info(model_path)
    print(f"  Size: {model_info['size_mb']} MB")
    if model_info["num_parameters"]:
        print(f"  Parameters: {model_info['num_parameters']:,}")
    if model_info["estimated_flops"]:
        print(f"  Estimated FLOPs: {model_info['estimated_flops']:,}")

    # Input shape
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape
    print(f"  Input: {input_meta.name} {input_shape} ({input_meta.type})")

    # Output shapes
    for out in session.get_outputs():
        print(f"  Output: {out.name} {out.shape}")

    # Generate or use provided test data
    if test_data is None:
        # Infer shape from model
        if input_shape and all(isinstance(d, int) for d in input_shape):
            batch = max(1, input_shape[0] if input_shape[0] > 0 else 1)
            test_data = np.random.randn(*[batch if d <= 0 else d for d in input_shape]).astype(np.float32)
        else:
            test_data = generate_synthetic_input(1)

    # Latency benchmark
    print(f"  Running {warmup} warmup + {runs} benchmark iterations...")
    latency = benchmark_latency(session, test_data, warmup=warmup, runs=runs)
    print(f"  Latency: {latency['mean_ms']:.2f} +/- {latency['std_ms']:.2f} ms")
    print(f"  P50/P95/P99: {latency['p50_ms']:.2f} / {latency['p95_ms']:.2f} / {latency['p99_ms']:.2f} ms")
    print(f"  Throughput: {latency['throughput_fps']:.1f} fps")

    # Accuracy (if ground truth provided or we can do synthetic evaluation)
    pck = None
    mpjpe = None
    num_samples = 0

    if gt_keypoints is not None and test_data is not None:
        input_name = session.get_inputs()[0].name
        all_preds = []

        for i in range(len(test_data)):
            outputs = session.run(None, {input_name: test_data[i : i + 1]})
            # Assume first output is keypoint heatmaps [1, K, H, W]
            heatmaps = outputs[0]
            if heatmaps.ndim == 4:
                kp = extract_keypoints_from_heatmaps(heatmaps)
                all_preds.append(kp[0])

        if all_preds:
            predictions = np.stack(all_preds)
            gt = gt_keypoints[: len(predictions)]
            pck = compute_pck(predictions, gt)
            distances = np.linalg.norm(predictions - gt, axis=-1)
            mpjpe = float(np.mean(distances))
            num_samples = len(predictions)
            print(f"  PCK@{PCK_THRESHOLD}: {pck:.4f}")
            print(f"  MPJPE: {mpjpe:.2f} px")
            print(f"  Samples: {num_samples}")

    result = BenchmarkResult(
        model_path=model_path,
        model_size_mb=model_info["size_mb"],
        num_parameters=model_info["num_parameters"],
        estimated_flops=model_info["estimated_flops"],
        warmup_runs=warmup,
        benchmark_runs=runs,
        latency_mean_ms=round(latency["mean_ms"], 3),
        latency_std_ms=round(latency["std_ms"], 3),
        latency_p50_ms=round(latency["p50_ms"], 3),
        latency_p95_ms=round(latency["p95_ms"], 3),
        latency_p99_ms=round(latency["p99_ms"], 3),
        throughput_fps=round(latency["throughput_fps"], 1),
        pck_at_02=round(pck, 4) if pck is not None else None,
        mean_per_joint_error=round(mpjpe, 2) if mpjpe is not None else None,
        num_test_samples=num_samples,
        input_shape=list(input_shape) if input_shape else [],
        provider=provider,
    )

    return result


# ── Comparison table ─────────────────────────────────────────────────────────

def print_comparison_table(results: list[BenchmarkResult]):
    """Print a formatted comparison table of multiple models."""
    if not results:
        return

    print("\n" + "=" * 100)
    print("  Model Comparison")
    print("=" * 100)

    # Header
    print(
        f"{'Model':<35} {'Size(MB)':>8} {'Params':>10} "
        f"{'Lat(ms)':>8} {'P95(ms)':>8} {'FPS':>7} {'PCK@0.2':>8}"
    )
    print("-" * 100)

    for r in results:
        name = Path(r.model_path).stem[:33]
        params = f"{r.num_parameters:,}" if r.num_parameters else "?"
        pck = f"{r.pck_at_02:.4f}" if r.pck_at_02 is not None else "N/A"

        print(
            f"{name:<35} {r.model_size_mb:>8.2f} {params:>10} "
            f"{r.latency_mean_ms:>8.2f} {r.latency_p95_ms:>8.2f} "
            f"{r.throughput_fps:>7.1f} {pck:>8}"
        )

    print("=" * 100)

    # Best model by latency
    best_latency = min(results, key=lambda r: r.latency_mean_ms)
    print(f"\n  Fastest: {Path(best_latency.model_path).stem} ({best_latency.latency_mean_ms:.2f} ms)")

    # Best by PCK (if available)
    pck_results = [r for r in results if r.pck_at_02 is not None]
    if pck_results:
        best_pck = max(pck_results, key=lambda r: r.pck_at_02)
        print(f"  Best accuracy: {Path(best_pck.model_path).stem} (PCK@0.2={best_pck.pck_at_02:.4f})")

    # Smallest model
    smallest = min(results, key=lambda r: r.model_size_mb)
    print(f"  Smallest: {Path(smallest.model_path).stem} ({smallest.model_size_mb:.2f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark WiFi-DensePose ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, help="Path to a single ONNX model")
    parser.add_argument("--sweep-dir", type=str, help="Directory containing multiple ONNX models to compare")
    parser.add_argument("--test-data", type=str, help="Path to .csi.jsonl test data file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test data")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of synthetic samples (default: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark iterations (default: 100)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--gpu", action="store_true", help="Force GPU execution provider")

    args = parser.parse_args()

    if not args.model and not args.sweep_dir:
        parser.error("Specify --model or --sweep-dir")

    # Prepare test data
    test_data = None
    gt_keypoints = None

    if args.test_data:
        print(f"Loading test data from: {args.test_data}")
        test_data = load_test_data(args.test_data)
        print(f"  Loaded {len(test_data)} windowed samples")
    elif args.synthetic:
        print(f"Generating {args.num_samples} synthetic samples...")
        test_data = generate_synthetic_input(args.num_samples)
        gt_keypoints = generate_synthetic_keypoints(args.num_samples)
        print(f"  Input shape: {test_data.shape}")

    # Collect models
    model_paths = []
    if args.model:
        model_paths.append(args.model)
    if args.sweep_dir:
        sweep = Path(args.sweep_dir)
        if sweep.is_dir():
            model_paths.extend(sorted(str(p) for p in sweep.glob("**/*.onnx")))
        else:
            # Glob pattern
            from glob import glob
            model_paths.extend(sorted(glob(str(sweep))))

    if not model_paths:
        print("ERROR: No ONNX models found.")
        sys.exit(1)

    print(f"Found {len(model_paths)} model(s) to benchmark.")

    # Benchmark each model
    results = []
    for path in model_paths:
        if not Path(path).exists():
            print(f"  Skipping (not found): {path}")
            continue
        try:
            result = benchmark_model(
                path,
                test_data=test_data,
                gt_keypoints=gt_keypoints,
                warmup=args.warmup,
                runs=args.runs,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR benchmarking {path}: {e}")

    # Comparison table
    if len(results) > 1:
        print_comparison_table(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "benchmark_results": [asdict(r) for r in results],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "num_models": len(results),
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {output_path}")

    if not results:
        print("No models were successfully benchmarked.")
        sys.exit(1)


if __name__ == "__main__":
    main()
