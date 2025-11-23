"""Benchmark script to measure pipeline performance."""

import time
import numpy as np
import psutil
from dual_stream_face_recognition.pipeline import DualStreamPipeline
from dual_stream_face_recognition.detection.yolo_detector import YOLODetector
from dual_stream_face_recognition.detection.retinaface_detector import RetinaFaceDetector
from dual_stream_face_recognition.buffer.adaptive_buffer import AdaptiveBuffer
from dual_stream_face_recognition.voting.temporal_voting import TemporalVotingAlgorithm


def generate_test_frames(num_frames=100, width=640, height=480):
    """Generate dummy test frames."""
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def measure_cpu_usage():
    """Get current CPU usage percentage."""
    return psutil.cpu_percent(interval=0.1)


def benchmark_yolo_only(yolo, frames):
    """Benchmark YOLO-only mode."""
    print("\nBenchmarking YOLO-only...")
    
    # Warmup
    for frame in frames[:5]:
        yolo.detect_faces(frame)
    
    cpu_before = measure_cpu_usage()
    latencies = []
    start_time = time.time()
    
    for frame in frames:
        frame_start = time.time()
        yolo.detect_faces(frame)
        latencies.append((time.time() - frame_start) * 1000)
    
    total_time = time.time() - start_time
    cpu_during = measure_cpu_usage()
    
    return {
        'mode': 'YOLO-only',
        'fps': len(frames) / total_time,
        'latency_ms': np.mean(latencies),
        'cpu_percent': max(cpu_before, cpu_during)
    }


def benchmark_retina_only(retina, frames):
    """Benchmark RetinaFace-only mode."""
    print("\nBenchmarking RetinaFace-only...")
    
    # Warmup
    for frame in frames[:5]:
        retina.detect_faces(frame)
    
    cpu_before = measure_cpu_usage()
    latencies = []
    start_time = time.time()
    
    for frame in frames:
        frame_start = time.time()
        retina.detect_faces(frame)
        latencies.append((time.time() - frame_start) * 1000)
    
    total_time = time.time() - start_time
    cpu_during = measure_cpu_usage()
    
    return {
        'mode': 'RetinaFace-only',
        'fps': len(frames) / total_time,
        'latency_ms': np.mean(latencies),
        'cpu_percent': max(cpu_before, cpu_during)
    }


def benchmark_dual_stream(pipeline, frames):
    """Benchmark dual-stream mode."""
    print("\nBenchmarking Dual-stream...")
    
    # Warmup
    for idx, frame in enumerate(frames[:5]):
        pipeline.process_frame(frame, frame_id=idx)
    
    cpu_before = measure_cpu_usage()
    latencies = []
    start_time = time.time()
    
    for idx, frame in enumerate(frames):
        frame_start = time.time()
        pipeline.process_frame(frame, frame_id=idx)
        latencies.append((time.time() - frame_start) * 1000)
    
    total_time = time.time() - start_time
    cpu_during = measure_cpu_usage()
    
    return {
        'mode': 'Dual-stream',
        'fps': len(frames) / total_time,
        'latency_ms': np.mean(latencies),
        'cpu_percent': max(cpu_before, cpu_during)
    }


def run_benchmark(num_frames=100):
    """Run benchmark for all modes."""
    print(f"Generating {num_frames} test frames...")
    test_frames = generate_test_frames(num_frames)
    
    results = []
    
    # 1. YOLO-only mode
    yolo = YOLODetector(device='cpu')
    result_yolo = benchmark_yolo_only(yolo, test_frames)
    results.append(result_yolo)
    
    # 2. RetinaFace-only mode
    retina = RetinaFaceDetector(device='cpu')
    result_retina = benchmark_retina_only(retina, test_frames)
    results.append(result_retina)
    
    # 3. Dual-stream mode
    yolo_dual = YOLODetector(device='cpu')
    retina_dual = RetinaFaceDetector(device='cpu')
    buffer_dual = AdaptiveBuffer(buffer_size=30)
    voting_dual = TemporalVotingAlgorithm(window_size=8)
    pipeline_dual = DualStreamPipeline(
        yolo_detector=yolo_dual,
        retina_detector=retina_dual,
        buffer=buffer_dual,
        voting=voting_dual
    )
    result_dual = benchmark_dual_stream(pipeline_dual, test_frames)
    results.append(result_dual)
    
    return results


def print_results(results, save_path='benchmark_results.txt'):
    """Print and save benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    # Print table
    print(f"\n{'Mode':<20} {'FPS':<10} {'Latency':<12} {'CPU%':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['mode']:<20} {r['fps']:<10.1f} {r['latency_ms']:<12.1f} {r['cpu_percent']:<10.1f}")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write("BENCHMARK RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Mode':<20} {'FPS':<10} {'Latency (ms)':<15} {'CPU%':<10}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['mode']:<20} {r['fps']:<10.1f} {r['latency_ms']:<15.1f} {r['cpu_percent']:<10.1f}\n")
    
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    import sys
    
    num_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    save_path = sys.argv[2] if len(sys.argv) > 2 else 'benchmark_results.txt'
    
    print(f"Running benchmark with {num_frames} frames...")
    results = run_benchmark(num_frames)
    print_results(results, save_path)

