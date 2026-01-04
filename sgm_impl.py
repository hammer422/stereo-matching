import numpy as np
import cv2
import time
from typing import Callable, List, Tuple, Any

def census_transform_naive(image: np.ndarray, window_size: int) -> np.ndarray:
    assert window_size in [3, 5, 7], "window_size must be 3, 5 or 7"
    assert image.ndim == 2 and image.dtype == np.uint8

    H, W = image.shape
    pad = window_size // 2
    padded = np.pad(image, pad, mode='edge')
    
    census = np.zeros((H, W), dtype=np.uint64)

    for i in range(H):
        for j in range(W):
            center = padded[i + pad, j + pad]
            census_val = np.uint64(0)
            for di in range(-pad, pad + 1):
                for dj in range(-pad, pad + 1):
                    if di == 0 and dj == 0:
                        continue
                    census_val = census_val << 1
                    neighbor = padded[i + pad + di, j + pad + dj]
                    if neighbor < center:
                        census_val += np.uint64(1)
            census[i, j] = census_val
    return census

def census_transform_optimized(image: np.ndarray, window_size: int) -> np.ndarray:
    assert window_size in [3, 5, 7], "window_size must be 3, 5 or 7"
    assert image.ndim == 2 and image.dtype == np.uint8

    H, W = image.shape
    pad = window_size // 2
    padded = np.pad(image, pad, mode='edge')

    center = image  # shape (H, W)

    census = np.zeros((H, W), dtype=np.uint64)

    for di in range(window_size):
        for dj in range(window_size):
            if di == pad and dj == pad:
                continue
            census = census << 1
            neighbor = padded[di:di + H, dj:dj + W]
            census += (neighbor < center).astype(np.uint64)

    return census

def generate_test_image(shape: Tuple[int, int] = (100, 100), seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)


def verify_consistency(
    funcs: List[Callable],
    func_names: List[str],
    test_shapes: List[Tuple[int, int]],
    window_size: int,
    input_generator: Callable = generate_test_image,
    **kwargs
) -> bool:
    """
    Verify that all functions produce bit-wise identical outputs.
    
    Args:
        funcs: List of census transform functions to compare
        func_names: Names for display (e.g., ['naive', 'optimized'])
        test_shapes: List of (H, W) to test
        window_size: 3, 5, or 7
        input_generator: Function to generate test image
        **kwargs: Extra args passed to each function (e.g., window_size via partial)
    
    Returns:
        True if all consistent
    """
    print(f"\n[Correctness Check] window_size={window_size}")
    for shape in test_shapes:
        img = input_generator(shape, seed=42)
        results = []
        for func in funcs:
            out = func(img, window_size)
            results.append(out)
        
        # Compare all against the first
        base = results[0]
        for i, res in enumerate(results[1:], 1):
            if not np.array_equal(base, res):
                print(f"❌ Mismatch between {func_names[0]} and {func_names[i]} at shape {shape}")
                return False
    print(f"✅ All {len(funcs)} implementations are consistent for window_size={window_size}")
    return True
    
def benchmark_functions(
    funcs: List[Callable],
    func_names: List[str],
    shape: Tuple[int, int] = (500, 500),
    window_size: int = 5,
    n_runs: int = 3,
    input_generator: Callable = generate_test_image,
    warmup: bool = True
):
    """
    Benchmark multiple functions and report timing + speedup vs baseline (first function).
    """
    print(f"\n[Performance Benchmark] shape={shape}, window_size={window_size}, n_runs={n_runs}")
    
    img = input_generator(shape, seed=123)
    
    # Warm-up
    if warmup:
        for func in funcs:
            func(img, window_size)
    
    timings = []
    for func in funcs:
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(img, window_size)
            times.append(time.perf_counter() - start)
        timings.append(np.array(times))
    
    # Report
    baseline_mean = np.mean(timings[0])
    print(f"{'Function':<15} {'Mean (s)':<10} {'Std (s)':<10} {'Speedup':<10}")
    print("-" * 50)
    for i, (name, times) in enumerate(zip(func_names, timings)):
        mean_t = np.mean(times)
        std_t = np.std(times)
        speedup = baseline_mean / mean_t if mean_t > 0 else float('inf')
        speedup_str = f"{speedup:.2f}x" if i > 0 else "baseline"
        print(f"{name:<15} {mean_t:<10.4f} {std_t:<10.4f} {speedup_str:<10}")
    print("-" * 50)
    
    
if __name__ == "__main__":
    # census transform functions to test
    census_funcs = [
        census_transform_naive,
        census_transform_optimized,
    ]
    census_func_names = ["naive", "optimized"]
    census_shapes = [(64,64), (200,200), (512,512), (1280, 960)]
    for ws in [3, 5, 7]:
        verify_consistency(
            census_funcs,
            census_func_names,
            census_shapes,
            window_size=ws
        )
        benchmark_functions(
            census_funcs,
            census_func_names,
            census_shapes[-1],
            window_size=ws,
            n_runs=10
        )
