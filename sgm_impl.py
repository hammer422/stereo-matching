import numpy as np
import time
import cv2
from typing import Tuple
def census_transform_naive(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    原始 Census Transform 实现（循环版本）
    Args:
        image: 输入灰度图，shape (H, W)，dtype uint8
        window_size: 窗口大小，必须为奇数（3,5,7）
    Returns:
        census: Census 编码图，dtype uint64，shape (H, W)
    """
    assert window_size in [3, 5, 7], "window_size must be 3, 5 or 7"
    assert image.ndim == 2 and image.dtype == np.uint8

    H, W = image.shape
    pad = window_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    census = np.zeros((H, W), dtype=np.uint64)

    for i in range(H):
        for j in range(W):
            center = padded[i + pad, j + pad]
            census_val = np.uint64(0)
            for di in range(window_size):
                for dj in range(window_size):
                    census_val <<= 1
                    neighbor = padded[i + di, j + dj]
                    if neighbor < center:
                        census_val += np.uint64(1)
            census[i, j] = census_val
    return census

def census_transform_optimized(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    优化的 Census Transform（向量化版本）
    使用 NumPy 广播和位运算加速
    """
    assert window_size in [3, 5, 7], "window_size must be 3, 5 or 7"
    assert image.ndim == 2 and image.dtype == np.uint8

    H, W = image.shape
    pad = window_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)

    # 提取中心像素图
    center = image  # shape (H, W)

    census = np.zeros((H, W), dtype=np.uint64)

    for di in range(window_size):
        for dj in range(window_size):
            census = census << 1
            neighbor = padded[di:di + H, dj:dj + W]
            census += (neighbor < center).astype(np.uint64)

    return census


def generate_test_image(shape: Tuple[int, int] = (100, 100), seed: int = 42) -> np.ndarray:
    """生成随机测试图像"""
    np.random.seed(seed)
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def test_correctness(window_size: int = 5, shape: Tuple[int, int] = (3, 3)):
    """验证两个版本结果是否一致"""
    img = generate_test_image(shape)
    print(img[:5, :5])
    census1 = census_transform_naive(img, window_size)
    census2 = census_transform_optimized(img, window_size)
    print(census1)
    print(census2)
    assert np.array_equal(census1, census2), f"Results differ for window_size={window_size}"
    print(f"✅ Correctness test passed for window_size={window_size}")


if __name__ == "__main__":
    # 正确性验证
    for ws in [3, 5, 7]:
        test_correctness(ws)
