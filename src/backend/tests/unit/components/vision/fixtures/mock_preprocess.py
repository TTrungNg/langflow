# Ảnh 284×177 (nhỏ hơn 640) — không cần resize
MOCK_SMALL_IMAGE = {
    "original_width": 284, "original_height": 177,
    "expected_scale": 640 / 284,       # ~2.253
    "expected_resized_w": 640,
    "expected_resized_h": 398,         # int(177 * scale) = int(398.87) = 398
    "expected_pad_left": 0,
    "expected_pad_top": (640 - 398) // 2,  # 121
}

# Ảnh 1920×1080 (lớn hơn 640) — cần resize
MOCK_LARGE_IMAGE = {
    "original_width": 1920, "original_height": 1080,
    "expected_scale": 640 / 1920,      # ~0.333
    "expected_resized_w": 640,
    "expected_resized_h": 360,         # 1080 * scale
    "expected_pad_left": 0,
    "expected_pad_top": (640 - 360) // 2,  # 140
}

# Expected normalize values cho SCRFD preset
MOCK_SCRFD_NORMALIZE = {
    "preset": "SCRFD",
    "mean": [0, 0, 0],
    "std": 1.0,
    "divide_255": True,
    # Pixel 255 → 1.0 sau normalize
    # Pixel 0   → 0.0 sau normalize
}

# Expected normalize values cho ImageNet preset
MOCK_IMAGENET_NORMALIZE = {
    "preset": "ImageNet",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "divide_255": True,
}

# Expected tensor shape sau To Tensor
MOCK_TENSOR_NCHW = {
    "format": "NCHW",
    "shape": [1, 3, 640, 640],
    "dtype": "float32",
}

MOCK_TENSOR_NHWC = {
    "format": "NHWC",
    "shape": [1, 640, 640, 3],
    "dtype": "float32",
}