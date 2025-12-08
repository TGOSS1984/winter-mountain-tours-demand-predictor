# src/image_features.py

from __future__ import annotations

from io import BytesIO
from typing import Dict

import numpy as np
from PIL import Image


def _load_image_bytes(image_bytes: bytes, size: int = 224) -> np.ndarray:
   
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size))
    return np.asarray(img, dtype=np.uint8)


def extract_weather_features_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    
    arr = _load_image_bytes(image_bytes)  # shape (H, W, 3)
    arr_f = arr.astype("float32") / 255.0  # scale to [0, 1]

    # Basic luminance statistics
    brightness_mean = float(arr_f.mean())
    brightness_std = float(arr_f.std())

    # Channel-wise stats
    r = arr_f[..., 0]
    g = arr_f[..., 1]
    b = arr_f[..., 2]

    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
    r_std, g_std, b_std = float(r.std()), float(g.std()), float(b.std())

    # Proportion of "very bright" pixels – rough proxy for snow coverage / bright cloud
    bright_mask = arr_f.mean(axis=-1) > 0.8  # average over channels
    bright_ratio = float(bright_mask.mean())

    # Simple "colourfulness" proxy: std across channels
    # (foggy / low-visibility scenes often look flat and grey)
    per_pixel_channel_std = arr_f.std(axis=-1)
    colourfulness_mean = float(per_pixel_channel_std.mean())

    features = {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "r_std": r_std,
        "g_std": g_std,
        "b_std": b_std,
        "bright_ratio": bright_ratio,
        "colourfulness_mean": colourfulness_mean,
    }

    return features
