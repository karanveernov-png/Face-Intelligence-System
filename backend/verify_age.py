import sys
import os
import numpy as np
import cv2

os.environ.setdefault("FACE_INTL_MODELS", "D:/Opne CV/face-intelligence/face-intelligence/models")
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from app.services import age_estimation as ae
from app.services.age_estimation import _predict_single, _genderage_estimate, _get_genderage_session


def make_face(seed=0, size=224):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    img[..., 0] = np.clip(180 + 40 * np.sin(xx / 30.0) * np.sin(yy / 40.0), 0, 255).astype(np.uint8)
    img[..., 1] = np.clip(150 + 30 * np.cos(xx / 50.0), 0, 255).astype(np.uint8)
    img[..., 2] = np.clip(130 + 25 * np.cos(yy / 60.0), 0, 255).astype(np.uint8)
    cv2.circle(img, (size // 3, size // 2), 14, (40, 40, 40), -1)
    cv2.circle(img, (2 * size // 3, size // 2), 14, (40, 40, 40), -1)
    cv2.circle(img, (size // 2, size // 2 + 10), 9, (60, 60, 60), -1)
    cv2.ellipse(img, (size // 2, size // 2 + 34), (26, 9), 0, 0, 360, (70, 70, 70), -1)
    img = img.astype(np.int16) + rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    print("=== Age estimation after revert ===")
    for seed in range(3):
        face = make_face(seed=seed)
        h, w = face.shape[:2]
        r = _predict_single(face, (0, 0, w, h))
        ga = _genderage_estimate(face, 0, 0, w, h)
        ff = ae._fairface_estimate(face)
        print(f"  seed={seed}: genderage age={ga[0]:.1f} male={ga[1]} gconf={ga[2]*100:.1f}% | "
              f"fairface age={ff:.1f} | ensemble age={r.estimated_age} range={r.estimated_age_range} "
              f"gender={r.gender}({r.gender_confidence}%)")


if __name__ == "__main__":
    main()