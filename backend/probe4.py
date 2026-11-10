import sys
import os
import numpy as np
import cv2

# Force the real models dir (absolute) so cwd doesn't matter
os.environ.setdefault("FACE_INTL_MODELS", "D:/Opne CV/face-intelligence/face-intelligence/models")

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from app.services import age_estimation as ae
from app.services.age_estimation import _predict_single, _genderage_estimate, _softmax, _get_genderage_session


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


def genderage_fixed(image_bgr, x1, y1, x2, y2):
    """Same as _genderage_estimate but with the correct [-1,1] InsightFace normalization."""
    session = _get_genderage_session()
    aligned_96 = ae._align_insightface_96(image_bgr, x1, y1, x2 - x1, y2 - y1)
    blob = cv2.dnn.blobFromImage(
        aligned_96, scalefactor=1.0 / 127.5, size=(96, 96),
        mean=(127.5, 127.5, 127.5), swapRB=True,
    )
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    preds = np.squeeze(outputs[0])
    gender_logits = preds[0:2]
    is_male = bool(gender_logits[1] > gender_logits[0])
    gender_probs = _softmax(gender_logits)
    gender_confidence = float(np.max(gender_probs))
    continuous_age = float(preds[2]) * 100.0
    return continuous_age, is_male, gender_confidence


def main():
    for seed in range(3):
        face = make_face(seed=seed)
        h, w = face.shape[:2]

        r = _predict_single(face, (0, 0, w, h))
        ga = _genderage_estimate(face, 0, 0, w, h)
        ff = ae._fairface_estimate(face)
        ga_fixed = genderage_fixed(face, 0, 0, w, h)

        print(f"--- seed={seed} ---")
        print(f"  genderage (CURRENT, raw [0,255]): age={ga[0]:.1f} male={ga[1]} gconf={ga[2]*100:.1f}%")
        print(f"  genderage (FIXED,   [-1,1]    ): age={ga_fixed[0]:.1f} male={ga_fixed[1]} gconf={ga_fixed[2]*100:.1f}%")
        print(f"  fairface                         : age={ff}")
        print(f"  _predict_single CURRENT          : age={r.estimated_age} range={r.estimated_age_range} "
              f"unc=+-{r.estimated_age_uncertainty_years} gender={r.gender}({r.gender_confidence}%)")

        ests = [ga_fixed[0], ff]
        avg = float(np.mean(ests))
        print(f"  ensemble WOULD BE                : age={avg:.1f} range={ae._bucket_for_age(int(round(avg)))}")


if __name__ == "__main__":
    main()