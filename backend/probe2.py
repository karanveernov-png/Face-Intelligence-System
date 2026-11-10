import numpy as np
import cv2
import onnxruntime as ort

MODELS = "D:/Opne CV/face-intelligence/face-intelligence/models"


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def make_face(seed=0, size=224):
    """Deterministic synthetic skin-toned face-ish blob with a bit of structure."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    # skin-ish gradient
    yy, xx = np.mgrid[0:size, 0:size]
    img[..., 0] = np.clip(180 + 40 * np.sin(xx / 30.0) * np.sin(yy / 40.0), 0, 255).astype(np.uint8)
    img[..., 1] = np.clip(150 + 30 * np.cos(xx / 50.0), 0, 255).astype(np.uint8)
    img[..., 2] = np.clip(130 + 25 * np.cos(yy / 60.0), 0, 255).astype(np.uint8)
    # two eye-ish dark blobs + nose + mouth to give the model *something*
    cv2.circle(img, (size // 3, size // 2), 14, (40, 40, 40), -1)
    cv2.circle(img, (2 * size // 3, size // 2), 14, (40, 40, 40), -1)
    cv2.circle(img, (size // 2, size // 2 + 10), 9, (60, 60, 60), -1)
    cv2.ellipse(img, (size // 2, size // 2 + 34), (26, 9), 0, 0, 360, (70, 70, 70), -1)
    img = img.astype(np.uint8) + rng.integers(-12, 12, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def genderage_raw_input(aligned_96):
    """Current code's normalization: raw RGB [0,255]."""
    return cv2.dnn.blobFromImage(
        aligned_96, scalefactor=1.0, size=(96, 96), mean=(0.0, 0.0, 0.0), swapRB=True
    )


def genderage_correct_input(aligned_96):
    """InsightFace standard: (pixel - 127.5) / 127.5, RGB."""
    return cv2.dnn.blobFromImage(
        aligned_96, scalefactor=1.0 / 127.5, size=(96, 96), mean=(127.5, 127.5, 127.5), swapRB=True
    )


def main():
    sess = ort.InferenceSession(MODELS + "/genderage.onnx", providers=["CPUExecutionProvider"])
    print("=== genderage.onnx: raw [0,255] input vs correct [-1,1] input ===")
    for seed in range(4):
        face = make_face(seed=seed)
        # replicate the code's 96x96 alignment
        h, w = face.shape[:2]
        scale = 96.0 / (max(w, h) * 1.5)
        M = np.array([[scale, 0, 48 - (w * 0.5) * scale], [0, scale, 48 - (h * 0.5) * scale]], dtype=np.float32)
        aligned = cv2.warpAffine(face, M, (96, 96), borderValue=(0, 0, 0))

        for label, blob in (("RAW[0,255] (current code)", genderage_raw_input(aligned)),
                            ("NORM[-1,1] (correct)", genderage_correct_input(aligned))):
            out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
            preds = np.squeeze(out)
            g0, g1, age_norm = float(preds[0]), float(preds[1]), float(preds[2])
            gp = _softmax(np.array([g0, g1]))
            print(f"  seed={seed} {label}: raw_out={preds} -> age={age_norm*100:.1f} "
                  f"gender_logits=({g0:.2f},{g1:.2f}) male={g1>g0} gconf={max(gp)*100:.1f}%")


if __name__ == "__main__":
    main()