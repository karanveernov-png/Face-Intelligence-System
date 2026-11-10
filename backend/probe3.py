import numpy as np
import cv2
import onnxruntime as ort

MODELS = "D:/Opne CV/face-intelligence/face-intelligence/models"

_FAIRFACE_MIDPOINTS = np.array([1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 78.0], dtype=np.float64)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


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


def blob_imagenet(crop):
    """Current code path: blob_from_mobilenet_crops -> ImageNet mean/std."""
    blob = cv2.dnn.blobFromImages([crop], scalefactor=1.0 / 255.0, size=(224, 224),
                                  mean=(0.0, 0.0, 0.0), swapRB=True, crop=False, ddepth=cv2.CV_32F)
    return ((blob - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def blob_fairface_correct(crop):
    """FairFace standard: (pixel - 127.5) / 127.5, RGB."""
    return cv2.dnn.blobFromImages([crop], scalefactor=1.0 / 127.5, size=(224, 224),
                                  mean=(127.5, 127.5, 127.5), swapRB=True, crop=False, ddepth=cv2.CV_32F)


def main():
    sess = ort.InferenceSession(MODELS + "/fairface.onnx", providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print("fairface out_names:", out_names)
    ai = out_names.index("age_output")

    print("=== fairface.onnx: ImageNet-norm (current code) vs [-1,1] (correct) ===")
    for seed in range(4):
        face = make_face(seed=seed)
        for label, blob in (("IMAGENET (current code)", blob_imagenet(face)),
                            ("[-1,1] (correct)", blob_fairface_correct(face))):
            outs = sess.run(None, {in_name: blob})
            age_logits = np.squeeze(outs[ai])
            p = _softmax(age_logits.astype(np.float64))
            ev = float(np.dot(p, _FAIRFACE_MIDPOINTS))
            top = int(np.argmax(p))
            print(f"  seed={seed} {label}: expected_age={ev:5.1f}  top_bucket={top} "
                  f"({p[top]*100:.0f}%)  probs={['%.2f'%x for x in p]}")


if __name__ == "__main__":
    main()