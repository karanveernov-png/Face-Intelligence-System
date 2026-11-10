import sys
import numpy as np
import cv2
import onnxruntime as ort

print("cv2", cv2.__version__, "has CascadeClassifier:", hasattr(cv2, "CascadeClassifier"))
print("ort", ort.__version__)

MODELS = "D:/Opne CV/face-intelligence/face-intelligence/models"
for name in ["genderage.onnx", "fairface.onnx", "enet_b0_8_best_vgaf.onnx"]:
    try:
        s = ort.InferenceSession(MODELS + "/" + name, providers=["CPUExecutionProvider"])
        i = s.get_inputs()[0]
        outs = [
            (o.name, [d if isinstance(d, int) else str(d) for d in o.shape])
            for o in s.get_outputs()
        ]
        print(name, "| in:", i.name, i.shape, i.type, "| outs:", outs)
    except Exception as e:
        print(name, "ERR", repr(e))