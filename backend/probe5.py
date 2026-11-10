import onnxruntime as ort
import onnx
from onnx import numpy_helper

MODELS = "D:/Opne CV/face-intelligence/face-intelligence/models"

for name in ["genderage.onnx", "fairface.onnx"]:
    m = onnx.load(MODELS + "/" + name)
    print("=" * 60)
    print(name, "IR:", m.ir_version, "opset:", [o.version for o in m.opset_import])
    print("  producer:", m.producer_name, m.producer_version)
    print("  meta props:", [p.name for p in m.metadata_props])
    for p in m.metadata_props:
        print("     ", p.name, "=", p.value[:200] if len(p.value) < 200 else p.value[:200] + "...")
    # initializers summary (weights) - just count + first-few stats
    inits = m.graph.initializer
    print("  #initializers:", len(inits))
    for init in inits[:6]:
        arr = numpy_helper.to_array(init)
        print("     ", init.name, arr.shape, arr.dtype, "min", float(arr.min()), "max", float(arr.max()))


# Also: what does the genderage model's *last* linear weight look like?
# If the age output is a direct regression head, its weights tell us the scale.
m = onnx.load(MODELS + "/genderage.onnx")
for init in m.graph.initializer:
    arr = numpy_helper.to_array(init)
    if arr.ndim == 2 and 3 in arr.shape:
        print("genderage 2D init:", init.name, arr.shape, "mean", float(arr.mean()), "std", float(arr.std()))