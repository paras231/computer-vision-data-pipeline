import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

app = FastAPI()

print(f"cpu count: {os.cpu_count()}")

CPU_COUNT = os.cpu_count()

MODEL_PATH = "best.onnx"


IMG_SIZE = 640
CLASS_NAMES = [
    "Automatic Rifle",
    "Bazooka",
    "Grenade Launcher",
    "Handgun",
    "Knife",
    "Shotgun",
    "SMG",
    "Sniper",
    "Sword",
]  

# ---- Load session once at startup ----
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = CPU_COUNT  # tune to your CPU core count
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


outputs = session.run(None, {input_name: np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)})
print("Number of outputs:", len(outputs))
for i, o in enumerate(outputs):
    print(f"Output {i} shape: {o.shape}")
print(outputs[0][:5])  # peek at first 5 rows


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    pad_h, pad_w = new_shape - nh, new_shape - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def preprocess(img):
    img_letterboxed, r, (pad_x, pad_y) = letterbox(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch, r, pad_x, pad_y


def postprocess(output, r, pad_x, pad_y, orig_shape, conf_thres=0.25):
    try:
        dets = output[0][0]  # shape (1, 300, 6) -> (300, 6)
        results = []
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            if conf < conf_thres:
                continue
            x1 = (x1 - pad_x) / r
            y1 = (y1 - pad_y) / r
            x2 = (x2 - pad_x) / r
            y2 = (y2 - pad_y) / r
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_shape[1], x2), min(orig_shape[0], y2)
            results.append({
                "box": [round(float(x1), 2), round(float(y1), 2),
                        round(float(x2), 2), round(float(y2), 2)],
                "confidence": round(float(conf), 4),
                "class_id": int(cls),
                "class_name": CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(int(cls))
            })
        return results
    except Exception as e:
        print(f"Error in postprocess: {e}")
        return []

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    input_tensor, r, pad_x, pad_y = preprocess(img)
    outputs = session.run(None, {input_name: input_tensor})
    detections = postprocess(outputs, r, pad_x, pad_y, img.shape, conf_thres=0.25)

    return {"detections": detections}


@app.get("/health")
def health():
    return {"status": "ok"}