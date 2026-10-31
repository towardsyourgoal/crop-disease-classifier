import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "mobilenet_v3_small.onnx"

# label-id order used in training (sorted list of the 12 selected classes)
CLASSES = [
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

# same as the eval transform: Resize((224, 224)) -> ToTensor -> Normalize(ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BYTES = 4 * 1024 * 1024  # Vercel request body limit is 4.5 MB

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

app = FastAPI(title="Crop Disease Classifier")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # HWC -> NCHW


def pretty(name: str) -> dict:
    crop, _, condition = name.partition("___")
    return {
        "crop": crop.replace("_", " ").strip(),
        "condition": condition.replace("_", " ").strip(),
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "classes": CLASSES}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 4 MB)")
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, "File is not a valid image")

    logits = session.run(None, {"image": preprocess(img)})[0][0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    order = np.argsort(-probs)[: max(1, min(top_k, len(CLASSES)))]
    top = [{"label": CLASSES[i], **pretty(CLASSES[i]), "confidence": float(probs[i])} for i in order]
    return {"prediction": top[0], "top_k": top}
