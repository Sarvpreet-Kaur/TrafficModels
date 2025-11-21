from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, traceback, base64, io, os, requests
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from PIL import Image
import numpy as np
from embedder import image_to_embedding

TRAFFIC_API_URL = "https://dynamictrafficalgo-1.onrender.com/update"
MODEL_PATH = "vehicle_classifier.pkl"
LABEL_PATH = "label_encoder.pkl"

app = FastAPI(title="ML -> Traffic API")

# Enable CORS for browser dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["https://your-dashboard.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and encoder (best-effort)
classifier = None
label_encoder = None
load_errors = []

if os.path.exists(LABEL_PATH):
    try:
        label_encoder = joblib.load(LABEL_PATH)
        print("Loaded label encoder")
    except Exception as e:
        load_errors.append(f"label load: {e}")

if os.path.exists(MODEL_PATH):
    try:
        classifier = joblib.load(MODEL_PATH)
        print("Loaded classifier")
    except Exception as e:
        load_errors.append(f"classifier load: {e}")

# Pydantic models for input
class DetectedBox(BaseModel):
    lane_id: str
    pred_label: Optional[str] = None        # if client already classified
    embedding: Optional[List[float]] = None # optional embedding vector
    crop_base64: Optional[str] = None       # optional: base64 jpeg/ png of crop

class DetectionPayload(BaseModel):
    detections: List[DetectedBox]

# helper utils
def decode_label(pred):
    try:
        if label_encoder is None:
            return str(pred)
        return label_encoder.inverse_transform([pred])[0]
    except Exception:
        return str(pred)

def classify_from_embedding(emb):
    # emb: 1D numpy array or list
    if classifier is None:
        raise RuntimeError("Classifier not loaded")
    arr = np.array(emb).reshape(1, -1)
    pred = classifier.predict(arr)[0]
    # if classifier returns numeric, decode via label encoder else return string
    return decode_label(pred)

def aggregate_counts(detections):
    counts = {}
    for d in detections:
        lane = d.get("lane_id")
        if lane is None:
            continue
        if lane not in counts:
            counts[lane] = {"normal": 0, "emergency": 0}

        label = None
        if d.get("pred_label"):
            label = d["pred_label"]
        elif d.get("embedding") is not None:
            try:
                label = classify_from_embedding(d["embedding"])
            except Exception:
                label = None
        elif d.get("crop_base64"):
            try:
                img_bytes = base64.b64decode(d["crop_base64"])
                pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                emb = image_to_embedding(pil)
                label = classify_from_embedding(emb)
            except Exception:
                label = None
        else:
            label = None

        if label:
            lab = str(label).lower()
            if any(x in lab for x in ("ambulance","emergency","police","fire")):
                counts[lane]["emergency"] += 1
            else:
                counts[lane]["normal"] += 1
        else:
            counts[lane]["normal"] += 1

    # format list of dicts for traffic API
    out = []
    for lane, v in counts.items():
        out.append({"lane_id": lane, "normal": int(v["normal"]), "emergency": int(v["emergency"])})
    return out

@app.post("/predict_and_update")
def predict_and_update(payload: DetectionPayload):
    if load_errors:
        # still proceed but warn
        warning = {"load_errors": load_errors}
    else:
        warning = None

    detections = [d.dict() for d in payload.detections]
    lane_input = aggregate_counts(detections)

    # POST to traffic controller; increase timeout for sleeping services
    try:
        resp = requests.post(TRAFFIC_API_URL, json=lane_input, timeout=60)
        resp.raise_for_status()
        controller_resp = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error": str(e), "lane_input": lane_input})

    result = {"ml_counts": lane_input, "controller_response": controller_resp}
    if warning:
        result["warning"] = warning
    return result

# simple test endpoint
@app.get("/status")
def status():
    return {"classifier_loaded": classifier is not None, "encoder_loaded": label_encoder is not None, "load_errors": load_errors}
