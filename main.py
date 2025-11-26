import base64
import cv2
import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

# =========================
# YOLO MODEL
# =========================
model = YOLO("yolov8x.pt")

# =========================
# CONFIG (زي ما كنت عامل)
# =========================
config = {
    "detect": "both",   # person / phone / both
    "mode": "active"    # active / passive
}

WORKFLOW_URL = "https://training.aimicromind.com/api/v1/prediction/5b3ec798-a462-4ee4-ae3f-a410220f80e5"

phone_event_triggered = False


# =========================
# SCHEMAS
# =========================
class DetectPayload(BaseModel):
    detect: str


class ModePayload(BaseModel):
    mode: str


class ImagePayload(BaseModel):
    image: str   # data URL من المتصفح


# =========================
# HELPERS
# =========================
def send_update(persons: int, phones: int):
    try:
        requests.post(
            WORKFLOW_URL,
            json={"question": f"persons={persons}, phones={phones}"}
        )
    except:
        # م نعملش حاجة لو فشل
        pass


# =========================
# CONFIG ENDPOINTS (نفس القديم)
# =========================
@app.post("/set-detect")
def set_detect(payload: DetectPayload):
    config["detect"] = payload.detect
    return {
        "status": "ok",
        "message": f"detect mode set to {payload.detect}"
    }


@app.post("/set-mode")
def set_mode(payload: ModePayload):
    config["mode"] = payload.mode
    return {
        "status": "ok",
        "message": f"mode set to {payload.mode}"
    }


@app.get("/config")
def get_config():
    return config


@app.post("/reset")
def reset():
    config["detect"] = "both"
    config["mode"] = "active"
    return {
        "status": "ok",
        "message": "config reset"
    }


# =========================
# YOLO PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(payload: ImagePayload):
    global phone_event_triggered

    data_url = payload.image

    # لو المود مش active رجّع الصورة زي ما هي
    if config["mode"] != "active":
        return {"image": data_url}

    # ===== Decode image =====
    img_bytes = base64.b64decode(data_url.split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ===== YOLO =====
    results = model(img)[0]

    persons = 0
    phones = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            persons += 1
        elif cls == 67:
            phones += 1

    # ===== APPLY CONFIG DETECT =====
    if config["detect"] == "person":
        phones = 0
    elif config["detect"] == "phone":
        persons = 0

    # ===== EVENT LOGIC (زي كودك القديم) =====
    if persons > 0 and phones > 0 and not phone_event_triggered:
        send_update(persons, phones)
        phone_event_triggered = True

    if phones == 0 or persons == 0:
        phone_event_triggered = False

    # ===== RETURN ANNOTATED IMAGE =====
    annotated = results.plot()
    _, buf = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buf).decode()

    return {"image": f"data:image/jpeg;base64,{b64}"}
