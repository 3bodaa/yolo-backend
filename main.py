from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

config = {
    "detect": "both",
    "mode": "active"
}

class DetectPayload(BaseModel):
    detect: str

class ModePayload(BaseModel):
    mode: str

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
