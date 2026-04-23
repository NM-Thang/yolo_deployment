import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.main import demo
from fastapi import WebSocket
import base64
import json
import cv2



app = FastAPI()

# Dictionary to keep track of active FFmpeg processes by stream_id
active_streams = {}

# Data models for request validation
class RestreamRequest(BaseModel):
    stream_id: str
    source_url: str
    destination_urls: list[str]

class StopRequest(BaseModel):
    stream_id: str

@app.get("/video")
def video_feed():
    return StreamingResponse(demo(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
    for frame, event in demo():  # demo() phải yield (frame, event)
        # Encode frame to JPEG base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        data = {
            "frame": jpg_as_text,
            "event": event  # hoặc list event, text, ...
        }
        await websocket.send_text(json.dumps(data))
    await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)