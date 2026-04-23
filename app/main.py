import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.main import demo


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



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)