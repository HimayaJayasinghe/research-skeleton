"""
services/video_processing/main.py
FastAPI microservice for video frame preprocessing.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

from .processor import VideoProcessor

app = FastAPI(title="Video Processing Service", version="1.0.0")
processor = VideoProcessor()


class FrameRequest(BaseModel):
    frame_base64: str
    target_width: int = 640
    target_height: int = 480


class FrameResponse(BaseModel):
    processed_frame_base64: str
    original_shape: list
    processed_shape: list


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "video_processing"}


@app.post("/process-frame", response_model=FrameResponse)
async def process_frame(req: FrameRequest):
    """Preprocess a video frame for pose estimation."""
    try:
        # Decode base64 → BGR numpy
        frame = VideoProcessor.base64_to_frame(req.frame_base64)
        original_shape = list(frame.shape)

        # Update target size if specified
        processor.target_size = (req.target_width, req.target_height)

        # Preprocess (resize + BGR→RGB + CLAHE)
        processed = processor.preprocess_frame(frame)
        processed_shape = list(processed.shape)

        # Re-encode to base64 (convert RGB back to BGR for JPEG encoding)
        import cv2
        bgr_out = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        out_b64 = VideoProcessor.frame_to_base64(bgr_out)

        return FrameResponse(
            processed_frame_base64=out_b64,
            original_shape=original_shape,
            processed_shape=processed_shape,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
