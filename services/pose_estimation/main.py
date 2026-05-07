"""
services/pose_estimation/main.py
FastAPI microservice for skeleton pose estimation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import cv2
import base64

from .estimator import PoseEstimator

app = FastAPI(title="Pose Estimation Service", version="1.0.0")
estimator = PoseEstimator(model_complexity=1)


class PoseRequest(BaseModel):
    frame_base64: str
    min_visibility: float = 0.5


class PoseResponse(BaseModel):
    detected: bool
    all_keypoints: Optional[List[Dict]] = None
    body_keypoints: Optional[Dict[str, Dict]] = None
    num_visible: int = 0


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "pose_estimation"}


@app.post("/estimate-pose", response_model=PoseResponse)
async def estimate_pose(req: PoseRequest):
    """Extract skeleton keypoints from a video frame."""
    try:
        # Decode base64 → numpy
        img_bytes = base64.b64decode(req.frame_base64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Estimate pose
        all_kps = estimator.estimate(rgb)

        if all_kps is None:
            return PoseResponse(detected=False)

        # Filter body keypoints
        body_kps = estimator.get_body_keypoints(all_kps, req.min_visibility)
        num_visible = sum(
            1 for kp in all_kps if kp["visibility"] >= req.min_visibility
        )

        return PoseResponse(
            detected=True,
            all_keypoints=all_kps,
            body_keypoints=body_kps,
            num_visible=num_visible,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown():
    estimator.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
