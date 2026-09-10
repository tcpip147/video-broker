"""CPU VideoFrame in/out, CUDA YOLO inference example."""

import os

import av
import cv2
import torch
from ultralytics import YOLO


_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        model_path = os.getenv("YOLO_MODEL", "yolo11n.pt")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "YOLO CUDA 추론을 사용할 수 없습니다: CUDA 장치를 찾지 못했습니다."
            )
        _model = YOLO(model_path)
        _model.to("cuda:0")
    return _model


def on_frame(frame: av.VideoFrame) -> av.VideoFrame:
    model = _get_model()

    # 입력은 CPU 메모리의 VideoFrame이며, BGR ndarray로 변환한 뒤
    # Ultralytics가 CUDA로 입력을 이동시켜 추론한다.
    image = frame.to_ndarray(format="bgr24")
    results = model.predict(
        source=image,
        device="cuda:0",
        classes=[2],  # COCO class 2 = car
        conf=float(os.getenv("YOLO_CONF", "0.35")),
        verbose=False,
    )

    result = results[0]
    if result.boxes is not None:
        for box in result.boxes.xyxy.detach().cpu().numpy().astype(int):
            x1, y1, x2, y2 = box.tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output = av.VideoFrame.from_ndarray(image, format="bgr24")
    output.pts = frame.pts
    output.time_base = frame.time_base
    return output
