"""GPU-resident NV12 YOLO inference and overlay example."""
from __future__ import annotations
import os
from typing import Any
import torch
from torch.nn import functional as F
from ultralytics import YOLO

_model: YOLO | None = None
_last_boxes: torch.Tensor | None = None

class Nv12CudaFrame:
    def __init__(self, y: torch.Tensor, uv: torch.Tensor) -> None:
        height, width = y.shape
        self.data = torch.cat((y, uv), dim=0).contiguous()
        self.y = self.data[:height].view(height, width, 1)
        self.uv = self.data[height:].view(height // 2, width // 2, 2)

    def cuda(self) -> list[torch.Tensor]:
        return [self.y, self.uv]

def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-capable PyTorch is required")
        _model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))
        _model.to("cuda:0")
    return _model

def _nv12_to_rgb(frame: Any) -> torch.Tensor:
    nv12 = torch.from_dlpack(frame)
    if nv12.ndim != 2:
        nv12 = nv12.reshape(frame.shape)
    height, width = nv12.shape[0] * 2 // 3, nv12.shape[1]
    y = nv12[:height].float()
    uv = nv12[height:].reshape(height // 2, width // 2, 2).float()
    u = F.interpolate(uv[..., 0][None, None], (height, width), mode="nearest")[0, 0]
    v = F.interpolate(uv[..., 1][None, None], (height, width), mode="nearest")[0, 0]
    c, d, e = y - 16, u - 128, v - 128
    return torch.stack((1.164*c+1.596*e, 1.164*c-0.392*d-0.813*e, 1.164*c+2.017*d)).clamp_(0, 255).to(torch.uint8)

def _draw_boxes(rgb: torch.Tensor, boxes: torch.Tensor, thickness: int = 2) -> None:
    _, height, width = rgb.shape
    green = rgb.new_tensor([0, 255, 0])[:, None, None]
    for coordinates in boxes.round().to(torch.int64):
        x1, y1, x2, y2 = (int(value) for value in coordinates)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1: continue
        rgb[:, y1:y1+thickness, x1:x2] = green
        rgb[:, y2-thickness:y2, x1:x2] = green
        rgb[:, y1:y2, x1:x1+thickness] = green
        rgb[:, y1:y2, x2-thickness:x2] = green

def _rgb_to_nv12(rgb: torch.Tensor) -> Nv12CudaFrame:
    r, g, b = rgb.float().unbind(0)
    y = (0.257*r + 0.504*g + 0.098*b + 16).clamp(0, 255)
    u = (-0.148*r - 0.291*g + 0.439*b + 128).clamp(0, 255)
    v = (0.439*r - 0.368*g - 0.071*b + 128).clamp(0, 255)
    u = F.avg_pool2d(u[None, None], 2, 2)[0, 0]
    v = F.avg_pool2d(v[None, None], 2, 2)[0, 0]
    uv = torch.stack((u, v), -1).reshape(u.shape[0], -1)
    return Nv12CudaFrame(y.to(torch.uint8), uv.to(torch.uint8))

@torch.inference_mode()
def on_frame(frame: Any, *, infer: bool = True) -> Nv12CudaFrame:
    global _last_boxes
    rgb = _nv12_to_rgb(frame)
    if infer:
        result = _get_model().predict(source=rgb.float().div_(255).unsqueeze(0), device="cuda:0", classes=[2], conf=float(os.getenv("YOLO_CONF", "0.35")), verbose=False)[0]
        _last_boxes = result.boxes.xyxy.detach() if result.boxes is not None else torch.empty((0, 4), device=rgb.device)
    if _last_boxes is not None:
        _draw_boxes(rgb, _last_boxes)
    return _rgb_to_nv12(rgb)
