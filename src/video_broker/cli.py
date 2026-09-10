from pathlib import Path
import logging
import os
import time
from queue import Queue
from typing import Any
from queue import Empty, Full, Queue
from threading import Thread

import yaml
import av

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import importlib.util
from pathlib import Path
from types import ModuleType

from video_broker.receive import (
    receive_cpu_packets,
    receive_cuda_packets,
)
from video_broker.decode import decode_cuda, decode_cpu
from video_broker.encode import encode_cuda, encode_cpu
from video_broker.send import send_cpu_packets, send_cuda_packets


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_module(path: str | Path) -> ModuleType:
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location(
        path.stem,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def configure_cuda_dll_path() -> None:
    """Make CUDA 13.x DLLs discoverable by PyNvVideoCodec on Windows."""
    if os.name != "nt":
        return
    cuda_paths = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_PATH_V12_6"),
    ]
    for cuda_path in filter(None, cuda_paths):
        for directory in (
            Path(cuda_path) / "bin",
            Path(cuda_path) / "bin" / "x64",
        ):
            if directory.is_dir() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(directory))


def put_latest(
    queue: Queue[tuple[av.video.stream.VideoStream, av.VideoFrame, float]],
    frame: tuple[av.video.stream.VideoStream, av.VideoFrame, float],
) -> None:
    try:
        queue.put_nowait(frame)
    except Full:
        try:
            dropped_stream, dropped_frame, dropped_received_at = queue.get_nowait()
            logger.warning(
                "frame dropped: queue full, pts=%s, received_age=%.3f sec",
                dropped_frame.pts,
                time.monotonic() - dropped_received_at,
            )
        except Empty:
            pass

        try:
            queue.put_nowait(frame)
        except Full:
            logger.warning(
                "frame dropped: queue remained full while inserting latest frame"
            )


def main() -> None:
    config = load_config("videep.yml")

    input_rtsp = config["input"]["rtsp"]
    input_url = input_rtsp["url"]
    input_transport = input_rtsp["transport"]

    output_rtsp = config["output"]["rtsp"]
    output_url = output_rtsp["url"]
    output_transport = output_rtsp["transport"]

    backend = config["backend"]
    if backend not in {"cpu", "cuda"}:
        raise ValueError(f"지원하지 않는 backend: {backend}")
    inference_interval = int(config.get("inference_interval", 1))
    if inference_interval < 1:
        raise ValueError("inference_interval은 1 이상의 정수여야 합니다.")

    model = config["model"]
    model_module = load_module(model)

    frame_queue: Queue[tuple[av.video.stream.VideoStream, av.VideoFrame, float]] = (
        Queue(maxsize=3)
    )

    def decode_worker(packets) -> None:
        frames = decode_cpu(packets)

        for frame in frames:
            put_latest(frame_queue, frame)

    def process_worker() -> None:
        def processed_frames():
            frame_index = 0
            while True:
                input_stream, frame, received_at = frame_queue.get()
                try:
                    while True:
                        try:
                            input_stream, frame, received_at = frame_queue.get_nowait()
                            frame_queue.task_done()
                        except Empty:
                            break

                    should_infer = frame_index % inference_interval == 0
                    if model_module is not None:
                        frame = model_module.on_frame(frame, infer=should_infer)

                    frame_index += 1

                    if frame is not None:
                        yield input_stream, frame, received_at
                finally:
                    frame_queue.task_done()

        frames = processed_frames()
        packets = encode_cpu(frames)
        send_cpu_packets(packets, output_url, output_transport)

    if backend == "cpu":
        packets = receive_cpu_packets(input_url, input_transport)

        decode_thread = Thread(target=decode_worker, args=(packets,), daemon=True)
        process_thread = Thread(target=process_worker, daemon=True)

        decode_thread.start()
        process_thread.start()

        decode_thread.join()
        process_thread.join()

    elif backend == "cuda":
        configure_cuda_dll_path()
        packets = receive_cuda_packets(input_url, input_transport)

        frames = decode_cuda(packets)

        def processed_frames():
            frame_index = 0
            for input_stream, frame, received_at in frames:
                should_infer = frame_index % inference_interval == 0
                processed_frame = model_module.on_frame(frame, infer=should_infer)
                frame_index += 1

                if processed_frame is not None:
                    yield input_stream, processed_frame, received_at

        packets = encode_cuda(processed_frames())
        send_cuda_packets(packets, output_url, output_transport)


if __name__ == "__main__":
    main()
