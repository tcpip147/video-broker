from collections.abc import Iterable, Iterator
from fractions import Fraction
from itertools import chain
import logging

import av


logger = logging.getLogger(__name__)


def get_stream_fps(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate
    if rate is None or rate <= 0:
        logger.warning("입력 스트림 FPS를 확인할 수 없어 30 FPS를 사용합니다.")
        return Fraction(30, 1)
    return Fraction(rate.numerator, rate.denominator)


def encode_cpu(
    frames: Iterable[tuple[av.video.stream.VideoStream, av.VideoFrame, float]],
    fps: Fraction | None = None,
) -> Iterator[tuple[av.video.stream.VideoStream, av.Packet, float]]:
    iterator = iter(frames)

    try:
        input_stream, first_frame, first_received_at = next(iterator)
    except StopIteration:
        return

    if fps is None:
        fps = get_stream_fps(input_stream)
    logger.info("output FPS: %s", fps)

    codec = av.CodecContext.create("libx264", "w")
    codec.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "keyint": "15",
        "min-keyint": "15",
        "scenecut": "0",
        "repeat-headers": "1",
    }
    codec.width = first_frame.width
    codec.height = first_frame.height
    codec.pix_fmt = "yuv420p"
    codec.framerate = fps
    codec.time_base = Fraction(fps.denominator, fps.numerator)
    frame_count = 0

    last_pts = None
    for sequence, (_, frame, received_at) in enumerate(
        chain(((input_stream, first_frame, first_received_at),), iterator)
    ):
        source_pts = frame.pts
        source_time_base = frame.time_base
        frame_count += 1
        pts = round((received_at - first_received_at) * float(fps))
        if last_pts is not None and pts <= last_pts:
            pts = last_pts + 1
        frame = frame.reformat(
            width=codec.width,
            height=codec.height,
            format="yuv420p",
        )
        frame.pts = pts
        frame.time_base = codec.time_base
        last_pts = pts

        for packet in codec.encode(frame):
            source_seconds = (
                float(source_pts * source_time_base)
                if source_pts is not None and source_time_base is not None
                else None
            )
            packet_seconds = (
                float(packet.pts * packet.time_base)
                if packet.pts is not None and packet.time_base is not None
                else None
            )
            yield input_stream, packet, received_at

    for packet in codec.encode(None):
        yield input_stream, packet, first_received_at


def encode_cuda(
    frames: Iterable[tuple[object, object, float]],
    fps: Fraction | None = None,
) -> Iterator[tuple[object, object, float]]:
    import PyNvVideoCodec as nvc
    import torch
    iterator = iter(frames)

    try:
        input_stream, first_frame, first_received_at = next(iterator)
    except StopIteration:
        return

    frame_rate = float(fps or input_stream["fps"])
    codec = nvc.CreateEncoder(
        input_stream["width"], input_stream["height"], "NV12", False,
        gpu_id=0,
        codec="h264",
        fps=frame_rate,
        bf=0,
    )
    frame_index = 0
    synchronization_logged = False
    for stream, frame, source_pts in chain(
        ((input_stream, first_frame, first_received_at),), iterator
    ):
        torch.cuda.current_stream().synchronize()
        if not synchronization_logged:
            logger.info("CUDA stream synchronized before NVENC input")
            synchronization_logged = True
        encoded_packets = codec.Encode(frame)
        for packet_index, encoded_packet in enumerate(encoded_packets):
            data = encoded_packet.get("data", b"")
            yield stream, encoded_packet, source_pts
        frame_index += 1

    flushed_packets = codec.EndEncode()
    logger.info("NVENC flush encoded=%d", len(flushed_packets))
    for packet_index, encoded_packet in enumerate(flushed_packets):
        data = encoded_packet.get("data", b"")
        logger.info(
            "NVENC flush packet=%d timestamp=%s picture_type=%s bytes=%d",
            packet_index,
            encoded_packet.get("timestamp"),
            encoded_packet.get("picture_type"),
            len(data),
        )
        yield input_stream, encoded_packet, source_pts
