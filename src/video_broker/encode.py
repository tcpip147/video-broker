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

    codec = av.CodecContext.create("h264_nvenc", "w")
    codec.width = first_frame.width
    codec.height = first_frame.height
    codec.pix_fmt = "yuv420p"
    codec.framerate = fps
    codec.time_base = Fraction(fps.denominator, fps.numerator)
    codec.options = {
        "preset": "p1",
        "tune": "ull",
        "zerolatency": "1",
        "bf": "0",
        "rc-lookahead": "0",
        "delay": "0",
        "g": "30",
        "keyint_min": "30",
        "sc_threshold": "0",
        "profile": "baseline",
        "repeat-headers": "1",
        "forced-idr": "1",
    }

    last_pts = None
    for sequence, (_, frame, received_at) in enumerate(
        chain(((input_stream, first_frame, first_received_at),), iterator)
    ):
        source_pts = frame.pts
        source_time_base = frame.time_base
        pts = round((received_at - first_received_at) * float(fps))
        if last_pts is not None and pts <= last_pts:
            pts = last_pts + 1
        new_frame = frame.reformat(
            width=codec.width,
            height=codec.height,
            format="yuv420p",
        )
        new_frame.pts = pts
        new_frame.time_base = codec.time_base
        last_pts = pts

        for packet in codec.encode(new_frame):
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
