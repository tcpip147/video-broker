from collections.abc import Iterable, Iterator
import logging
import time

import av
from av.codec.hwaccel import HWAccel


logger = logging.getLogger(__name__)


def decode_cpu(
    packets: Iterator[tuple[av.video.stream.VideoStream, av.Packet]],
) -> Iterator[tuple[av.video.stream.VideoStream, av.VideoFrame, float]]:
    codec = None
    input_stream = None

    for input_stream, packet in packets:
        if codec is None:
            codec = input_stream.codec_context

        for frame in codec.decode(packet):
            yield input_stream, frame, time.monotonic()


def decode_cuda(
    packets: Iterator[tuple[av.video.stream.VideoStream, av.Packet]],
) -> Iterator[tuple[av.video.stream.VideoStream, av.VideoFrame, float]]:
    """Decode H.264 packets with FFmpeg's CUDA hardware acceleration.

    ``is_hw_owned=False`` downloads decoded frames to system memory so the
    existing Python model and the CPU/GPU encoders can consume them normally.
    """
    codec = None
    input_stream = None

    for input_stream, packet in packets:
        if codec is None:
            codec_name = input_stream.codec_context.name
            codec = av.CodecContext.create(
                codec_name,
                "r",
                hwaccel=HWAccel(
                    "cuda",
                    allow_software_fallback=False,
                    is_hw_owned=True,
                ),
            )
            if input_stream.codec_context.extradata:
                codec.extradata = input_stream.codec_context.extradata
            logger.info("CUDA decoder initialized: %s", codec_name)

        for frame in codec.decode(packet):
            yield input_stream, frame, time.monotonic()

    if codec is not None:
        for frame in codec.decode(None):
            yield input_stream, frame, time.monotonic()
