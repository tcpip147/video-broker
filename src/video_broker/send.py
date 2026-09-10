from fractions import Fraction
from itertools import chain
import logging
import time

import av

logger = logging.getLogger(__name__)


def send_cpu_packets(
    packets,
    output_url: str,
    transport: str = "tcp",
    fps: Fraction | None = None,
) -> None:
    iterator = iter(packets)

    try:
        input_stream, first_packet, first_received_at = next(iterator)
    except StopIteration:
        return

    target = av.open(
        output_url,
        mode="w",
        format="rtsp",
        options={
            "rtsp_transport": transport,
            "muxdelay": "0",
            "muxpreload": "0",
            "flush_packets": "1",
        },
        timeout=(5.0, 5.0),
    )

    if fps is None:
        fps = input_stream.average_rate or Fraction(30, 1)
        fps = Fraction(fps.numerator, fps.denominator)
        logger.info("output stream FPS: %s", fps)

    output_stream = target.add_mux_stream(
        input_stream.codec_context.name,
        rate=fps,
        width=input_stream.width,
        height=input_stream.height,
        time_base=Fraction(fps.denominator, fps.numerator),
    )

    try:
        for _, packet, received_at in chain(
            ((input_stream, first_packet, first_received_at),), iterator
        ):
            packet.stream = output_stream
            target.mux(packet)
    finally:
        target.close()


def send_cuda_packets():
    pass
