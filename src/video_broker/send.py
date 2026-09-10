from fractions import Fraction
from itertools import chain
import logging
import time

import av


def _encoded_bytes(encoded) -> bytes:
    if isinstance(encoded, (bytes, bytearray, memoryview)):
        return bytes(encoded)
    if isinstance(encoded, dict):
        for key in ("bitstream", "data", "packet", "encoded_data", "payload"):
            if key in encoded:
                value = encoded[key]
                if isinstance(value, (bytes, bytearray, memoryview)):
                    return bytes(value)
                if isinstance(value, (list, tuple)):
                    return bytes(value)
        for value in encoded.values():
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value)
            if isinstance(value, (dict, list, tuple)):
                try:
                    return _encoded_bytes(value)
                except TypeError:
                    continue
    if isinstance(encoded, (list, tuple)):
        if all(isinstance(value, int) for value in encoded):
            return bytes(encoded)
        for value in encoded:
            try:
                return _encoded_bytes(value)
            except TypeError:
                continue
    raise TypeError(
        f"Unsupported PyNvVideoCodec encoded packet type: {type(encoded).__name__}"
    )


def _encoded_timestamp(encoded, name: str):
    if isinstance(encoded, dict):
        value = encoded.get(name)
        if value is None and name == "pts":
            value = encoded.get("timestamp")
        if isinstance(value, int):
            return value
    return None


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


def send_cuda_packets(packets, output_url: str, transport: str = "tcp") -> None:
    iterator = iter(packets)
    try:
        stream, first_packet, first_source_pts = next(iterator)
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
    output_fps = Fraction(str(stream["fps"]))
    output_stream = target.add_stream("h264", rate=output_fps)
    output_stream.width, output_stream.height = stream["width"], stream["height"]
    output_stream.time_base = Fraction(output_fps.denominator, output_fps.numerator)
    packet_index = 0
    try:
        for _, encoded, _ in chain(((stream, first_packet, first_source_pts),), iterator):
            packet = av.Packet(_encoded_bytes(encoded))
            packet.stream = output_stream
            packet.pts = packet_index
            packet.dts = packet_index
            packet.duration = 1
            packet.time_base = output_stream.time_base
            packet_index += 1
            target.mux(packet)
    finally:
        target.close()
