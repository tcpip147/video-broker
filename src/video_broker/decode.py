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
    """Decode packets with PyNvVideoCodec NVDEC; frames stay on the GPU.

    ``is_hw_owned=False`` downloads decoded frames to system memory so the
    existing Python model and the CPU/GPU encoders can consume them normally.
    """
    import PyNvVideoCodec as nvc
    decoder = None
    input_stream = None
    packet_index = 0
    frame_index = 0
    last_frame_pts = None

    for input_stream, packet in packets:
        if decoder is None:
            decoder = nvc.CreateDecoder(
                gpuid=0,
                codec=input_stream["codec"],
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.NATIVE,
                # NATIVE keeps NVDEC's display-order reorder buffer enabled.
                # LOW can expose decode order for streams containing B-frames.
                latency=nvc.DisplayDecodeLatencyType.NATIVE,
            )

        decoded_frames = decoder.Decode(packet)
        frame_pts_list = [frame.getPTS() for frame in decoded_frames]
        logger.info(
            "TRACE DEC packet=%d input_pts=%s input_dts=%s decoded=%d frame_pts=%s",
            packet_index, packet.pts, packet.dts,
            len(decoded_frames), frame_pts_list,
        )
        packet_index += 1

        for frame, frame_pts in zip(decoded_frames, frame_pts_list):
            if last_frame_pts is not None and frame_pts <= last_frame_pts:
                logger.warning(
                    "TRACE DEC_ORDER frame=%d pts=%s last_pts=%s",
                    frame_index,
                    frame_pts,
                    last_frame_pts,
                )
            last_frame_pts = frame_pts
            logger.info(
                "TRACE DEC_OUT frame=%d pts=%s source_packet=%d",
                frame_index, frame_pts, packet_index - 1,
            )
            frame_index += 1
            yield input_stream, frame, frame_pts

    if decoder is not None:
        for frame in decoder.Flush():
            yield input_stream, frame, time.monotonic()
