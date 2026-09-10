import av
import logging
import time
import ctypes
from collections.abc import Iterator

logger = logging.getLogger(__name__)


def receive_cpu_packets(
    url: str,
    transport: str,
) -> Iterator[tuple[av.video.stream.VideoStream, av.Packet]]:
    while True:
        container = None

        try:
            print(f"RTSP 연결 시도: {url}")

            container = av.open(
                url,
                mode="r",
                options={
                    "rtsp_transport": transport,
                    "rw_timeout": "5000000",
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "max_delay": "0",
                    "reorder_queue_size": "0",
                    "probesize": "32",
                    "analyzeduration": "0",
                },
                timeout=(5.0, 5.0),
            )

            print("RTSP 연결 성공")

            input_stream = container.streams.video[0]

            for packet in container.demux(video=0):
                if packet.dts is not None:
                    yield input_stream, packet

        except (av.FFmpegError, OSError) as error:
            print(f"RTSP 연결 끊김: {error}")

        finally:
            if container is not None:
                container.close()

        delay = 3
        print(f"{delay}초 후 재접속합니다.")
        time.sleep(delay)


def receive_cuda_packets(
    url: str,
    transport: str,
) -> Iterator[tuple[dict[str, object], object]]:
    import PyNvVideoCodec as nvc

    codec_ids = {
        "h264": nvc.cudaVideoCodec.H264,
        "hevc": nvc.cudaVideoCodec.HEVC,
        "h265": nvc.cudaVideoCodec.HEVC,
    }
    container = av.open(
        url,
        mode="r",
        options={
            "rtsp_transport": transport,
            "rw_timeout": "5000000",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "max_delay": "0",
            "reorder_queue_size": "0",
        },
        timeout=(5.0, 5.0),
    )
    try:
        input_stream = container.streams.video[0]
        codec_name = input_stream.codec_context.name.lower()
        if codec_name not in codec_ids:
            raise ValueError(f"Unsupported NVDEC codec: {codec_name}")
        stream = {
            "width": input_stream.width,
            "height": input_stream.height,
            "fps": float(input_stream.average_rate or 30.0),
            "codec": codec_ids[codec_name],
            "time_base_num": input_stream.time_base.numerator,
            "time_base_den": input_stream.time_base.denominator,
        }
        packet_index = 0
        for source_packet in container.demux(video=0):
            if source_packet.dts is None or not source_packet:
                continue
            logger.info(
                "TRACE RX packet=%d pts=%s dts=%s duration=%s key=%s bytes=%d",
                packet_index, source_packet.pts, source_packet.dts,
                source_packet.duration, source_packet.is_keyframe,
                source_packet.size,
            )
            packet = nvc.PacketData()
            bitstream = ctypes.create_string_buffer(bytes(source_packet))
            packet.bsl_data = ctypes.addressof(bitstream)
            packet.bsl = len(bitstream) - 1
            packet.pts = source_packet.pts or 0
            packet.dts = source_packet.dts or 0
            packet.duration = source_packet.duration or 0
            packet.key = bool(source_packet.is_keyframe)
            packet.is_video = True
            yield stream, packet
            packet_index += 1
    finally:
        container.close()
