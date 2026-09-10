import av
import logging
import time
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
) -> Iterator[tuple[av.video.stream.VideoStream, av.Packet]]:
    pass
