from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Literal, Optional, Tuple
import time

import cv2
import depthai as dai
import numpy as np

Size = Tuple[int, int]
Roi = Tuple[int, int, int, int]
SocketName = Literal["left", "right"]


@dataclass(frozen=True)
class DepthCropGeometry:
    """Geometry for the cropped stereo/depth processing area."""

    full_w: int
    full_h: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    proc_x1: int
    proc_y1: int
    proc_x2: int
    proc_y2: int
    proc_w: int
    proc_h: int
    inner_x1: int
    inner_y1: int
    inner_x2: int
    inner_y2: int


def snap_width_crop_to_multiple_of_16(x1: int, x2: int, full_w: int) -> Tuple[int, int]:
    """Stereo input widths must be multiples of 16 on this DepthAI path."""
    width = int(x2 - x1)
    snapped_width = max(16, (width // 16) * 16)

    center = (x1 + x2) / 2.0
    new_x1 = int(round(center - snapped_width / 2.0))
    new_x2 = new_x1 + snapped_width

    if new_x1 < 0:
        new_x1 = 0
        new_x2 = snapped_width
    if new_x2 > full_w:
        new_x2 = full_w
        new_x1 = full_w - snapped_width

    return new_x1, new_x2


def compute_depth_crop_geometry(
    *,
    full_size: Size = (640, 400),
    roi: Optional[Roi] = (120, 0, 550, 400),
    pad: Tuple[int, int] = (40, 0),
) -> DepthCropGeometry:
    """
    Compute the crop used before StereoDepth.

    The defaults mirror the working camtest.py setup:
    a holder/board ROI is padded before depth is computed, and the crop width is
    snapped to a multiple of 16 so StereoDepth accepts it reliably.
    """
    full_w, full_h = full_size
    if roi is None:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, full_w, full_h
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi

    pad_x, pad_y = pad
    proc_x1 = max(0, roi_x1 - pad_x)
    proc_y1 = max(0, roi_y1 - pad_y)
    proc_x2 = min(full_w, roi_x2 + pad_x)
    proc_y2 = min(full_h, roi_y2 + pad_y)

    proc_x1, proc_x2 = snap_width_crop_to_multiple_of_16(proc_x1, proc_x2, full_w)

    proc_w = proc_x2 - proc_x1
    proc_h = proc_y2 - proc_y1

    return DepthCropGeometry(
        full_w=full_w,
        full_h=full_h,
        roi_x1=roi_x1,
        roi_y1=roi_y1,
        roi_x2=roi_x2,
        roi_y2=roi_y2,
        proc_x1=proc_x1,
        proc_y1=proc_y1,
        proc_x2=proc_x2,
        proc_y2=proc_y2,
        proc_w=proc_w,
        proc_h=proc_h,
        inner_x1=roi_x1 - proc_x1,
        inner_y1=roi_y1 - proc_y1,
        inner_x2=roi_x2 - proc_x1,
        inner_y2=roi_y2 - proc_y1,
    )


def get_latest_packet(queue):
    """Return the newest packet currently in a non-blocking queue."""
    pkt = queue.tryGet()
    if pkt is None:
        return None
    while True:
        newer = queue.tryGet()
        if newer is None:
            break
        pkt = newer
    return pkt


def depth_to_grayscale(
    depth_frame: np.ndarray,
    near_percentile: float = 3.0,
    far_percentile: float = 95.0,
    invert: bool = False,
    dmin: Optional[float] = None,
    dmax: Optional[float] = None,
) -> np.ndarray:
    """Convert a raw depth frame in mm into an 8-bit grayscale image."""
    valid = depth_frame[depth_frame > 0]
    if valid.size == 0:
        return np.zeros(depth_frame.shape, dtype=np.uint8)

    if dmin is None:
        dmin = float(np.percentile(valid, near_percentile))
    if dmax is None:
        dmax = float(np.percentile(valid, far_percentile))
    if dmax <= dmin:
        dmax = dmin + 1.0

    clipped = np.clip(depth_frame.astype(np.float32), dmin, dmax)
    norm = ((clipped - dmin) * 255.0 / (dmax - dmin)).astype(np.uint8)

    if invert:
        norm = 255 - norm

    norm[depth_frame == 0] = 0
    return norm


def median_filter_from_mode(mode: int):
    if mode <= 0:
        return None
    if mode == 1:
        return dai.MedianFilter.KERNEL_3x3
    if mode == 2:
        return dai.MedianFilter.KERNEL_5x5
    return dai.MedianFilter.KERNEL_7x7


class OakSRStreams:
    """
    Reusable OAK-D SR stream provider for other scripts.

    This version intentionally follows the same DepthAI v3 pipeline pattern as
    the user's working camtest.py:
      - Camera nodes on CAM_B and CAM_C
      - RGB888p preview streams for left/right viewing
      - GRAY8 left/right streams into ImageManip crops
      - cropped GRAY8 stereo inputs into StereoDepth
      - queue maxSize=1, blocking=False, latest-packet reads

    Depth is computed on the cropped processing region, then returned as a
    full-size canvas by default so it spatially matches the left/right previews.
    Use get_depth_crop_frame() when you specifically need the raw cropped depth.
    """

    def __init__(
        self,
        *,
        enable_rgb: bool = False,
        rgb_socket: SocketName = "left",
        enable_left: bool = False,
        enable_right: bool = False,
        enable_depth: bool = False,
        enable_depth_gray: bool = False,
        fps: float = 30.0,
        view_size: Size = (640, 400),
        stereo_size: Optional[Size] = None,
        depth_roi: Optional[Roi] = (120, 0, 550, 400),
        depth_pad: Tuple[int, int] = (40, 0),
        align_mode: Literal["left", "center", "centre"] = "centre",
        lr_check: bool = True,
        lr_threshold: int = 6,
        subpixel: bool = True,
        subpixel_bits: int = 5,
        extended_disparity: bool = False,
        confidence: int = 100,
        median_mode: int = 2,
        temporal_filter: bool = True,
        spatial_filter: bool = True,
        speckle_filter: bool = True,
        device_depth_min_mm: int = 360,
        device_depth_max_mm: int = 450,
    ) -> None:
        if enable_rgb:
            if rgb_socket == "left":
                enable_left = True
            else:
                enable_right = True

        self.enable_left = enable_left
        self.enable_right = enable_right
        self.enable_depth = enable_depth or enable_depth_gray
        self.enable_depth_gray = enable_depth_gray
        self.enable_rgb = enable_rgb
        self.rgb_socket = rgb_socket
        self.view_size = view_size
        self.fps = fps

        if not (self.enable_left or self.enable_right or self.enable_depth):
            raise ValueError("Enable at least one stream: rgb, left, right, depth, or depth_gray.")

        if extended_disparity and subpixel:
            # The working control script disables extended disparity when subpixel is enabled.
            extended_disparity = False

        self.depth_geometry = compute_depth_crop_geometry(
            full_size=view_size,
            roi=depth_roi,
            pad=depth_pad,
        )

        self.pipeline = dai.Pipeline()
        try:
            self.pipeline.setXLinkChunkSize(0)
        except Exception:
            pass

        self._queues: Dict[str, object] = {}
        self._cache: Dict[str, np.ndarray] = {}
        self._running = False

        need_left_cam = self.enable_left or self.enable_depth
        need_right_cam = self.enable_right or self.enable_depth

        self._left_cam = (
            self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            if need_left_cam
            else None
        )
        self._right_cam = (
            self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            if need_right_cam
            else None
        )

        if self.enable_left:
            left_out = self._left_cam.requestOutput(
                view_size,
                type=dai.ImgFrame.Type.RGB888p,
                fps=fps,
            )
            self._queues["left"] = left_out.createOutputQueue(maxSize=1, blocking=False)

        if self.enable_right:
            right_out = self._right_cam.requestOutput(
                view_size,
                type=dai.ImgFrame.Type.RGB888p,
                fps=fps,
            )
            self._queues["right"] = right_out.createOutputQueue(maxSize=1, blocking=False)

        if self.enable_depth:
            stereo = self.pipeline.create(dai.node.StereoDepth)
            self._stereo = stereo

            full_w, full_h = view_size if stereo_size is None else stereo_size
            if (full_w, full_h) != view_size:
                # Keep this conservative: the crop geometry and full-frame canvas assume view_size.
                raise ValueError("stereo_size must currently match view_size for aligned full-frame depth output.")

            left_proc_src = self._left_cam.requestOutput(
                view_size,
                type=dai.ImgFrame.Type.GRAY8,
                fps=fps,
            )
            right_proc_src = self._right_cam.requestOutput(
                view_size,
                type=dai.ImgFrame.Type.GRAY8,
                fps=fps,
            )

            left_crop = self.pipeline.create(dai.node.ImageManip)
            right_crop = self.pipeline.create(dai.node.ImageManip)

            try:
                left_crop.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
                right_crop.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            except Exception:
                pass

            g = self.depth_geometry
            left_crop.initialConfig.addCrop(g.proc_x1, g.proc_y1, g.proc_w, g.proc_h)
            right_crop.initialConfig.addCrop(g.proc_x1, g.proc_y1, g.proc_w, g.proc_h)

            try:
                left_crop.setMaxOutputFrameSize(g.proc_w * g.proc_h)
                right_crop.setMaxOutputFrameSize(g.proc_w * g.proc_h)
            except Exception:
                pass

            left_proc_src.link(left_crop.inputImage)
            right_proc_src.link(right_crop.inputImage)
            left_crop.out.link(stereo.left)
            right_crop.out.link(stereo.right)

            stereo.setExtendedDisparity(bool(extended_disparity))
            stereo.setLeftRightCheck(bool(lr_check))
            stereo.setSubpixel(bool(subpixel))

            if subpixel:
                try:
                    stereo.setSubpixelFractionalBits(int(np.clip(subpixel_bits, 3, 5)))
                except Exception:
                    pass

            if align_mode.lower() in ("center", "centre"):
                try:
                    stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)
                except Exception:
                    try:
                        stereo.initialConfig.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.CENTER)
                    except Exception:
                        pass

            try:
                stereo.initialConfig.costMatching.enableSwConfidenceThresholding = True
                stereo.initialConfig.costMatching.confidenceThreshold = int(confidence)
            except Exception:
                try:
                    stereo.initialConfig.setConfidenceThreshold(int(confidence))
                except Exception:
                    pass

            try:
                stereo.initialConfig.setLeftRightCheckThreshold(int(lr_threshold))
            except Exception:
                try:
                    stereo.initialConfig.algorithmControl.leftRightCheckThreshold = int(lr_threshold)
                except Exception:
                    pass

            med = median_filter_from_mode(int(median_mode))
            try:
                stereo.initialConfig.setMedianFilter(med if med is not None else dai.MedianFilter.MEDIAN_OFF)
            except Exception:
                pass

            pp = stereo.initialConfig.postProcessing
            try:
                pp.speckleFilter.enable = bool(speckle_filter)
                pp.speckleFilter.speckleRange = 32
            except Exception:
                pass
            try:
                pp.thresholdFilter.minRange = int(device_depth_min_mm)
                pp.thresholdFilter.maxRange = int(device_depth_max_mm)
            except Exception:
                pass
            try:
                pp.decimationFilter.decimationFactor = 1
            except Exception:
                pass
            try:
                pp.temporalFilter.enable = bool(temporal_filter)
            except Exception:
                pass
            try:
                pp.spatialFilter.enable = bool(spatial_filter)
            except Exception:
                pass
            try:
                pp.holeFilling.enable = False
            except Exception:
                pass

            self._queues["depth_crop"] = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

    def start(self) -> "OakSRStreams":
        if not self._running:
            self.pipeline.start()
            self._running = True
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        if self._running:
            try:
                self.pipeline.stop()
                try:
                    self.pipeline.wait()
                except Exception:
                    pass
            finally:
                self._running = False

    def __enter__(self) -> "OakSRStreams":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def is_running(self) -> bool:
        try:
            return bool(self._running and self.pipeline.isRunning())
        except Exception:
            return bool(self._running)

    def _read_packet(self, name: str, block: bool, use_cached: bool) -> Optional[np.ndarray]:
        if not self._running:
            raise RuntimeError("Call start() before reading frames.")
        if name not in self._queues:
            raise RuntimeError(f"Stream '{name}' was not enabled.")

        queue = self._queues[name]
        packet = queue.get() if block else get_latest_packet(queue)

        if packet is not None:
            if name in ("left", "right"):
                self._cache[name] = packet.getCvFrame()
            else:
                self._cache[name] = packet.getFrame()

        if packet is not None:
            return self._cache[name]
        if use_cached:
            return self._cache.get(name)
        return None

    def depth_crop_to_full_frame(self, depth_crop: np.ndarray) -> np.ndarray:
        """Place a cropped depth frame back into a full preview-sized canvas."""
        g = self.depth_geometry
        full = np.zeros((g.full_h, g.full_w), dtype=depth_crop.dtype)
        h = min(depth_crop.shape[0], g.proc_h)
        w = min(depth_crop.shape[1], g.proc_w)
        full[g.proc_y1 : g.proc_y1 + h, g.proc_x1 : g.proc_x1 + w] = depth_crop[:h, :w]
        return full

    def get_rgb_frame(self, block: bool = False, use_cached: bool = True) -> Optional[np.ndarray]:
        stream_name = "left" if self.rgb_socket == "left" else "right"
        return self._read_packet(stream_name, block=block, use_cached=use_cached)

    def get_left_frame(self, block: bool = False, use_cached: bool = True) -> Optional[np.ndarray]:
        return self._read_packet("left", block=block, use_cached=use_cached)

    def get_right_frame(self, block: bool = False, use_cached: bool = True) -> Optional[np.ndarray]:
        return self._read_packet("right", block=block, use_cached=use_cached)

    def get_depth_crop_frame(self, block: bool = False, use_cached: bool = True) -> Optional[np.ndarray]:
        return self._read_packet("depth_crop", block=block, use_cached=use_cached)

    def get_depth_frame(
        self,
        block: bool = False,
        use_cached: bool = True,
        as_full_frame: bool = True,
    ) -> Optional[np.ndarray]:
        depth_crop = self.get_depth_crop_frame(block=block, use_cached=use_cached)
        if depth_crop is None:
            return None
        if as_full_frame:
            return self.depth_crop_to_full_frame(depth_crop)
        return depth_crop

    def get_depth_grayscale(
        self,
        block: bool = False,
        use_cached: bool = True,
        near_percentile: float = 3.0,
        far_percentile: float = 95.0,
        invert: bool = False,
        dmin: Optional[float] = None,
        dmax: Optional[float] = None,
        as_full_frame: bool = True,
    ) -> Optional[np.ndarray]:
        depth = self.get_depth_frame(block=block, use_cached=use_cached, as_full_frame=as_full_frame)
        if depth is None:
            return None
        return depth_to_grayscale(
            depth,
            near_percentile=near_percentile,
            far_percentile=far_percentile,
            invert=invert,
            dmin=dmin,
            dmax=dmax,
        )

    def iter_rgb(self) -> Iterator[np.ndarray]:
        while self._running:
            frame = self.get_rgb_frame(block=True, use_cached=False)
            if frame is not None:
                yield frame

    def iter_left(self) -> Iterator[np.ndarray]:
        while self._running:
            frame = self.get_left_frame(block=True, use_cached=False)
            if frame is not None:
                yield frame

    def iter_right(self) -> Iterator[np.ndarray]:
        while self._running:
            frame = self.get_right_frame(block=True, use_cached=False)
            if frame is not None:
                yield frame

    def iter_depth(self, *, as_full_frame: bool = True) -> Iterator[np.ndarray]:
        while self._running:
            frame = self.get_depth_frame(block=True, use_cached=False, as_full_frame=as_full_frame)
            if frame is not None:
                yield frame

    def iter_depth_grayscale(
        self,
        near_percentile: float = 3.0,
        far_percentile: float = 95.0,
        invert: bool = False,
    ) -> Iterator[np.ndarray]:
        while self._running:
            frame = self.get_depth_grayscale(
                block=True,
                use_cached=False,
                near_percentile=near_percentile,
                far_percentile=far_percentile,
                invert=invert,
            )
            if frame is not None:
                yield frame


def create_rgb_stream(rgb_socket: SocketName = "left", **kwargs) -> OakSRStreams:
    return OakSRStreams(enable_rgb=True, rgb_socket=rgb_socket, **kwargs)


def create_left_stream(**kwargs) -> OakSRStreams:
    return OakSRStreams(enable_left=True, **kwargs)


def create_right_stream(**kwargs) -> OakSRStreams:
    return OakSRStreams(enable_right=True, **kwargs)


def create_depth_stream(**kwargs) -> OakSRStreams:
    return OakSRStreams(enable_depth=True, **kwargs)


def create_depth_gray_stream(**kwargs) -> OakSRStreams:
    return OakSRStreams(enable_depth_gray=True, **kwargs)


def create_all_streams(rgb_socket: SocketName = "left", **kwargs) -> OakSRStreams:
    return OakSRStreams(
        enable_rgb=True,
        rgb_socket=rgb_socket,
        enable_left=True,
        enable_right=True,
        enable_depth=True,
        enable_depth_gray=True,
        **kwargs,
    )


__all__ = [
    "OakSRStreams",
    "DepthCropGeometry",
    "compute_depth_crop_geometry",
    "snap_width_crop_to_multiple_of_16",
    "get_latest_packet",
    "depth_to_grayscale",
    "create_rgb_stream",
    "create_left_stream",
    "create_right_stream",
    "create_depth_stream",
    "create_depth_gray_stream",
    "create_all_streams",
]
