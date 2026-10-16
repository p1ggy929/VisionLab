"""相机后端抽象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry import Intrinsics


@dataclass
class FrameBundle:
    """彩色 + 深度帧的统一容器。"""

    color: np.ndarray                      # BGR uint8 HxWx3
    depth_m: np.ndarray                    # float32 HxW，单位米（已乘 depth_scale）
    depth_raw: np.ndarray                  # uint16 HxW，原始值（无损落盘用）
    depth_scale: float                     # 米/单位
    timestamp: float
    intrinsics: Intrinsics = field(default_factory=Intrinsics)
    frame_number: int = 0
    serial: str = ""
    source: str = "mock"                   # mock | realsense | replay

    @property
    def shape(self) -> tuple[int, int]:
        return self.color.shape[:2]

    def clone(self) -> "FrameBundle":
        return FrameBundle(
            color=self.color.copy(),
            depth_m=self.depth_m.copy(),
            depth_raw=self.depth_raw.copy(),
            depth_scale=self.depth_scale,
            timestamp=self.timestamp,
            intrinsics=self.intrinsics,
            frame_number=self.frame_number,
            serial=self.serial,
            source=self.source,
        )


class CameraBackend:
    """相机后端接口。所有实现必须保证 get_latest 不阻塞超过 timeout。"""

    name = "base"

    def describe(self) -> dict:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_latest(self, timeout: float = 1.0) -> Optional[FrameBundle]:
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        return False
