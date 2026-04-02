"""
perception/preprocessor.py

一个给 YOLO 新手看的、尽量简单的图片预处理模块。

当前只保留一件事：
  - 高斯去噪：轻轻模糊一下图片，减少一点噪声。

说明：
  - 旧参数 `clahe` 和 `edge_enhancement` 还保留着，主要是为了兼容
    之前的配置文件和调用方式；但现在它们不会再参与实际处理。
  - 这样做的好处是代码更容易看懂，也更适合刚接触 Python / YOLO 的人。
"""

from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """在输入 YOLO 检测器之前，先做最基础的图片处理。

    Parameters
    ----------
    denoise : bool
        是否开启高斯去噪。
    clahe : bool
        兼容旧配置的参数，当前保留但不再使用。
    edge_enhancement : bool
        兼容旧配置的参数，当前保留但不再使用。
    """

    def __init__(
        self,
        denoise: bool = True,
        clahe: bool = True,
        edge_enhancement: bool = False,
    ) -> None:
        # 是否启用去噪：这是现在唯一真正生效的预处理步骤。
        self.denoise = denoise

        # 下面两个参数保留，是为了兼容旧代码和旧配置文件。
        # 但我们已经不再做 CLAHE 和边缘增强了。
        self.clahe = clahe
        self.edge_enhancement = edge_enhancement

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """对 BGR 图像做最基础的预处理。

        Parameters
        ----------
        image : np.ndarray
            BGR 格式的输入图像（通常由 `cv2.imread()` 返回）。

        Returns
        -------
        np.ndarray
            处理后的 BGR 图像，直接送给 YOLO 推断。
        """
        # 空图片不能处理，直接报错，方便排查问题。
        if image is None or image.size == 0:
            raise ValueError("ImagePreprocessor.process() received an empty image.")

        # 先复制一份，避免改到原图。
        img = image.copy()

        # 现在只保留去噪这一步。
        if self.denoise:
            img = self._apply_denoise(img)

        return img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_denoise(image: np.ndarray) -> np.ndarray:
        """用高斯模糊轻轻平滑图片，减少噪声。"""
        return cv2.GaussianBlur(image, (3, 3), 0)

