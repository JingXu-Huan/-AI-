"""
perception/preprocessor.py

增强版图像预处理模块，支持多种预处理策略。
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Literal


class ImagePreprocessor:
    """在输入 YOLO 检测器之前，做图像预处理。
    
    支持多种预处理策略：
    - 高斯去噪
    - CLAHE 对比度增强
    - 边缘增强
    - 图像锐化
    - 自适应直方图均衡化

    Parameters
    ----------
    denoise : bool
        是否开启高斯去噪。
    clahe : bool
        是否开启 CLAHE 对比度增强。
    edge_enhancement : bool
        是否开启边缘增强。
    sharpen : bool
        是否开启图像锐化。
    mode : str
        预处理模式: "default", "high_contrast", "edge_focused"
    """

    def __init__(
        self,
        denoise: bool = True,
        clahe: bool = True,
        edge_enhancement: bool = False,
        sharpen: bool = False,
        mode: Literal["default", "high_contrast", "edge_focused"] = "default",
    ) -> None:
        self.denoise = denoise
        self.clahe = clahe
        self.edge_enhancement = edge_enhancement
        self.sharpen = sharpen
        self.mode = mode
        
        # 初始化 CLAHE
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """对 BGR 图像做预处理。

        Parameters
        ----------
        image : np.ndarray
            BGR 格式的输入图像（通常由 `cv2.imread()` 返回）。

        Returns
        -------
        np.ndarray
            处理后的 BGR 图像，直接送给 YOLO 推断。
        """
        if image is None or image.size == 0:
            raise ValueError("ImagePreprocessor.process() received an empty image.")

        img = image.copy()

        # 根据模式选择预处理策略
        if self.mode == "default":
            img = self._process_default(img)
        elif self.mode == "high_contrast":
            img = self._process_high_contrast(img)
        elif self.mode == "edge_focused":
            img = self._process_edge_focused(img)

        return img

    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """批量处理图像列表。
        
        Parameters
        ----------
        images : list[np.ndarray]
            BGR 格式的图像列表。
            
        Returns
        -------
        list[np.ndarray]
            处理后的图像列表。
        """
        return [self.process(img) for img in images]

    # ------------------------------------------------------------------
    # Private helpers - 模式实现
    # ------------------------------------------------------------------

    def _process_default(self, image: np.ndarray) -> np.ndarray:
        """默认预处理：去噪 + CLAHE"""
        img = image
        
        if self.denoise:
            img = self._apply_denoise(img)
        
        if self.clahe:
            img = self._apply_clahe(img)
            
        return img

    def _process_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """高对比度预处理：增强对比度和亮度"""
        img = image
        
        if self.denoise:
            img = self._apply_denoise(img)
        
        # 多次 CLAHE 增强
        if self.clahe:
            img = self._apply_clahe(img, clip_limit=4.0)
            img = self._apply_clahe(img, clip_limit=3.0)
        
        # 锐化
        if self.sharpen:
            img = self._apply_sharpen(img)
            
        return img

    def _process_edge_focused(self, image: np.ndarray) -> np.ndarray:
        """边缘聚焦预处理：强调边缘特征"""
        img = image
        
        if self.denoise:
            img = self._apply_denoise(img)
        
        # 边缘增强
        if self.edge_enhancement:
            img = self._apply_edge_enhancement(img)
        
        # 轻度 CLAHE
        if self.clahe:
            img = self._apply_clahe(img, clip_limit=2.0)
        
        return img

    # ------------------------------------------------------------------
    # Private helpers - 具体实现
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_denoise(image: np.ndarray) -> np.ndarray:
        """用高斯模糊轻轻平滑图片，减少噪声。"""
        return cv2.GaussianBlur(image, (3, 3), 0)

    def _apply_clahe(self, image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
        """CLAHE 对比度增强。"""
        # 每次调用创建新的 CLAHE 对象以支持不同参数
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _apply_edge_enhancement(image: np.ndarray) -> np.ndarray:
        """边缘增强 - 使用锐化卷积核。"""
        # 锐化卷积核
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        
        # 先提取边缘
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 混合原图和边缘
        result = cv2.addWeighted(image, 0.8, edges_color, 0.2, 0)
        
        # 应用锐化
        sharpened = cv2.filter2D(result, -1, kernel)
        
        return sharpened

    @staticmethod
    def _apply_sharpen(image: np.ndarray) -> np.ndarray:
        """图像锐化。"""
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _apply_adaptive_histogram(image: np.ndarray) -> np.ndarray:
        """自适应直方图均衡化。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)