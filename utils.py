import cv2
import numpy as np


def refine_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask
