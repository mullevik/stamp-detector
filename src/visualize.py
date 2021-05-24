
"""
This file contains all visualization functions.
"""
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def show_bordered_image(bounding_rectangles: List[Tuple],
                        rgb_image: np.ndarray, thickness: int = 2) -> None:
    """
    :param bounding_rectangles: list of (x, y, w, h) rectangles
    :param rgb_image: image to draw rectangles over
    :param thickness: how thick should the borders be
    """
    bordered_image = rgb_image.copy()

    tab_colors = [[int(y * 255) for y in colors.to_rgb(x)]
                  for x in colors.TABLEAU_COLORS.values()]
    for i, bbox in enumerate(bounding_rectangles):
        x, y, w, h = bbox
        color = tab_colors[i % len(tab_colors)]
        bordered_image = cv2.rectangle(bordered_image,
                                       (x, y),  # top left
                                       (x + w, y + h),  # bottom right
                                       color,
                                       thickness)
    plt.imshow(bordered_image)
    plt.show()
