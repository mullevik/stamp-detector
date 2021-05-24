import os
from typing import Tuple, List
import logging

import cv2
import numpy as np

from src.clustering import cluster_document
from src.visualize import show_bordered_image

log = logging.getLogger(__name__)


MAX_IMAGE_SIZE = 1024
NUMBER_OF_INFORMATION_BLOBS = 8
COLOR_SIGNIFICANCE = 5.  # how much color influences the blobs


def downscale_image(image: np.ndarray, larger_axis_size: int,
                    interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """
    Downscales 'image' so that it's larger axis is at most 'larger_axis_size'.
    If the image is smaller, then this function does nothing.
    :param image: image to downscale
    :param larger_axis_size: max size of the larger axis
    :param interpolation
    :return: downscaled (or unmodified) 'image'
    """
    current_larger_axis_size = max(image.shape[0], image.shape[1])
    if current_larger_axis_size > larger_axis_size:
        scale_factor = larger_axis_size / current_larger_axis_size
        image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor,
                           interpolation=interpolation)
    return image


def separate_bg_from_fg(image: np.ndarray, blur_amount: float = 0.):
    """
    Creates a binary image separating foreground from background using Otsu's
    thresholding method.
    (https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
    :param image: bgr_image
    :param blur_amount: if 0 (default), then no blur is added
    before thresholding, else the blur_amount is used for Gaussian blur
    :return: binary image (0 - foreground, 255 - background)
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur_amount == 0:
        # Otsu's thresholding
        threshold, binary_image = cv2.threshold(
            grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    else:
        # Otsu's thresholding after Gaussian filtering
        blurred_image = cv2.GaussianBlur(grayscale_image,
                                         (blur_amount, blur_amount), 0)
        threshold, binary_image = cv2.threshold(
            blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image


def store_cut_information_blobs(image: np.ndarray,
                                bounding_rectangles: List[Tuple],
                                destination: str,
                                file_name: str = "document_cluster") -> None:
    """
    Writes extracted information
    (only saves the bounding rectangle cut assigned to a specific
    information blob)
    :param image: bgr image
    :param bounding_rectangles: list of rectangles (4-tuple)
    :param destination: output directory path
    :param file_name: name of the files
    (they will always end with '_<cluster_number>.jpg')
    :return:
    """
    for i, bounding_rectangle in enumerate(bounding_rectangles):
        x, y, w, h = bounding_rectangle
        cut_image = image[y:y+h, x:x+w, :]
        file_path = os.path.join(destination,
                                 f"{file_name}_{str(i).zfill(3)}.jpg")
        log.debug(f"writing image file {file_path}")
        cv2.imwrite(file_path, cut_image)


def store_extracted_information_blobs(image: np.ndarray,
                                      cluster_images: List[np.ndarray],
                                      destination: str,
                                      file_name: str = "document_cluster") \
        -> None:
    """
    Writes extracted information
    (only pixels assigned to a specific information blob)
    :param image: bgr image
    :param cluster_images: list of binary images
    (0 - background, 255 - foreground)
    :param destination: output directory path
    :param file_name: name of the files
    (they will always end with '_<cluster_number>.jpg')
    """
    WHITE_COLOR = (255, 255, 255)
    BACKGROUND = 0
    for i, binary_image in enumerate(cluster_images):
        extracted_image = image.copy()
        extracted_image[binary_image == BACKGROUND, :] = WHITE_COLOR
        file_path = os.path.join(destination,
                                 f"{file_name}_{str(i).zfill(3)}.jpg")
        log.debug(f"writing image file {file_path}")
        cv2.imwrite(file_path, extracted_image)


def extract_information_blobs(document_image_file: str,
                              visualize: bool = False,
                              output: str = None) -> List[Tuple]:
    """
    Extract information blobs from document image.
    :param document_image_file: path to document image file
    :return: list of bounding rectangles of information blobs
    """
    full_image = cv2.imread(document_image_file)
    image = downscale_image(full_image, MAX_IMAGE_SIZE)

    binary_image = separate_bg_from_fg(image)

    cluster_images, cost = cluster_document(image, binary_image,
                                            NUMBER_OF_INFORMATION_BLOBS,
                                            COLOR_SIGNIFICANCE)

    bounding_boxes = []
    for cluster_image in cluster_images:
        contours, hierarchy = cv2.findContours(cluster_image,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes.append(cv2.boundingRect(cluster_image))

    if visualize:
        show_bordered_image(bounding_boxes,
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if output is not None:
        store_cut_information_blobs(image, bounding_boxes, output)
        # store_extracted_information_blobs(image, cluster_images, output)

    return bounding_boxes


if __name__ == "__main__":
    doc_file = "../data/documents/stamps/Faktura č. 2016275 za 12_2016.pdf-1.jpg"

    out = extract_information_blobs(doc_file, visualize=True, output="../out/")
