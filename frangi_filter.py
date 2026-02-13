import os
import logging
import warnings
import numpy as np
import cv2
from skimage import io, filters
from PIL import Image


warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.ERROR)


def load_image(path: str) -> np.ndarray:
    """
    Load an image as a grayscale numpy array.
    """
    img = Image.open(path).convert('L')
    return np.array(img)


def save_image(img: np.ndarray, path: str) -> None:
    """
    Save a numpy array as an image.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def sato_filter(image: np.ndarray, sigmas: list = [1, 2, 3, 4], border: int = 3) -> np.ndarray:
    """
    Apply Sato Hessian-based vesselness filter and zero out borders.
    """
    sato = filters.sato(image.astype(np.uint8), sigmas=sigmas, black_ridges=True, mode="reflect", cval=0)
    result = sato.astype(np.uint8)
    # mask borders
    h, w = result.shape
    result[:border, :] = 0
    result[-border:, :] = 0
    result[:, :border] = 0
    result[:, -border:] = 0
    return result


def threshold_image(image: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    """
    Threshold image by percentile, zeroing values below threshold.
    """
    thresh_val = np.percentile(image, percentile)
    mask = np.where(image >= thresh_val, image, 0)
    return mask


def region_grow(img: np.ndarray, seed: tuple = None) -> np.ndarray:
    """
    Grow region from a seed point on nonzero pixels.
    If seed is None, uses the maximum-intensity pixel.
    Returns a binary mask of the grown region.
    """
    if seed is None:
        if np.any(img):
            idx = np.unravel_index(np.argmax(img), img.shape)
            seed = (idx[0], idx[1])
        else:
            return np.zeros_like(img, dtype=np.uint8)

    visited = np.zeros_like(img, dtype=bool)
    mask = np.zeros_like(img, dtype=np.uint8)
    stack = [seed]
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    while stack:
        x, y = stack.pop()
        if not (0 <= x < img.shape[0] and 0 <= y < img.shape[1]):
            continue
        if visited[x,y] or img[x,y] == 0:
            continue
        visited[x,y] = True
        mask[x,y] = 255
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1] and not visited[nx,ny] and img[nx,ny] > 0:
                stack.append((nx,ny))
    return mask


def segment_image(input_path: str,
                  output_path: str,
                  sato_sigmas: list = [1,2,3,4],
                  border: int = 3,
                  threshold_pct: float = 92.0) -> None:
    """
    Complete pipeline: load image, apply Sato filter, threshold, region grow, and save mask.
    """
    img = load_image(input_path)
    vesselness = sato_filter(img, sigmas=sato_sigmas, border=border)
    thresh = threshold_image(vesselness, percentile=threshold_pct)
    mask = region_grow(thresh)
    save_image(mask, output_path)


if __name__ == "__main__":
    border = 20
    percentile = 92.0 # default
    for dataset in ["cadica", "syntax", "xcad", "coronarydominance"]:
        image_folder = fr"/path/to/dataset/{dataset}"
        mask_folder = fr"/path/to/dataset/{dataset}_frangi"

        os.makedirs(mask_folder, exist_ok=True)

        image_items = os.listdir(image_folder)

        from tqdm import tqdm
        iterator = tqdm(image_items, desc=f"Processing {dataset}")

        for item in iterator:
            input_path = os.path.join(image_folder, item)
            output_path = os.path.join(mask_folder, item)
            
            segment_image(input_path, output_path, border=border, threshold_pct=percentile)