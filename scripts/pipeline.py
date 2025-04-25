#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import SimpleITK as sitk


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("DeepBrainScanPipeline")
    logger.setLevel(log_level.upper())
    
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    
    return logger

#1. n4
def n4corr(
    sitk_image: sitk.Image,
    logger: Optional[logging.Logger] = None
) -> sitk.Image:
    try:
        mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        image_float = sitk.Cast(sitk_image, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        corrected = corrector.Execute(image_float, mask)
        return corrected
    except Exception as e:
        raise

#2. hist matching
def apply_histogram_matching(
    image: sitk.Image,
    reference: sitk.Image,
    logger: Optional[logging.Logger] = None
) -> sitk.Image:
    try:
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(256)
        matcher.SetNumberOfMatchPoints(7)
        matched = matcher.Execute(image, reference)
        return matched
    except Exception as e:
        raise


# 2b. window intensity to separate bone peak

def window_intensity(
    image: sitk.Image,
    win_min: float = 0.0,
    win_max: float = 150.0,
    out_min: float = 0.0,
    out_max: float = 255.0
) -> sitk.Image:
    return sitk.IntensityWindowing(
        image,
        windowMinimum=win_min,
        windowMaximum=win_max,
        outputMinimum=out_min,
        outputMaximum=out_max
    )


# 2c. CLAHE (contrast))

def apply_clahe(
    np_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(np_img)


# 2d. global adjustment

def adjust_brightness_contrast(
    np_img: np.ndarray,
    alpha: float = 1.2,
    beta: float = -10
) -> np.ndarray:
    return cv2.convertScaleAbs(np_img, alpha=alpha, beta=beta)


# 3. Orientation step

def correct_orientation(
    image: sitk.Image,
    logger: Optional[logging.Logger] = None
) -> sitk.Image:
    try:
        image_3d = sitk.JoinSeries([image])
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAS")
        oriented_3d = orienter.Execute(image_3d)
        size_2d = oriented_3d.GetSize()[:2] + (0,)
        oriented_2d = sitk.Extract(oriented_3d, size_2d, (0, 0, 0))
        return oriented_2d
    except Exception:
        raise

#4. crop image
def crop_intracranial_region(
    image: sitk.Image,
    original_size: tuple = (640, 640),
    logger: Optional[logging.Logger] = None
) -> sitk.Image:
    try:
        np_image = sitk.GetArrayFromImage(image)
        # Flatten high-intensity bone peaks
        mask_bone = np_image > 200
        np_image[mask_bone] = 200
        np_image = cv2.normalize(np_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        median = np.median(np_image)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(np_image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        min_area = 0.05 * original_size[0] * original_size[1]
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid_contours:
            return image

        mask = np.zeros_like(np_image)
        cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(np_image, np_image, mask=mask)

        result_image = sitk.GetImageFromArray(masked_image)
        result_image.CopyInformation(image)
        return result_image
    except Exception as e:
        raise


def process_image(
    input_path: str,
    output_path: str,
    reference: sitk.Image,
    logger: logging.Logger
) -> bool:
    file_name = os.path.basename(input_path)
    logger.info(f"proccessing {file_name}")

    try:
        original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            return False

        sitk_image = sitk.GetImageFromArray(original_image)
        sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)

        # N4 bias correction
        corrected = n4corr(sitk_image, logger)

        # Histogram matching
        reference_slice = sitk.Extract(reference, reference.GetSize()[:2] + (0,), (0, 0, 0))
        matched = apply_histogram_matching(corrected, reference_slice, logger)

        # Windowing to reduce bone influence
        matched = window_intensity(matched, win_min=0, win_max=150, out_min=0, out_max=255)

        # Orientation correction
        oriented = correct_orientation(matched, logger)

        # Crop intracranial region
        cropped = crop_intracranial_region(oriented, original_size=original_image.shape, logger=logger)

        # Convert back to array and normalize
        processed_array = sitk.GetArrayFromImage(cropped)
        processed_array = cv2.normalize(processed_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Enhance local contrast and adjust brightness
        processed_array = apply_clahe(processed_array, clip_limit=2.0, tile_grid_size=(8, 8))
        processed_array = adjust_brightness_contrast(processed_array, alpha=1.2, beta=-10)

        cv2.imwrite(output_path, processed_array)
        return True

    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--coco_input", required=True)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    logger = setup_logger(args.log_level)
    logger.info("Pipeline starting...")

    with open(args.coco_input, 'r') as f:
        coco_data: Dict[str, Any] = json.load(f)

    coco_imgs: List[Dict[str, Any]] = coco_data['images']
    coco_ann: List[Dict[str, Any]] = coco_data['annotations']
    
    filt_imgs: List[Dict[str, Any]] = []
    _dis: set = set()

    for img_info in coco_imgs:
        img_path = os.path.join(args.input_dir, img_info['file_name'])
        if os.path.isfile(img_path):
            filt_imgs.append(img_info)
            _dis.add(img_info['id'])

    filtered_annotations: List[Dict[str, Any]] = [
        anno for anno in coco_ann if anno['image_id'] in _dis
    ]

    input_files: List[str] = [img['file_name'] for img in filt_imgs]
    reference_image = sitk.ReadImage(args.reference)

    os.makedirs(args.output_dir, exist_ok=True)

    succ_img: set = set()
    n = 0

    if args.max_workers > 1:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for f in input_files:
                in_path = os.path.join(args.input_dir, f)
                out_path = os.path.join(args.output_dir, f)
                futures[executor.submit(process_image, in_path, out_path, reference_image, logger)] = f

            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    if future.result():
                        succ_img.add(file_name)
                        n += 1
                except Exception as e:
                    logger.error(f"err processing {file_name}: {str(e)}")
    else:
        for f in input_files:
            in_path = os.path.join(args.input_dir, f)
            out_path = os.path.join(args.output_dir, f)
            if process_image(in_path, out_path, reference_image, logger):
                succ_img.add(f)
                n += 1

    successful_image_ids: set = {img['id'] for img in filt_imgs if img['file_name'] in succ_img}
    successful_images: List[Dict[str, Any]] = [img for img in filt_imgs if img['file_name'] in succ_img]
    successful_annotations: List[Dict[str, Any]] = [
        anno for anno in filtered_annotations if anno['image_id'] in successful_image_ids
    ]

    new_coco: Dict[str, Any] = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data.get('categories', []),
        'images': successful_images,
        'annotations': successful_annotations
    }

    output_coco_path = os.path.join(args.output_dir, '_annotations.coco.json')
    with open(output_coco_path, 'w') as f:
        json.dump(new_coco, f, indent=2)
    logger.info(f"coco saved: {output_coco_path}")
    logger.info(f"finished. images: {n}")


if __name__ == "__main__":
    main()