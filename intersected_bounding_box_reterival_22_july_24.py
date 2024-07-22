



import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

bounding_box_mask_path = '/home/usama/EasyOCR_high_resolution_text_localization/text_masks_results_july_13_2024/ca_dana_pointtext_masktext_mask.png'
original_mask_path = "/home/usama/Denoised_mask_results_3_july_2024/ca_dana_point/ca_dana_point_5_mask.jpg"


def validate_polygon(polygon):
    try:
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon
    except Exception as e:
        print(f"Error validating polygon: {e}")
        return None

def retrieve_intersections(args):
    bounding_box, mask_polygons = args
    cnt_bb_2d = np.squeeze(bounding_box)
    if cnt_bb_2d.shape[0] < 4:
        return None

    polygon_bb = validate_polygon(Polygon(cnt_bb_2d))
    if not polygon_bb:
        return None

    intersections = []
    for polygon_mask in mask_polygons:
        # if polygon_mask.intersects(polygon_bb)
        if polygon_bb:
            intersection = polygon_bb.intersection(polygon_mask)
            if not intersection.is_empty:
                intersections.append(polygon_bb)

    return intersections if intersections else None

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cnt.shape[0] >= 4:
            polygon = Polygon(np.squeeze(cnt))
            valid_polygon = validate_polygon(polygon)
            if valid_polygon:
                polygons.append(valid_polygon)
    return polygons

def process_images(mask_image_path, bbox_image_path, output_dir):
    mask_image_name = os.path.splitext(os.path.basename(mask_image_path))[0]
    print(f"Processing mask image: {mask_image_name}")

    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        print(f"Error reading mask image: {mask_image_path}")
        return None

    bbox_image = cv2.imread(bbox_image_path, cv2.IMREAD_GRAYSCALE)
    if bbox_image is None:
        print(f"Error reading bounding box image: {bbox_image_path}")
        return None

    _, thresh_mask = cv2.threshold(mask_image, 20, 255, cv2.THRESH_BINARY)
    cv2.imwrite("thresholded_image_ca_dana_point_5.png",thresh_mask)
    # thresh_mask = cv2.medianBlur(thresh_mask,5)
    thresh_mask_resized = cv2.resize(thresh_mask,(900,800))
    # cv2.imshow("thresholded_mask",thresh_mask_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask_polygons = mask_to_polygons(thresh_mask)

    _, thresh_bbox = cv2.threshold(bbox_image, 25, 255, cv2.THRESH_BINARY)
    cv2.imwrite("thresholded_image_ca_dana_point_bbox.png",thresh_bbox)
    contours_bbox, _ = cv2.findContours(thresh_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    args = [(bbox, mask_polygons) for bbox in contours_bbox if bbox.shape[0] > 0]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(retrieve_intersections, args))

    output_mask = np.zeros(mask_image.shape, dtype=np.uint8)
    for result in results:
        if result is not None:
            for poly in result:
                if isinstance(poly, Polygon):
                    exterior_coords = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.drawContours(output_mask, [exterior_coords], -1, 255, cv2.FILLED)
                elif isinstance(poly, MultiPolygon):
                    for subpoly in poly.geoms:
                        exterior_coords = np.array(subpoly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.drawContours(output_mask, [exterior_coords], -1, 255, cv2.FILLED)

    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(mask_image_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{mask_image_name}_output_mask.jpg")
    cv2.imwrite(output_file_path, output_mask)

    print(f"Processing complete for: {mask_image_name}")
    return output_mask


process_images(original_mask_path, bounding_box_mask_path, 'output_results')





