
import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import time
import re

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

    bounding_box_area = polygon_bb.area

    intersections = []
    for polygon_mask in mask_polygons:
        intersection = polygon_bb.intersection(polygon_mask)
        if not intersection.is_empty:
            intersection_area = intersection.area
            if intersection_area / bounding_box_area >= 0.1:
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

def process_image(mask_image_path, bbox_image_path, output_dir):
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
    mask_polygons = mask_to_polygons(thresh_mask)

    _, thresh_bbox = cv2.threshold(bbox_image, 25, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("thresholded_image_ca_dana_point_bbox.png",thresh_bbox)
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

    print(f"Processing complete for: {mask_image_name}")
    return output_mask


def process_images(input_dir, output_dir, bounding_box_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    file_count = 0
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_count += 1
                image_path = os.path.join(root, filename)
                mask_image_name = os.path.splitext(os.path.basename(image_path))[0]
                print("mask image name :",mask_image_name)
                match = re.match(r'(.+)_\d+_mask', mask_image_name )
                if match:
                    mask_image_name1  = match.group(1)
                # return base_name
                print("Mask image name 1:",mask_image_name1)
                ori_image_mask = cv2.imread(image_path)
                
                bbox_mask_path = os.path.join(bounding_box_dir, f"{mask_image_name1}_text_mask_text_mask.png")
                if not os.path.exists(bbox_mask_path):
                    print(f"Bounding box mask not found for {filename}")
                    continue
                
                bbox_mask = process_image(image_path, bbox_mask_path, output_dir)
                if bbox_mask is None:
                    continue

                bbox_mask = cv2.merge([bbox_mask, bbox_mask, bbox_mask])
                bbox_mask = cv2.resize(bbox_mask, (ori_image_mask.shape[1], ori_image_mask.shape[0]))
                
                ori_image_mask = ori_image_mask.astype(np.uint8)
                bbox_mask = bbox_mask.astype(np.uint8)
                mask_with_bounding_boxes = cv2.bitwise_or(ori_image_mask, bbox_mask)
                
                output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
                output_subdir = f"{output_subdir}_intersection_of_0.1"
                os.makedirs(output_subdir, exist_ok=True)
                
                output_file_path = os.path.join(output_subdir, f"{mask_image_name}_output_mask.jpg")
                cv2.imwrite(output_file_path, mask_with_bounding_boxes)
                
                print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    input_directory = '/home/usama/Denoised_mask_results_3_july_2024/'
    bounding_box_dir = '/home/usama/EasyOCR_high_resolution_text_localization/text_filled_bounding_box_masks_results_july_27_2024/'
    output_directory = '/home/usama/text_erased_using_reteival_results_july_29_2024/'
    
    process_images(input_directory, output_directory, bounding_box_dir)


# def process_images(input_dir, output_dir, bounding_box_path):
#     os.makedirs(output_dir, exist_ok=True)
#     start_time = time.time()
#     for root, _, files in os.walk(input_dir):
#         for filename in files:
#             if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 image_path = os.path.join(root, filename)
#                 mask_image_name = os.path.splitext(os.path.basename(image_path))[0]
#                 ori_image_mask = cv2.imread(image_path)
#                 bbox_mask = process_image(image_path, bounding_box_path,output_dir)
#                 bbox_mask = cv2.merge([bbox_mask, bbox_mask, bbox_mask])
#                 if bbox_mask is None:
#                     continue
#                 # print(f'ori_image_mask shape: {ori_image_mask.shape}')
#                 # print(f'bbox_mask: {bbox_mask.shape}')
#                 bbox_mask = cv2.resize(bbox_mask, (ori_image_mask.shape[1], ori_image_mask.shape[0]))
#                 ori_image_mask = ori_image_mask.astype(np.uint8)
#                 bbox_mask = bbox_mask.astype(np.uint8)
#                 mask_with_bounding_boxes = cv2.bitwise_or(ori_image_mask, bbox_mask)
#                 output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
#                 output_subdir = f"{output_subdir}_intersection_of_0.1"
#                 os.makedirs(output_subdir, exist_ok=True)
#                 output_file_path = os.path.join(output_subdir, f"{mask_image_name}_output_mask.jpg")
#                 cv2.imwrite(output_file_path, mask_with_bounding_boxes)

#                 print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")

# if __name__ == "__main__":
#     # input_directory = '/home/usama/Denoised_mask_results_3_july_2024/ma_canton/'
#     input_directory = '/home/usama/Denoised_mask_results_3_july_2024/Clewiston-Zoning-Map-page-001_modified'
#     # bounding_box_path = "/home/usama/EasyOcr_based_text_Erasing_latest_for_using/text_masks_results_july_13_2024_text_threshold_0.01/ma_cantontext_masktext_mask.png"
#     bounding_box_path = '/home/usama/EasyOCR_high_resolution_text_localization/text_filled_bounding_box_masks_results_july_25_2024/Clewiston-Zoning-Map-page-001_modifiedtext_masktext_mask.png'
#     output_directory = '/home/usama/text_erased_using_reteival_results_july_25_2024/'
#     process_images(input_directory, output_directory, bounding_box_path)








