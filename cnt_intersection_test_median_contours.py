
import time
import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon
from shapely.validation import make_valid

def retrieve_poly(args):
    ori, contours_filled = args
    cnt_ori_2d = np.squeeze(ori)
    if cnt_ori_2d.shape[0] < 4:
        return None

    polygon_ori = Polygon(cnt_ori_2d)
    valid_polygon_ori = make_valid(polygon_ori)
  
    # print("valid polygon original", valid_polygon_ori)

    for cnt_fill in contours_filled:
        
        # print("Contour Filled",cnt_fill)
        cnt_fill_2d = np.squeeze(cnt_fill)
        if cnt_fill_2d.shape[0] <4:
            continue

        polygon_fill = Polygon(cnt_fill_2d)
        polygon_fill = make_valid(polygon_fill)
        # print("walid Polygon fill",polygon_fill)
        if valid_polygon_ori.intersects(polygon_fill):
            return ori
    return None
    

def process_image(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing image: {image_name}")

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Error reading image: {image_path}")
        return None

    # Thresholding to get the original contours
    _, thresh_original = cv2.threshold(original, 20, 255, cv2.THRESH_BINARY)
    contours_original, _ = cv2.findContours(thresh_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Applying median blur and thresholding to get the filled contours
    median = cv2.medianBlur(original, 3)
    _, thresh_median = cv2.threshold(median, 25, 255, cv2.THRESH_BINARY)
    contours_filled, _ = cv2.findContours(thresh_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a blank mask
    mask = np.zeros(original.shape, dtype=np.uint8)

    # Prepare arguments for parallel processing
    args = []
    for ori in contours_original:
        # area = cv2.contourArea(ori)
        if ori.shape[0] > 3:
            # print("original contours",ori) 
            args.append((ori, contours_filled))
    # args = [(ori, contours_filled) for ori in contours_original if ori.shape[0] > 3]

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(retrieve_poly, args))

    # Drawing valid contours on the mask
    for result in results:
        # print("result",result)
        if result is not None:
            cv2.drawContours(mask, [result], -1, 255, cv2.FILLED)
    
    # mask = cv2.bitwise_and(mask,thresh_original) 
    # cv2.imwrite("mask_.jpg",mask)
    # cv2.imshow("Mask image",cv2.resize(mask , (700,800)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Save the mask
    output_subdir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
    os.makedirs(output_subdir, exist_ok=True)
    output_file_path = os.path.join(output_subdir, f"{image_name}_mask.jpg")
    cv2.imwrite(output_file_path, mask)

    print(f"Image processed: {image_name}")
    return mask

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                process_image(image_path, output_dir)
                print(f"Processed {filename} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    input_directory = '/home/usama/Test_masks/'
    output_directory = '/home/usama/Contour_median_reteival_results/'
    process_images(input_directory, output_directory)
