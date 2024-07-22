
import cv2
import numpy as np

bounding_box_mask_path = 'output_results/ca_dana_point/ca_dana_point_1_mask_output_mask.jpg'
original_mask_path = "/home/usama/Denoised_mask_results_3_july_2024/ca_dana_point/ca_dana_point_1_mask.jpg"

original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
bounding_box_mask = cv2.imread(bounding_box_mask_path, cv2.IMREAD_GRAYSCALE)
bounding_box_mask = cv2.resize(bounding_box_mask, (original_mask.shape[1], original_mask.shape[0]))
mask_with_bounding_boxes = cv2.bitwise_or(original_mask, bounding_box_mask)
cv2.imwrite("ca_dana_point_22_latest.png", mask_with_bounding_boxes)