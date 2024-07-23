
import numpy as np
import easyocr
import cv2
import math

bounding_boxes = []
def load_image(image_path):
    # Load the image
    return cv2.imread(image_path)

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    # Calculate the number of rows and columns
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols

def extract_tile(image, start_x, start_y, tile_width, tile_height):
    # Extract the tile from the image
    end_x = min(start_x + tile_width, image.shape[1])
    end_y = min(start_y + tile_height, image.shape[0])
    return image[start_y:end_y, start_x:end_x]

def detect_text_in_tile(image, tile_width, tile_height, reader):
    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)

    # Iterate over each row
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)
    for r in range(num_rows):
        # Iterate over each column
        for c in range(num_cols):
            # Calculate the starting coordinates of the tile
            start_x = c * tile_width
            start_y = r * tile_height

            # Extract the tile from the image
            tile = extract_tile(image, start_x, start_y, tile_width, tile_height)

            result = reader.readtext(tile,text_threshold=0.7)

            # Check if any bounding boxes were returned
            if len(result) > 0:
                # Extract the bounding box coordinates and text from the result
                bounding_boxes_tile = [bbox for bbox, _, _ in result]

                # Map the bounding box coordinates back to the original image coordinates
                for bbox in bounding_boxes_tile:
                    try:
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
                    except ValueError:
                        continue

                    # Adjust bounding box coordinates to fit the original image
                    x1 += start_x
                    y1 += start_y
                    x2 += start_x
                    y2 += start_y
                    x3 += start_x
                    y3 += start_y
                    x4 += start_x
                    y4 += start_y

                    mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    mapped_bbox = np.array(mapped_bbox, dtype=np.int32)
                    mapped_bbox = mapped_bbox.reshape((-1, 1, 2))
                    bounding_boxes.append(mapped_bbox)
                    output_image = cv2.polylines(output_image, [mapped_bbox], isClosed=True, color=(0, 255, 0), thickness=3)


    return bounding_boxes, output_image

def main(image_path, tile_width, tile_height):
    # Load the image
    image = load_image(image_path)

    # Initialize EasyOCR reader outside the loop
    reader = easyocr.Reader(['en','te'], gpu=True)  # this needs to run only once to load the model into memory

    # Detect text in tiles
    bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, reader)
    cv2.imwrite('Clewiston-Zoning-Map.png', output_image)
    return bounding_boxes

image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/Clewiston-Zoning-Map-page-001_modified.jpg'
image = cv2.imread(image_path)
tile_width = 1024
tile_height = 1024
bounding_boxes= main(image_path, tile_width, tile_height)




















































