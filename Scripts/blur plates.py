import os
import cv2
import torch
import numpy as np
import time

# Load YOLOv7 model (modify this path as needed)
model_path = 'c:/b/best.pt'  # Path to the trained YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path, source='github')

def blur_license_plate(image, coordinates):
    for (x1, y1, x2, y2) in coordinates:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        plate = image[y1:y2, x1:x2]
        blurred_plate = cv2.GaussianBlur(plate, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_plate
    return image

def process_images(input_folder, output_folder, show_result=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_files = []

    for img_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read image {input_path}")
            continue

        # YOLO model prediction
        results = model(image)
        detections = results.xyxy[0].cpu().numpy()

        # Extract coordinates of detected license plates (assuming class label for license plate is '0')
        license_plate_coords = [d[:4] for d in detections if int(d[5]) == 0]

        # Blur detected license plates
        blurred_image = blur_license_plate(image, license_plate_coords)

        # Save the blurred image
        cv2.imwrite(output_path, blurred_image)

        # Record the input and output file paths
        processed_files.append((input_path, output_path))

        # Show side-by-side comparison if requested
        if show_result:
            combined = np.hstack((cv2.imread(input_path), blurred_image))
            cv2.namedWindow('Original vs Blurred', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Original vs Blurred', 1024, 768)
            cv2.imshow('Original vs Blurred', combined)
            cv2.waitKey(1500)  # Display for 1,5 seconds
            cv2.destroyAllWindows()

    return processed_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Blur license plates in images using YOLOv7 model")
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder of images')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder of images')
    parser.add_argument('--show_blurring_result', action='store_true', help='Show blurring result')
    args = parser.parse_args()

    processed_files = process_images(args.input_folder, args.output_folder, args.show_blurring_result)
    for input_file, output_file in processed_files:
        print(f"Processed: Input file: {input_file}, Output file: {output_file}")

    # Output parameter for UiPath
    output_param = {'processed_files': processed_files}
    print(output_param)
