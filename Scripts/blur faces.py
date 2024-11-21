import torch
from facenet_pytorch import MTCNN
import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def enhance_image_for_detection(image):
    """
    Enhance image to improve face detection for darker skin tones
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge((l, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Increase brightness slightly
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    
    return enhanced

def apply_heavy_blur(image, kernel_size=99):
    """Apply multiple passes of heavy blur"""
    blurred = image.copy()
    blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
    blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 30)
    return blurred

def detect_faces_with_rotation(image, mtcnn):
    """
    Detect faces with image rotation to catch profile views
    Returns all unique detections
    """
    boxes_list = []
    
    # Original image
    boxes, _ = mtcnn.detect(image)
    if boxes is not None:
        boxes_list.extend(boxes)
    
    # Try with slight rotations to catch profile views
    angles = [-15, 15]  # Degrees
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    for angle in angles:
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        boxes, _ = mtcnn.detect(rotated)
        if boxes is not None:
            # Transform boxes back to original image coordinates
            rotation_matrix_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
            for box in boxes:
                # Transform each corner of the box
                corners = np.array([[box[0], box[1]], [box[2], box[1]], 
                                  [box[2], box[3]], [box[0], box[3]]])
                corners = np.hstack((corners, np.ones((4, 1))))
                corners = np.dot(rotation_matrix_inv, corners.T).T
                
                # Get new bounding box from transformed corners
                x1 = max(0, min(corners[:, 0]))
                y1 = max(0, min(corners[:, 1]))
                x2 = min(width, max(corners[:, 0]))
                y2 = min(height, max(corners[:, 1]))
                
                boxes_list.append([x1, y1, x2, y2])
    
    if not boxes_list:
        return None
        
    # Remove overlapping boxes
    boxes_list = np.array(boxes_list)
    return non_max_suppression(boxes_list)

def non_max_suppression(boxes, overlap_thresh=0.3):
    """Remove overlapping detections"""
    if len(boxes) == 0:
        return None
    
    boxes = boxes.astype("float")
    pick = []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick].astype("int")

def blur_faces(image, mtcnn):
    """
    Detect and heavily blur faces in a single image using MTCNN
    with enhanced detection for darker skin tones and profile views
    """
    # Create enhanced version for detection
    enhanced_image = enhance_image_for_detection(image)
    
    # Convert BGR to RGB for MTCNN
    rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    
    # Detect faces with rotation
    boxes = detect_faces_with_rotation(rgb_image, mtcnn)
    
    if boxes is None:
        return image, 0
    
    # Blur each detected face
    for box in boxes:
        # Get integer coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Add padding around the face (15% of face size)
        w, h = x2 - x1, y2 - y1
        padding_w = int(w * 0.15)
        padding_h = int(h * 0.15)
        
        # Calculate padded coordinates
        x1 = max(0, x1 - padding_w)
        y1 = max(0, y1 - padding_h)
        x2 = min(image.shape[1], x2 + padding_w)
        y2 = min(image.shape[0], y2 + padding_h)
        
        # Extract the face region with padding
        face_roi = image[y1:y2, x1:x2]
        
        # Apply heavy blur effect
        blurred_face = apply_heavy_blur(face_roi)
        
        # Replace the original face region with the blurred version
        image[y1:y2, x1:x2] = blurred_face
    
    return image, len(boxes)

def process_images(input_folder, output_folder, show_blurring_result):
    """
    Process all images in the input folder and save blurred versions to output folder
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize MTCNN with very low confidence threshold for better detection
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        margin=0,
        min_face_size=20,
        thresholds=[0.4, 0.5, 0.5],  # Even lower thresholds for better detection
        factor=0.709,
        post_process=False,
        device=device,
        keep_all=True
    )
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    total_images = 0
    total_faces = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Read the image
                original_image = cv2.imread(input_path)
                if original_image is None:
                    print(f"Error: Could not read image: {filename}")
                    continue
                
                # Create a copy for processing
                image_to_process = original_image.copy()
                
                # Process the image
                processed_image, num_faces = blur_faces(image_to_process, mtcnn)
                
                # Save the processed image
                cv2.imwrite(output_path, processed_image)
                
                total_images += 1
                total_faces += num_faces
                
                if show_blurring_result:
                    # Scale down large images for display
                    max_display_width = 1200
                    scale = min(1.0, max_display_width / (original_image.shape[1] * 2))
                    if scale < 1.0:
                        display_original = cv2.resize(original_image, None, fx=scale, fy=scale)
                        display_processed = cv2.resize(processed_image, None, fx=scale, fy=scale)
                    else:
                        display_original = original_image
                        display_processed = processed_image
                    
                    # Display original and processed images side by side
                    combined = np.hstack((display_original, display_processed))
                    cv2.imshow('Original vs Processed', combined)
                    
                    # Wait for 3 seconds (3000 milliseconds)
                    key = cv2.waitKey(3000)
                    
                    # If ESC is pressed during the 3 seconds, break the loop
                    if key == 27:  # ESC key
                        break
                    
                    # Close the window after 3 seconds
                    cv2.destroyAllWindows()
                
                print(f"Processed: {filename} - Found {num_faces} faces")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    if show_blurring_result:
        cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Total faces detected: {total_faces}")

def main():
    parser = argparse.ArgumentParser(description='Blur faces in images')
    parser.add_argument('--input_folder', required=True, help='Folder containing input images')
    parser.add_argument('--output_folder', required=True, help='Folder for saving processed images')
    parser.add_argument('--show_blurring_result', action='store_true', 
                        help='Show before/after comparison for each image')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return
    
    process_images(args.input_folder, args.output_folder, args.show_blurring_result)

if __name__ == "__main__":
    main()