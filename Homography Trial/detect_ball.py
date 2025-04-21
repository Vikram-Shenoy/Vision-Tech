import os
import cv2
from ultralytics import YOLO
import time # Optional: to measure processing time

# --- Configuration ---
MODEL_PATH = 'yolov8x.pt'       # Path to your downloaded YOLOv8 model file
IMAGE_FOLDER = 'raw_frames' # <<< CHANGE THIS to the folder containing your image frames
OUTPUT_FOLDER = 'detected_frames' # <<< Optional: CHANGE THIS if you want to save images with drawings
SAVE_IMAGES = True               # Set to True to save images with detections, False otherwise
CONFIDENCE_THRESHOLD = 0.3       # Minimum confidence score to consider a detection (adjust as needed)
FOOTBALL_CLASS_ID = 32           # Class ID for 'sports ball' in the COCO dataset (YOLOv8 is often trained on COCO)

# --- Setup ---
# Create output folder if it doesn't exist and SAVE_IMAGES is True
if SAVE_IMAGES:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
    # Get class names from the model to verify the football class ID (optional)
    class_names = model.names
    if FOOTBALL_CLASS_ID in class_names:
        print(f"Using class ID {FOOTBALL_CLASS_ID} which corresponds to '{class_names[FOOTBALL_CLASS_ID]}'")
    else:
        print(f"[Warning] Class ID {FOOTBALL_CLASS_ID} not found in model's class names. Detections might fail.")
        print("Available classes:", class_names)

except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the model path is correct and the file is not corrupted.")
    exit()

# --- Storage for results ---
detection_results = []

# --- Process Images ---
print(f"Processing images from: {IMAGE_FOLDER}")
start_time = time.time()

# Get list of image files
try:
    image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    if not image_files:
        print(f"No image files found in {IMAGE_FOLDER}")
        exit()
except FileNotFoundError:
    print(f"Error: Folder not found - {IMAGE_FOLDER}")
    exit()


for image_name in image_files:
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    print(f"--- Processing: {image_name} ---")

    # Perform inference, filtering by class ID and confidence
    try:
        results = model.predict(
            source=image_path,
            classes=[FOOTBALL_CLASS_ID], # Filter results server-side to only include footballs
            conf=CONFIDENCE_THRESHOLD,
            verbose=False # Suppress detailed YOLO output per image
        )
    except Exception as e:
        print(f"  Error during prediction for {image_name}: {e}")
        continue # Skip to the next image

    # results is a list, usually with one element for the single image
    if not results or not results[0].boxes:
        print("  No footballs detected.")
        if SAVE_IMAGES: # Save the original image if no detections and saving is enabled
            img = cv2.imread(image_path)
            if img is not None:
                 output_path = os.path.join(OUTPUT_FOLDER, image_name)
                 cv2.imwrite(output_path, img)
            else:
                 print(f"  Could not read image {image_name} for saving.")
        continue

    # Load image with OpenCV only if we need to draw or save
    img = None
    if SAVE_IMAGES:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Warning: Could not read image {image_name} with OpenCV. Skipping drawing/saving.")
            SAVE_IMAGES = False # Disable saving for this image if it can't be read

    detected_balls_in_frame = []

    # Extract bounding boxes and calculate ground contact points
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
    confs = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    print(f"  Detected {len(boxes)} football(s).")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box) # Convert coordinates to integers
        confidence = confs[i]

        # Calculate bottom-center point (approximate ground contact)
        bottom_center_x = int((x1 + x2) / 2)
        bottom_center_y = int(y2) # The bottom y-coordinate

        ball_info = {
            'image': image_name,
            'bbox': [x1, y1, x2, y2],
            'confidence': float(confidence),
            'ground_contact_point': (bottom_center_x, bottom_center_y)
        }
        detected_balls_in_frame.append(ball_info)
        detection_results.append(ball_info) # Add to overall results

        print(f"    - Confidence: {confidence:.2f}, BBox: [{x1}, {y1}, {x2}, {y2}], Ground Point: ({bottom_center_x}, {bottom_center_y})")

        # Draw on image (if saving is enabled and image was loaded)
        if SAVE_IMAGES and img is not None:
            # Draw bounding box (green)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw ground contact point (red circle)
            # cv2.circle(img, (bottom_center_x, bottom_center_y), 5, (0, 0, 255), -1)
            # Put label
            label = f"Football: {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10 if y1 > 10 else y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with drawings (if saving is enabled)
    if SAVE_IMAGES and img is not None:
        output_path = os.path.join(OUTPUT_FOLDER, image_name)
        cv2.imwrite(output_path, img)
        print(f"  Saved annotated image to: {output_path}")

# --- Post-processing ---
end_time = time.time()
print("\n--- Processing Complete ---")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(f"Total football detections across all images: {len(detection_results)}")

# You can now use the 'detection_results' list for further analysis.
# For example, print the results for the first few detections:
print("\n--- Sample Detection Results ---")
for i, result in enumerate(detection_results[:5]): # Print first 5 results
    print(f"{i+1}: {result}")

# Optional: Save results to a JSON file
import json
results_json_path = 'football_detections.json'
with open(results_json_path, 'w') as f:
    json.dump(detection_results, f, indent=4)
print(f"\nSaved detailed detection results to: {results_json_path}")