import cv2
import numpy as np
import json
import csv
import os
import time # To add timestamp for context

# --- Configuration ---
HOMOGRAPHY_FILENAME = "homography_matrix.npy"
JSON_FILENAME = "football_detections.json" # Your JSON file with detections
INPUT_FRAME_FOLDER = "detected_frames" # Folder with images listed in JSON
OUTPUT_FRAME_FOLDER = "output_frames_with_box" # New folder for output
CSV_FILENAME = "world_coordinates.csv" # Output CSV filename (only ball coords)

# --- Fixed Pixel Points for the Box ---
# Define the four points P1, P2, P3, P4 in pixel coordinates
FIXED_POINTS_PIXELS = {
    "P1": (627, 545),
    "P2": (1026, 674),
    "P3": (576, 1329), # Note: Y seems large, ensure it's within image bounds
    "P4": (68, 970)
}
# Convert to NumPy array for cv2.polylines later (shape needs to be Nx1x2)
box_vertices = np.array([FIXED_POINTS_PIXELS["P1"],
                         FIXED_POINTS_PIXELS["P2"],
                         FIXED_POINTS_PIXELS["P3"],
                         FIXED_POINTS_PIXELS["P4"]], dtype=np.int32)
box_vertices = box_vertices.reshape((-1, 1, 2))

# --- Annotation Settings ---
# Ball annotations
BALL_DOT_COLOR = (0, 0, 0) # black
BALL_DOT_RADIUS = 5
BALL_TEXT_COLOR = (0, 0, 0) # black
# Box annotations
BOX_COLOR = (255, 0, 0) # Blue
BOX_THICKNESS = 2
# Fixed point annotations
FIXED_PT_DOT_COLOR = (0, 0, 0) # black
FIXED_PT_DOT_RADIUS = 4
FIXED_PT_TEXT_COLOR = (0, 0, 0) # black
# General text settings
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_THICKNESS = 2
TEXT_OFFSET_X = 10
TEXT_OFFSET_Y = 10

# --- Initialization ---
print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}") # Added Timezone
print(f"Current location context: Bengaluru, Karnataka, India")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FRAME_FOLDER, exist_ok=True)
print(f"Output frames will be saved in: '{OUTPUT_FRAME_FOLDER}'")
print(f"Ball world coordinates will be saved to: '{CSV_FILENAME}'")

# 1. Load Homography Matrix H
try:
    H = np.load(HOMOGRAPHY_FILENAME)
    print(f"\nLoaded Homography matrix H from '{HOMOGRAPHY_FILENAME}'")
    np.set_printoptions(suppress=True, precision=4) # Pretty print
    print("H Matrix:")
    print(H)
except FileNotFoundError:
    print(f"\nError: Homography matrix file '{HOMOGRAPHY_FILENAME}' not found.")
    exit()
except Exception as e:
    print(f"\nError loading homography matrix: {e}")
    exit()

# 2. Calculate Inverse Homography H_inv
try:
    H_inv = np.linalg.inv(H)
    print("\nCalculated Inverse Homography Matrix (H_inv):")
    print(H_inv)
except np.linalg.LinAlgError:
    print("\nError: Could not compute the inverse of the Homography matrix H.")
    exit()

# 3. Load Detection Data from JSON
try:
    with open(JSON_FILENAME, 'r') as f:
        detection_data = json.load(f)
    print(f"\nLoaded {len(detection_data)} detections from '{JSON_FILENAME}'")
    if not isinstance(detection_data, list):
         print("Error: JSON data should be a list of detection dictionaries.")
         exit()
except FileNotFoundError:
    print(f"\nError: Detection JSON file '{JSON_FILENAME}' not found.")
    exit()
except json.JSONDecodeError as e:
    print(f"\nError decoding JSON file '{JSON_FILENAME}': {e}")
    exit()
except Exception as e:
    print(f"\nAn unexpected error occurred loading JSON: {e}")
    exit()

# 4. Setup CSV Output (Only for Ball)
try:
    csv_file = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Write Header
    csv_writer.writerow(['Frame_name', 'X', 'Y']) # Changed header slightly
except IOError as e:
     print(f"\nError opening CSV file '{CSV_FILENAME}' for writing: {e}")
     exit()

print("\nStarting processing of frames...")
# --- Processing Loop ---
processed_count = 0
error_count = 0
for i, det in enumerate(detection_data):
    # --- Basic validation ---
    if not all(key in det for key in ['image', 'ground_contact_point']):
        print(f"Warning: Skipping entry {i+1} due to missing 'image' or 'ground_contact_point' key.")
        error_count += 1
        continue

    image_filename = det['image']
    ground_contact_point = det['ground_contact_point']

    if not (isinstance(ground_contact_point, (list, tuple)) and len(ground_contact_point) == 2):
         print(f"Warning: Skipping {image_filename} due to invalid 'ground_contact_point' format: {ground_contact_point}")
         error_count += 1
         continue

    u_ball, v_ball = ground_contact_point

    # --- Load image ---
    image_path = os.path.join(INPUT_FRAME_FOLDER, image_filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image '{image_path}'. Skipping.")
        error_count += 1
        continue

    # --- Process Ball ---
    x_ball_world, y_ball_world = float('nan'), float('nan') # Default to NaN
    try:
        # Create homogeneous coordinate vector
        pixel_coords_h = np.array([[u_ball], [v_ball], [1]], dtype=np.float32)
        # Apply the inverse homography
        world_coords_h = H_inv @ pixel_coords_h
        # Normalize
        w = world_coords_h[2, 0]
        if abs(w) >= 1e-7: # Check for non-zero w
            x_ball_world = world_coords_h[0, 0] / w
            y_ball_world = world_coords_h[1, 0] / w
        else:
             print(f"Warning: Ball mapping failed for {image_filename} (w near zero).")
             error_count += 1

        # Write ball coordinates to CSV
        csv_writer.writerow([image_filename, f"{x_ball_world:.2f}" if not np.isnan(x_ball_world) else "NaN", f"{y_ball_world:.2f}" if not np.isnan(y_ball_world) else "NaN"])

        # Annotate Ball on Image (if mapping successful)
        if not np.isnan(x_ball_world):
            cv2.circle(img, (int(u_ball), int(v_ball)), BALL_DOT_RADIUS, BALL_DOT_COLOR, -1)
            ball_text = f"B:({x_ball_world:.1f}, {y_ball_world:.1f})"
            cv2.putText(img, ball_text, (int(u_ball) + TEXT_OFFSET_X, int(v_ball) + TEXT_OFFSET_Y),
                        TEXT_FONT, TEXT_SCALE, BALL_TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

    except Exception as e:
        print(f"Error processing ball for {image_filename}: {e}")
        error_count += 1
        csv_writer.writerow([image_filename, "Error", "Error"]) # Mark error in CSV

    # --- Process Fixed Points and Draw Box ---
    # Draw the box connecting the fixed points
    cv2.polylines(img, [box_vertices], isClosed=True, color=BOX_COLOR, thickness=BOX_THICKNESS)

    # Map and annotate each fixed point
    for name, point_pixel in FIXED_POINTS_PIXELS.items():
        u_pt, v_pt = point_pixel
        try:
            # Map pixel to world
            pixel_coords_h = np.array([[u_pt], [v_pt], [1]], dtype=np.float32)
            world_coords_h = H_inv @ pixel_coords_h
            w = world_coords_h[2, 0]

            if abs(w) >= 1e-7:
                x_pt_world = world_coords_h[0, 0] / w
                y_pt_world = world_coords_h[1, 0] / w

                # Annotate fixed point on image
                cv2.circle(img, (int(u_pt), int(v_pt)), FIXED_PT_DOT_RADIUS, FIXED_PT_DOT_COLOR, -1)
                pt_text = f"{name}:({x_pt_world:.1f}, {y_pt_world:.1f})"
                # Adjust text position slightly for each point if needed, basic offset here
                cv2.putText(img, pt_text, (int(u_pt) + TEXT_OFFSET_X, int(v_pt) + TEXT_OFFSET_Y),
                            TEXT_FONT, TEXT_SCALE, FIXED_PT_TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
            else:
                 print(f"Warning: Fixed point {name} mapping failed for {image_filename} (w near zero).")
                 # Draw dot but maybe indicate mapping failure?
                 cv2.circle(img, (int(u_pt), int(v_pt)), FIXED_PT_DOT_RADIUS, (0,0,0), -1) # Black dot for fail?

        except Exception as e:
            print(f"Error processing fixed point {name} for {image_filename}: {e}")
            # Optionally draw a different marker if mapping fails
            cv2.circle(img, (int(u_pt), int(v_pt)), FIXED_PT_DOT_RADIUS, (255, 255, 255), -1) # White dot for error?


    # --- Save Annotated Image ---
    try:
        output_path = os.path.join(OUTPUT_FRAME_FOLDER, image_filename)
        success = cv2.imwrite(output_path, img)
        if not success:
             print(f"Warning: Failed to save annotated image '{output_path}'")
             error_count += 1
        processed_count += 1 # Increment only if save is attempted
    except Exception as e:
         print(f"Error saving image {output_path}: {e}")
         error_count += 1


    if (processed_count + error_count) % 10 == 0: # Print progress
         print(f"  Processed {processed_count + error_count}/{len(detection_data)} entries...")


# --- Finalization ---
try:
    csv_file.close() # Close the CSV file
except Exception as e:
    print(f"Warning: Error closing CSV file: {e}")

print(f"\n--- Processing Complete ---")
print(f"Total entries in JSON: {len(detection_data)}")
print(f"Successfully processed images (saved/attempted save): {processed_count}")
print(f"Entries skipped or with errors during processing: {error_count}")
print(f"Annotated images saved in: '{OUTPUT_FRAME_FOLDER}'")
print(f"Ball world coordinates saved to: '{CSV_FILENAME}'")
print(f"Script finished at: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
