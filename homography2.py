import cv2
import numpy as np
import os

# --- Configuration ---
# 1. Define the real-world coordinates (X, Y) of the points you will click on.
#    Make sure the order here MATCHES the order you click the points in the image!
#    Units are in inches, based on the 20x20 tile size.
WORLD_POINTS = np.array([
    [0,   0],   # Point 0: Bottom-left corner (example)
    [40,  0],   # Point 1: 2 tiles right on bottom edge
    [100, 0],   # Point 2: 5 tiles right on bottom edge (end)
    [0,  60],   # Point 3: 3 tiles up on left edge (end)
    [40, 60],   # Point 4: 2 right, 3 up
    [100, 60]    # Point 5: 5 right, 3 up (top-right corner)
], dtype=np.float32)

NUM_POINTS = len(WORLD_POINTS)

FRAME_FOLDER = 'raw_frames'
IMAGE_FILENAME = 'video_A_frame_0297.jpg' # CHANGE THIS to an actual frame filename

# --- Display Configuration ---
# Set a maximum width for the display window in pixels
MAX_DISPLAY_WIDTH = 1200 # Adjust this based on your screen size

# List to store clicked image points (in original image coordinates)
image_points_collected = []
click_count = 0
scale_factor = 1.0 # To store the calculated scale factor

# --- Mouse Callback Function ---
def click_event(event, x, y, flags, params):
    """
    Callback function to handle mouse clicks.
    (x, y) received here are relative to the RESIZED display window.
    """
    global image_points_collected, click_count, img_display, scale_factor, img_resized_for_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count < NUM_POINTS:
            # Scale coordinates back to the original image size
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)

            # Store the ORIGINAL coordinates
            image_points_collected.append([original_x, original_y])
            print(f"Point {click_count}: World = {WORLD_POINTS[click_count]}, Clicked(Resized) = ({x}, {y}), Stored(Original) = ({original_x}, {original_y})")

            # Draw feedback on the RESIZED image using the clicked (x, y)
            cv2.circle(img_resized_for_display, (x, y), 5, (0, 255, 0), -1) # Green circle
            cv2.putText(img_resized_for_display, str(click_count), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA) # Red number

            # Update the display to show the drawn point
            cv2.imshow("Select Points - Click in Order! (Resized)", img_resized_for_display)

            click_count += 1
        else:
            print("All points collected. Press any key to exit.")

# --- Main Script ---
image_path = os.path.join(FRAME_FOLDER, IMAGE_FILENAME)

# Load the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image '{image_path}'")
    print("Please ensure the FRAME_FOLDER and IMAGE_FILENAME are correct.")
    exit()

# Get original dimensions
original_height, original_width = img.shape[:2]

# Calculate scaling factor
if original_width > MAX_DISPLAY_WIDTH:
    scale_factor = MAX_DISPLAY_WIDTH / original_width
else:
    scale_factor = 1.0 # No scaling needed if image is already small enough

# Calculate display dimensions
display_width = int(original_width * scale_factor)
display_height = int(original_height * scale_factor)

# Create a copy for drawing feedback (we'll resize this for display)
img_display = img.copy()

# Create a resizable window and set the mouse callback
WINDOW_NAME = "Select Points - Click in Order! (Resized)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Use WINDOW_NORMAL for resizability
cv2.setMouseCallback(WINDOW_NAME, click_event)

# Display instructions
print(f"--- Image Info ---")
print(f"Original Size: {original_width} x {original_height}")
print(f"Display Size:  {display_width} x {display_height} (Scale: {scale_factor:.3f})")
print(f"\nPlease click on the {NUM_POINTS} grid points in the image window in this EXACT order:")
for i, wp in enumerate(WORLD_POINTS):
    print(f"  Click {i}: Corresponding to World Point {wp} inches")
print("\nClick coordinates will be scaled to original image size.")
print("Close the image window by pressing any key AFTER clicking all points.")

# Resize the image for the initial display
img_resized_for_display = cv2.resize(img_display, (display_width, display_height), interpolation=cv2.INTER_AREA)
cv2.imshow(WINDOW_NAME, img_resized_for_display)

cv2.waitKey(0) # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()

# --- Verification and Output ---
if len(image_points_collected) == NUM_POINTS:
    # Convert collected points to a NumPy array
    image_points_np = np.array(image_points_collected, dtype=np.float32)

    print("\n--- Collected Points ---")
    print("World Points (X, Y) in inches:")
    print(WORLD_POINTS)
    print("\nCorresponding Image Points (u, v) in ORIGINAL pixel coordinates:")
    print(image_points_np)

    # TODO later: Save these points if needed, e.g., using np.savez
    np.savez("homography_points.npz", world_points=WORLD_POINTS, image_points=image_points_np)
    print("\nThese two arrays (World Points and Image Points) are the input for the next step: calculating the homography matrix.")
else:
    print(f"\nError: Expected {NUM_POINTS} points, but only collected {len(image_points_collected)}. Please run again.")