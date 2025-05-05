import cv2
import numpy as np
import os
import glob
# import rawpy # Uncomment if using rawpy for DNG

# --- Configuration ---
# Define the dimensions of the checkerboard (internal corners)
CHECKERBOARD_WIDTH = 9  # Number of internal corners along width
CHECKERBOARD_HEIGHT = 6 # Number of internal corners along height
# Define the real-world size of each square side (e.g., in millimeters)
SQUARE_SIZE_MM = 24.0
# Path to the calibration images (use wildcard * for multiple files)
# IMPORTANT: Ensure these images are in a format cv2.imread can handle (PNG, TIFF, JPG)
# OR uncomment and adapt the rawpy section below if using DNG directly.
IMAGE_PATH_PATTERN = './calibration_images2/*.jpg' # ADJUST THIS PATH

# --- Setup ---
# Criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on checkerboard dimensions and square size
# These are the theoretical 3D coordinates of the corners in the checkerboard plane (Z=0)
objp = np.zeros((CHECKERBOARD_HEIGHT * CHECKERBOARD_WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_WIDTH, 0:CHECKERBOARD_HEIGHT].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM # Scale to real-world size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# --- Image Processing ---
images = glob.glob(IMAGE_PATH_PATTERN)

if not images:
    print(f"Error: No images found at '{IMAGE_PATH_PATTERN}'. Please check the path.")
    exit()

print(f"Found {len(images)} images. Processing...")

image_size = None # To store image dimensions

for fname in images:
    print(f"Processing {fname}...")

    # === Option A: Load standard image format (PNG, TIFF, JPG) ===
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}. Skipping.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # === End Option A ===

    # # === Option B: Load DNG using rawpy (Uncomment and adapt if needed) ===
    # try:
    #     with rawpy.imread(fname) as raw:
    #         # Postprocess: Use options for minimal processing, e.g., linear output if possible
    #         # Note: Getting true linear output might need specific options depending on camera/DNG
    #         # rgb = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, user_flip=0) # Example options
    #         rgb = raw.postprocess(use_camera_wb=True, output_bps=8, no_auto_bright=True, user_flip=0) # 8-bit for easier cvtColor
    #     img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # rawpy gives RGB, OpenCV uses BGR
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     print("DNG processed using rawpy.")
    # except Exception as e:
    #     print(f"Warning: Could not process DNG file {fname} with rawpy: {e}. Skipping.")
    #     continue
    # # === End Option B ===


    if image_size is None:
        image_size = gray.shape[::-1] # Get (width, height)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"  Corners found in {fname}")
        objpoints.append(objp)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # --- Optional: Draw and display corners for verification ---
        cv2.drawChessboardCorners(img, (CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT), corners2, ret)
        scale_factor = 0.25  # Adjust this value to control the resizing (e.g., 0.5 for 50%)
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height))
        cv2.imshow(f'Corners found in {os.path.basename(fname)}', resized_img)
        cv2.waitKey(500) # Display for 0.5 seconds
        cv2.destroyAllWindows() # Close windows if display was used
        # --- End Optional ---

    else:
        print(f"  Warning: Corners not found in {fname}. Skipping this image.")

cv2.destroyAllWindows() # Close windows if display was used

if not objpoints or not imgpoints:
     print("Error: No valid checkerboard corners detected in any image. Cannot calibrate.")
     exit()

if image_size is None:
    print("Error: Could not determine image size.")
    exit()

print(f"\nProcessed {len(objpoints)} images successfully.")
print("Running calibration...")

# --- Camera Calibration ---
# cv2.calibrateCamera returns:
# ret: Overall RMS re-projection error
# mtx: Intrinsic camera matrix (3x3)
# dist: Distortion coefficients (k1, k2, p1, p2, k3[, k4, k5, k6])
# rvecs: Rotation vectors (1 per image) - part of Extrinsics
# tvecs: Translation vectors (1 per image) - part of Extrinsics
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

if ret:
    print("\nCalibration successful!")
    print("--------------------------------------")

    # --- Intrinsic Parameters ---
    print("Intrinsic Camera Matrix (K):")
    print(mtx)
    print("\nFocal Length (fx, fy):", mtx[0, 0], mtx[1, 1])
    print("Principal Point (cx, cy):", mtx[0, 2], mtx[1, 2])
    print("--------------------------------------")

    # --- Distortion Coefficients ---
    # Format: (k1, k2, p1, p2, [k3, [k4, k5, k6]])
    # k1, k2, k3: Radial distortion coefficients
    # p1, p2: Tangential distortion coefficients
    print("Distortion Coefficients:")
    print(dist)
    print("--------------------------------------")

    # --- Extrinsic Parameters ---
    # rvecs and tvecs are lists, with one entry for each calibration image.
    # Each entry represents the transformation from the World (checkerboard) coordinate system
    # to the Camera coordinate system for that specific image view.
    print(f"Extrinsic Parameters (Rotation and Translation Vectors) for each of the {len(rvecs)} images:")
    for i in range(len(rvecs)):
        print(f"\n--- Image {i+1} ---")
        # Convert rotation vector to rotation matrix (Optional)
        R, _ = cv2.Rodrigues(rvecs[i])
        print(f"Rotation Vector (rvec {i}):\n{rvecs[i]}")
        # print(f"Rotation Matrix (R {i}):\n{R}") # Uncomment to print matrix
        print(f"Translation Vector (tvec {i} in mm/chosen unit):\n{tvecs[i]}") # Units depend on SQUARE_SIZE_MM units

    print("--------------------------------------")

    # --- Accuracy Assessment ---
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Total Average Re-projection Error: {mean_error / len(objpoints)} pixels")
    print("(Lower is better, typically < 1.0 pixels for good calibration)")
    print("--------------------------------------")

    # --- Saving Calibration Data (Optional) ---
    # You might want to save these parameters for later use
    # np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    # print("Calibration data saved to camera_calibration_data.npz")
    # print("--------------------------------------")


else:
    print("\nCalibration failed.")