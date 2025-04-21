import cv2
import numpy as np
from icecream import ic

# --- Configuration ---
# Option 1: If you have the arrays directly from the previous script's run
# Make sure these variables hold the NumPy arrays you got previously
# WORLD_POINTS = ... (Your Nx2 world points array)
# image_points_np = ... (Your Nx2 image points array)

# Option 2: Load the points from the saved .npz file (RECOMMENDED)
try:
    data = np.load("homography_points.npz")
    WORLD_POINTS = data['world_points']
    image_points_np = data['image_points']
    print("Loaded points successfully from homography_points.npz")
    print("World Points shape:", WORLD_POINTS.shape)
    print("Image Points shape:", image_points_np.shape)
except FileNotFoundError:
    print("Error: homography_points.npz not found.")
    print("Please run the previous script (step1_select_points.py) first,")
    print("or make sure the variables WORLD_POINTS and image_points_np are defined.")
    # Provide dummy data to avoid crashing if needed for testing structure
    # You MUST replace this with your actual data
    # WORLD_POINTS = np.array([[0,0],[20,0],[0,20],[20,20]], dtype=np.float32)
    # image_points_np = np.array([[100,400],[300,405],[110,550],[310,555]], dtype=np.float32)
    exit() # Exit if points couldn't be loaded

# --- Sanity Check ---
if WORLD_POINTS.shape[0] != image_points_np.shape[0] or WORLD_POINTS.shape[0] < 4:
    print(f"Error: Need at least 4 corresponding points. Found {WORLD_POINTS.shape[0]} world points and {image_points_np.shape[0]} image points.")
    exit()

# --- Calculate Homography ---
# cv2.findHomography finds the matrix H that maps srcPoints to dstPoints
# We want H such that: image_points = H * world_points
# So, world_points are the source, image_points are the destination.
# We use cv2.RANSAC for robustness against slight inaccuracies in clicked points.
ransac_threshold = 5.0 # Pixels - allowed reprojection error for RANSAC
ic(WORLD_POINTS,image_points_np)
H, mask = cv2.findHomography(WORLD_POINTS, image_points_np, cv2.RANSAC, ransac_threshold)

# --- Output ---
if H is not None:
    print(f"\n--- Homography Matrix (H) ---")
    print("This matrix maps World Points (inches on grid) to Image Points (pixels)")
    np.set_printoptions(suppress=True) # Make matrix easier to read
    print(H)

    # (Optional) Print RANSAC inlier mask - shows which points were considered good fits
    # num_inliers = np.sum(mask)
    # print(f"\nRANSAC found {num_inliers} inliers out of {len(mask)} points.")
    # print("Inlier mask (1 = inlier, 0 = outlier):")
    # print(mask.T) # Transpose for easier reading

    # TODO later: Save the homography matrix
    np.save("homography_matrix.npy", H)
    print("\nHomography matrix H calculated successfully.")
    print("The next step will involve using the INVERSE of this matrix (H_inv)")
    print("to map points from the image (like the ball) back to world coordinates.")

else:
    print("\nError: Homography calculation failed. Could not find H.")
    print("Check the quality and distribution of your selected points.")
    print("Ensure at least 4 points are non-collinear in both world and image views.")