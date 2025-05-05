# Vision-Tech

This repository is a temporary codebase for storing and sharing Vision tech scripts.

### bounding_box_generator.py

- Generates bounding boxes around detected objects in images or video frames.

### grid_generator.py

- Creates a grid overlay for visual reference or region-based analysis on frames or images.

### obj_detection.py

- Runs object detection using a pre-trained model and returns detected classes and their positions.

### save_frames.py

- Extracts and saves frames from a video source.

### calibration.py
- Input: Checkerboard pattern at different angles, along with dimensions of the checkerboard
- Output: Camera intrinsics (Camera Matrix)

### 📂 Homography Trial

- detect_ball.py: Detect the ball in all frames of your input folder using yolov8
- gen_homography_matrix.py , homography.py: Generate homography matrix and apply the homography to get world co-ordinates.
- generate_final_frames.py: Uses the world coordinates json to tag coordinates to the ball in the image for the final output frames.
