import os

import cv2
from icecream import ic
from tqdm import tqdm
from ultralytics import YOLO


def perform_yolo_on_image(image_path, output_path="detected_image.jpg", model_name='yolov8n.pt', confidence_threshold=0.5):
    """
    Performs YOLO object detection on a given image and saves the output with bounding boxes.

    Args:
        image_path (str): Path to the input image file.
        output_path (str, optional): Path to save the output image with detections.
                                     Defaults to "detected_image.jpg".
        model_name (str, optional): Name of the YOLO model to use.
                                     Defaults to 'yolov8n.pt' (small and fast).
        confidence_threshold (float, optional): Minimum confidence score for detected objects.
                                             Defaults to 0.5.
    """
    try:
        # Load the YOLO model
        model = YOLO(model_name)

        # Read the input image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or find image file: {image_path}")
            return

        # Perform YOLO detection on the image
        results = model(image, conf=confidence_threshold, verbose=False)

        # Draw bounding boxes and labels on the image
        annotated_image = results[0].plot()

        # Save the annotated image
        cv2.imwrite(output_path, annotated_image)
        # print(f"Object detection complete. Output image saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    video_title: str = "test_video2"
    input_dir: str = f"data/frames/{video_title}"
    output_dir: str = f"data/tagged_frames/{video_title}"
    model: str = 'yolov8x.pt'
    confidence = 0.5

    os.makedirs(output_dir, exist_ok=True)

    for image_file in tqdm(os.listdir(input_dir), desc="YOLO Prediction of Images"):
        image_file_path: str = os.path.join(input_dir, image_file)
        output_file_path: str = f"{output_dir}/{image_file}"

        try:
            confidence_threshold = float(confidence)
            if not (0.0 <= confidence_threshold <= 1.0):
                print("Invalid confidence threshold. Using default 0.5.")
                confidence_threshold = 0.5
        except ValueError:
            print("Invalid confidence threshold. Using default 0.5.")
            confidence_threshold = 0.5

        perform_yolo_on_image(image_file_path, output_file_path, model, confidence_threshold)