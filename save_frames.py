import os
import cv2


def extract_frames_from_video(video_path: str, output_folder: str) -> None:
    """
    Extracts all frames from the given video and saves them into the output folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory where extracted frames will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Get total frame count for padding frame numbers
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digits = len(str(total_frames)) if total_frames > 0 else 6

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Construct frame filename with zero-padded index
        frame_filename = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{str(frame_idx).zfill(digits)}.jpg"
        )

        cv2.imwrite(frame_filename, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
        frame_idx += 1

    cap.release()


def main():
    source_folder = "raw_video"
    target_root = "raw_frames"

    # Create the root output folder if it doesn't exist
    os.makedirs(target_root, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(source_folder):
        video_path = os.path.join(source_folder, filename)
        # Only process files (skip directories)
        if os.path.isfile(video_path):
            name, _ = os.path.splitext(filename)
            output_folder = os.path.join(target_root, f"frames_{name}")
            extract_frames_from_video(video_path, output_folder)


if __name__ == "__main__":
    main()