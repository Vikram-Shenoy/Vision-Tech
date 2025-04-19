import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────
FRAMES_DIR   = Path('raw_frames/frames_video_B')     # input folder containing your frames
OUTPUT_DIR   = Path('bounding_box_b')   # output folder for annotated frames
"""
VIDEO A
(618, 554)
(1015, 680)
(569, 1337)
(64, 980)
VIDEO B
(585, 852)
(951, 957)
(507, 1599)
(30, 1222)
"""
# Static rectangle corners (clockwise)
# ─── Replace the x,y placeholders with your pixel coordinates ───
P1 = (585, 852)  # e.g. top‑left
P2 = (951, 957)   # e.g. top‑right
P3 = (507, 1599)   # e.g. bottom‑right
P4 = (30, 1222) # e.g. bottom‑left
# Drawing style
COLOR     = (0, 255, 0)   # BGR color (green here)
THICKNESS = 2             # line width in pixels

# ─── END CONFIG ──────────────────────────────────────────────────────

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare the points array for cv2.polylines: shape = (4, 1, 2)
pts = np.array([P1, P2, P3, P4], dtype=np.int32).reshape((-1, 1, 2))

# Supported image extensions
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

for img_path in tqdm(FRAMES_DIR.iterdir(), desc="Drawing boxes", unit="frames"):
    if not img_path.is_file() or img_path.suffix.lower() not in EXTS:
        continue

    # Load the frame
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Draw the rectangle (quadrilateral)
    cv2.polylines(img, [pts], isClosed=True, color=COLOR, thickness=THICKNESS)

    # Save to bounding_box folder with same filename
    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), img)
