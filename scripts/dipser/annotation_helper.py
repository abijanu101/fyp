
import cv2
import json
import tempfile
import zipfile

from pathlib import Path
import random


DIPSER_PATH = Path(r"E:\Coding\Projects\fyp\data\DIPSER")

GROUP = "group_01"
EXPERIMENT = "experiment_08"
CAMERA = "context_cam_02"

zip_path = DIPSER_PATH / 'DIPSEER' / GROUP / EXPERIMENT / f"{CAMERA}.zip"
annotations_path = DIPSER_PATH / 'abi_labels' / GROUP / f'{EXPERIMENT}.json'

with open(annotations_path, "r") as f:
    annotations = json.load(f)

camera_annotations = annotations[CAMERA]

# Temporary extraction
temp_dir = tempfile.TemporaryDirectory()

with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall(temp_dir.name)

extracted_path = Path(temp_dir.name)
images_dir = next(extracted_path.iterdir())
images = sorted(images_dir.glob("*.png"))

current_index = random.randrange(len(images))

def load_image(index):
    global img, current_index

    current_index = index % len(images)
    img = cv2.imread(str(images[current_index]))

    if img is None:
        raise RuntimeError(f"Could not load image: {images[current_index]}")

    print(f"Sampled frame: {images[current_index].name}")

load_image(current_index)

# Annotation state
corners = []
preview_box = None

colors = [
    (0, 255, 0),       # Green
    (255, 100, 0),     # Blue-orange
    (0, 100, 255),     # Orange
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Yellow
    (255, 255, 0),     # Cyan
    (128, 0, 255),     # Violet
    (0, 165, 255),     # Orange
    (255, 0, 128),     # Pink
    (128, 255, 0),     # Lime
    (0, 128, 128),     # Olive
    (128, 0, 128),     # Purple
    (255, 128, 0),     # Light blue
    (0, 0, 255),       # Red
    (255, 255, 255),   # White
    (128, 128, 0),     # Teal
    (0, 255, 128),     # Spring green
    (255, 128, 128),   # Light cyan
    (128, 0, 0),       # Dark blue
    (0, 128, 255),     # Amber
]


def visualize():
    output = img.copy()

    # Existing annotations
    for i, (subject_id, bbox) in enumerate(camera_annotations.items()):
        x, y = bbox["x"], bbox["y"]
        w, h = bbox["w"], bbox["h"]
        color = colors[i % len(colors)]

        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            output, subject_id, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    # Current rectangle being drawn
    if preview_box:
        x1, y1, x2, y2 = preview_box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output


def on_mouse(event, x, y, flags, param):
    global preview_box

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        corners.append((x, y))

        if len(corners) == 1:
            print(f"First corner: ({x}, {y})")

        elif len(corners) == 2:
            (x1, y1), (x2, y2) = corners

            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            preview_box = (
                x_min, y_min, x_max, y_max
            )

            print(
                'Bounding box: {"x":', x_min, ', "y":', y_min, 
                ', "w":', x_max-x_min, ', "h":', y_max-y_min, "}", sep=''
            )

            corners.clear()


cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotation", 800, 600)
cv2.setMouseCallback("Annotation", on_mouse)

print("Left click: Get coordinates")
print("Right click twice: Draw bounding box")
print("R: Clear temporary rectangle")
print("Q / ESC: Exit")
print("N: Next image")
print("P: Previous image")

while True:
    cv2.imshow("Annotation", visualize())

    key = cv2.waitKey(20) & 0xFF

    if key == ord("q") or key == 27:
        break

    elif key == ord("r"):
        corners.clear()
        preview_box = None

    elif key == ord("n"):
        load_image(current_index + 1)
        corners.clear()
        preview_box = None

    elif key == ord("p"):
        load_image(current_index - 1)
        corners.clear()
        preview_box = None

cv2.destroyAllWindows()
temp_dir.cleanup()
