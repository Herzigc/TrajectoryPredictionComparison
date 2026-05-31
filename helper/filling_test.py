import os
import json
import cv2
import numpy as np

# -------------------------
# SETTINGS
# -------------------------
test_folder = "../data/final/test_filled"
output_folder = "../data/final/visualizations"
os.makedirs(output_folder, exist_ok=True)

width, height = 640, 480
keypoint_color = (0, 255, 0)
bbox_color = (255, 0, 0)
depth_marker_color = (0, 0, 255)
marker_radius = 5

# -------------------------
# PROCESS EACH SEQUENCE
# -------------------------
for f in os.listdir(test_folder):
    if not f.endswith("_rgxb.json"):
        continue

    base = f.replace("_rgxb.json", "")
    video_path = os.path.join(test_folder, base + "_rgxb.mp4")
    json_path = os.path.join(test_folder, f)

    # Load JSON
    with open(json_path) as jf:
        data = json.load(jf)

    # Prepare video writer
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = os.path.join(output_folder, f"{base}_vis.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get annotations for current frame
        ann = next((a for a in data["annotations"] if a["frame_number"] == frame_idx), None)
        if ann is not None:
            # Draw bounding boxes
            for box in ann.get("boxes", []):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

            # Draw keypoints
            for kpts in ann.get("keypoints", []):
                for x, y in kpts:
                    cv2.circle(frame, (int(x), int(y)), 3, keypoint_color, -1)

            # Draw depth marker at center of lower body (mid-hip)
            kpts_array = np.array(ann["keypoints"][0])  # shape (17,2)
            center = ((kpts_array[15] + kpts_array[16]) / 2).astype(int)
            depth = ann.get("depth", 0.0)
            # Use depth to scale marker radius slightly
            scaled_radius = max(3, min(marker_radius, int(marker_radius * (1.0 / depth))))
            cv2.circle(frame, tuple(center), scaled_radius, depth_marker_color, -1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved visualization: {out_path}")

print("All sequences visualized!")
