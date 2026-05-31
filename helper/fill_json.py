import os
import json
import numpy as np

# -------------------------
# SETTINGS
# -------------------------
num_keypoints = 17
json_folder = "E:/master/split_output/val"
depth_folder = "E:/master/split_output/val"
output_json_folder = "E:/master/split_output/val_filled"
os.makedirs(output_json_folder, exist_ok=True)

#outdated file just here for transparency
#outdated file just here for transparency
#outdated file just here for transparency
#outdated file just here for transparency
#outdated file just here for transparency

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def interpolate_keypoints(kpts_prev, kpts_next, alpha):
    return kpts_prev + alpha * (kpts_next - kpts_prev)

def interpolate_boxes(box_prev, box_next, alpha):
    return box_prev + alpha * (box_next - box_prev)

def interpolate_depth(d_prev, d_next, alpha):
    return d_prev + alpha * (d_next - d_prev)

def fill_array(arr, interp_fn):
    """Fill missing values in arr (list) using linear interpolation if possible."""
    for i in range(len(arr)):
        if arr[i] is not None:
            continue
        # find previous
        prev_idx = i-1
        while prev_idx >= 0 and arr[prev_idx] is None:
            prev_idx -= 1
        # find next
        next_idx = i+1
        while next_idx < len(arr) and arr[next_idx] is None:
            next_idx += 1
        # interpolate if possible
        if prev_idx >=0 and next_idx < len(arr):
            alpha = (i - prev_idx) / (next_idx - prev_idx)
            arr[i] = interp_fn(arr[prev_idx], arr[next_idx], alpha)
        elif prev_idx >=0:
            arr[i] = arr[prev_idx].copy() if isinstance(arr[prev_idx], np.ndarray) else arr[prev_idx]
        elif next_idx < len(arr):
            arr[i] = arr[next_idx].copy() if isinstance(arr[next_idx], np.ndarray) else arr[next_idx]
        else:
            # fallback zeros
            if isinstance(arr[i], np.ndarray) or arr[i] is None:
                arr[i] = np.zeros_like(arr[i] if arr[i] is not None else np.zeros((num_keypoints,2),dtype=np.float32))
            else:
                arr[i] = 0.0

def fill_sequence(frames, depth_stack, num_keypoints=17):
    """
    Fill missing keypoints, boxes, and depth.
    depth_stack: numpy array of shape (num_depth_frames, H, W)
    Returns:
        filled_frames: list of dicts for JSON, each frame has 'depth' added
    """
    max_frame = max(f["frame_number"] for f in frames)
    filled_frames = []

    # Prepare dict for fast lookup
    frame_dict = {f["frame_number"]: f for f in frames}

    kpts_all, boxes_all, depth_all = [], [], []

    # Prepare arrays
    for idx in range(1, max_frame+1):
        f = frame_dict.get(idx)
        # keypoints
        if f is None or len(f.get("keypoints", []))==0 or len(f["keypoints"][0])==0:
            kpts_all.append(None)
            boxes_all.append(None)
        else:
            kpts = np.array(f["keypoints"][0], dtype=np.float32)
            if kpts.shape[0] < num_keypoints:
                filled = np.zeros((num_keypoints,2), dtype=np.float32)
                filled[:kpts.shape[0]] = kpts
                kpts = filled
            kpts_all.append(kpts)

            # boxes
            if f.get("boxes") and len(f["boxes"])>0:
                boxes_all.append(np.array(f["boxes"][0], dtype=np.float32))
            else:
                boxes_all.append(None)

        # depth
        if depth_stack is not None and idx - 1 < len(depth_stack):
            depth_image = depth_stack[idx - 1]
            if f and f.get("boxes") and len(f["boxes"]) > 0:
                bbox = f["boxes"][0]
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(depth_image.shape[1] - 1, x2), min(depth_image.shape[0] - 1, y2)
                patch = depth_image[y1:y2 + 1, x1:x2 + 1]
                depth_all.append(np.median(patch) / 1000.0 if patch.size > 0 else None)
            else:
                depth_all.append(None)
        else:
            depth_all.append(None)

    # Fill missing values
    fill_array(kpts_all, interpolate_keypoints)
    fill_array(boxes_all, interpolate_boxes)
    fill_array(depth_all, interpolate_depth)

    # Build filled frames for JSON & store depth per frame
    for idx in range(max_frame):
        kpts_list = [k.tolist() for k in kpts_all[idx]]
        box_list = boxes_all[idx].tolist() if boxes_all[idx] is not None else []
        d_val = float(depth_all[idx])
        filled_frames.append({
            "frame_number": idx+1,
            "name": {"0":"person"},
            "boxes": [box_list] if box_list else [],
            "keypoints": [kpts_list],
            "depth": d_val
        })

    return filled_frames

# -------------------------
# PROCESS ALL JSONS
# -------------------------
for f_name in os.listdir(json_folder):
    if not f_name.endswith(".json"):
        continue

    base_name = f_name.replace("_rgxb.json", "")
    json_path = os.path.join(json_folder, f_name)
    depth_path = os.path.join(depth_folder, f"{base_name}_depth.npy")

    with open(json_path, "r") as jf:
        data = json.load(jf)

    # Load full depth stack if it exists
    depth_stack = np.load(depth_path) if os.path.exists(depth_path) else None

    filled_annotations = fill_sequence(data["annotations"], depth_stack, num_keypoints=num_keypoints)

    # Save filled JSON with depth per frame
    output_json_path = os.path.join(output_json_folder, f_name)
    output_data = {
        "video_name": data["video_name"],
        "annotations": filled_annotations
    }
    with open(output_json_path, "w") as of:
        json.dump(output_data, of, indent=4)

    print(f"Processed {base_name}, frames: {len(filled_annotations)}")

print("All sequences preprocessed. Depth per frame is now stored in the JSON.")
