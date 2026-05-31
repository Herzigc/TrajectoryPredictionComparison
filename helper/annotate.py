from ultralytics import YOLO
import os
import json

# Load the model and run the tracker with a custom configuration file
model = YOLO("yolo11n-pose.pt")

#root = '../data/final' # old path most mp4 data not included this script is just for transperancy
#root = './data/test/'

for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith(".mp4"):
            results = model.track(source=os.path.abspath(os.path.join(root, file)), show=False, tracker="bytetrack.yaml",
                                  imgsz=2880, iou=0.45, conf=0.60)
            video_annotations={
                'video_name': os.path.basename(file),
                'annotations': []
            }
            frame_count: int = 1
            for r in results:
                annotations = {'frame_number': frame_count,
                               'name': r.names,
                               'boxes': r.boxes.xyxy.tolist() if hasattr(r.boxes.xyxy, "tolist") else r.boxes.xyxy,
                               'keypoints': r.keypoints.xy.tolist() if hasattr(r.keypoints.xy, "tolist") else r.keypoints.xy
                               }
                frame_count += 1
                video_annotations['annotations'].append(annotations)

            json_path = os.path.join(os.path.dirname(os.path.abspath(os.path.join(root, file))),
                                   os.path.splitext((os.path.basename(file)))[0] + '.json')
            with open(json_path, 'w') as out:
                json.dump(video_annotations, out, indent=4)
