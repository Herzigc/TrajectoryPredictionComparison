import os
import json

# -------------------------
# SETTINGS
# -------------------------
json_folder = "../data/final/"
report_file = "depth_zero_report.txt"

found_any = False

with open(report_file, "w") as report:
    for root, _, files in os.walk(json_folder):
        for f in files:
            if not f.endswith("_rgxb.json"):
                continue
            path = os.path.join(root, f)
            with open(path) as jf:
                data = json.load(jf)
            for ann in data.get("annotations", []):
                if ann.get("depth", 1.0) == 0.0:
                    report.write(f"{path} - frame {ann['frame_number']}\n")
                    found_any = True

if found_any:
    print(f"Found frames with depth = 0.0. See {report_file}")
else:
    print("No depth = 0.0 found in any JSON file.")
