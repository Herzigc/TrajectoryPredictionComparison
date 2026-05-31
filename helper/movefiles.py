import os
import random
import shutil
from pathlib import Path

#outdated file just here for transparency
#outdated file just here for transparency
#outdated file just here for transparency
#outdated file just here for transparency
#outdated file just here for transparency

root = "E:/master/final"
output_root = "E:/master/split_output"

# Create split folders
for split in ["train", "val", "test"]:
    os.makedirs(Path(output_root) / split, exist_ok=True)

pairs = []

def get_base_name(filename):
    """Return everything before the FINAL underscore."""
    return "_".join(filename.split("_")[:-1])

for person_folder in Path(root).iterdir():
    if not person_folder.is_dir():
        continue

    person_name = person_folder.name
    files = list(person_folder.iterdir())

    # Collect depth files and json files
    depth_dict = {}
    json_dict = {}

    for file in files:
        fname = file.name

        # Detect depth file: NAME_depth.npy
        if fname.endswith("_depth.npy"):
            base = get_base_name(fname)
            depth_dict[base] = file

        # Detect JSON: NAME_rgb.json or NAME_rgxb.json
        elif fname.endswith("_rgb.json") or fname.endswith("_rgxb.json"):
            base = get_base_name(fname)
            json_dict[base] = file

    # Match depth with json by base name
    for base, depth_file in depth_dict.items():
        if base in json_dict:   # found matching json
            pairs.append((depth_file, json_dict[base], person_name))

print(f"Found {len(pairs)} valid pairs.\n")

# Shuffle
random.seed(42)
random.shuffle(pairs)

# Split 70/15/15
N = len(pairs)
train_end = int(0.7 * N)
val_end   = train_end + int(0.15 * N)

splits = {
    "train": pairs[:train_end],
    "val":   pairs[train_end:val_end],
    "test":  pairs[val_end:]
}

# Copy and rename
for split_name, pair_list in splits.items():
    for depth_file, json_file, person_name in pair_list:

        # base name again
        base = get_base_name(depth_file.name)

        # New filenames
        new_depth = f"{person_name}_{base}_depth.npy"
        new_json  = f"{person_name}_{json_file.name}"

        # Destination paths
        dest_dir = Path(output_root) / split_name
        shutil.copy2(depth_file, dest_dir / new_depth)
        shutil.copy2(json_file, dest_dir / new_json)

print("Done!")
print(f"Train pairs: {len(splits['train'])}")
print(f"Val pairs:   {len(splits['val'])}")
print(f"Test pairs:  {len(splits['test'])}")
