import os
import json
import random

# Define source and target directories
source_dir = "coco_grayscale"
target_dir = "coco_original"
output_file = "dataset.json"

# Define prompts
prompts = [
    "Add colors to this image",
    "Give realistic colors to this image",
    "Add realistic colors to this image",
    "Colorize this image",
    "Restore the original colors of this image",
    "Colorize this black and white image",
    "Restore the original colors of this image",
    "Add natural colors to this grayscale image",
    "Give realistic colors to this old photo",
    "Create the original colors of this image",
]

# Get list of images in the source directory
image_files = sorted(os.listdir(source_dir))  # Ensure consistent ordering

data = []

# Generate JSON entries
for filename in image_files:
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    prompt = random.choice(prompts) 
    
    entry = {
        "source": source_path,
        "target": target_path,
        "prompt": prompt
    }
    data.append(entry)

# Write to JSON file
with open(output_file, "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")

print(f"JSON file '{output_file}' created successfully.")