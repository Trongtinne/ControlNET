"""
This script downloads and processes images from the COCO dataset.
It loads the dataset, converts images to grayscale, and saves both the original and grayscale images to separate directories.

Dependencies:
- requests
- io.BytesIO
- PIL.Image
- datasets.load_dataset
- os

Workflow:
1. Load the COCO dataset from Hugging Face (`detection-datasets/coco`, train split).
2. Create directories `coco_grayscale` and `coco_original` if they do not exist.
3. Iterate through each image in the dataset:
   - Handle different image formats (URLs, file paths, raw byte data, PIL Image objects).
   - Convert the image to grayscale.
   - Save both original and grayscale images in their respective directories.
   - Handle errors gracefully and print debugging information when needed.

Usage:
Simply run the script to process and save images.

"""

import requests
from io import BytesIO
from PIL import Image
from datasets import load_dataset
import os

# Load COCO dataset
dataset = load_dataset("detection-datasets/coco", split="train")

# Create directories for saving grayscale and original images
os.makedirs("coco_grayscale", exist_ok=True)
os.makedirs("coco_original", exist_ok=True)

def save_images(example, idx):
    try:
        # Debugging info
        print(f"Processing image {idx}, Type: {type(example['image'])}")

        # Check if image is a valid path, URL or byte data
        if isinstance(example['image'], str):  # URL or file path
            if example['image'].startswith('http'):
                # Handle URL
                response = requests.get(example['image'])
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    raise ValueError(f"Failed to download image at index {idx}")
            else:
                # Handle file path
                image = Image.open(example['image']).convert("RGB")
        elif isinstance(example['image'], bytes):  # Raw byte stream
            image = Image.open(BytesIO(example['image'])).convert("RGB")
        elif isinstance(example['image'], Image.Image):  # Already a PIL Image object
            image = example['image']
        else:
            raise ValueError(f"Unsupported image format: {type(example['image'])}")

        # Convert image to grayscale
        grayscale_image = image.convert("L")

        # Save original and grayscale images
        image.save(f"coco_original/image_{idx}.jpg")
        grayscale_image.save(f"coco_grayscale/image_{idx}.jpg")

    except Exception as e:
        print(f"Error processing image {idx}: {e}")

# Iterate through dataset and save images
for idx, example in enumerate(dataset):
    save_images(example, idx)
