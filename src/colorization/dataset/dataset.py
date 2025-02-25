import json
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from torch.utils.data import Dataset
from PIL import Image


class ColorizationDataset(Dataset):
    def __init__(self, data_root="data/colorization/training/", image_size=1024):
        self.data = []
        self.data_root = data_root
        self.image_size = image_size 

        with open(f'{self.data_root}prompts.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform_source = T.Compose([
            T.Resize((image_size, image_size)), 
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])   # Normalize to [-1, 1]
        ])

        self.transform_target = T.Compose([
            T.Resize((image_size, image_size)),  
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # Load color image (RGB)
        target = Image.open(self.data_root + target_filename).convert("RGB")

        # Convert color image to grayscale (3-channel RGB)
        source = target.convert("L").convert("RGB")

        # Apply transformations
        source_tensor = self.transform_source(source)  # [3, 1024, 1024]
        target_tensor = self.transform_target(target)  # [3, 1024, 1024]

        return {
            "jpg": target_tensor, # Color image (Ground Truth) 
            "txt": prompt, # Description prompt
            "hint": source_tensor  # Grayscale image
        }
