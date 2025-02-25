import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os

class ColorizationDataset(Dataset):
    def __init__(self, data_root="data/colorization", image_size=1024):
        self.data = []
        self.data_root = data_root
        self.image_size = image_size
        
        # Check and print the full path
        prompts_path = os.path.join(self.data_root, 'prompts.json')
        print(f"Looking for prompts.json at: {os.path.abspath(prompts_path)}")
        
        # Read each line and parse JSON
        with open(prompts_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error details: {e}")
                        continue
        
        print(f"Found {len(self.data)} samples in the dataset")
        
        # Transformations for grayscale (source) and color (target) images
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Full paths for source and target images
        source_path = os.path.join(self.data_root, item['source'])
        target_path = os.path.join(self.data_root, item['target'])
        
        # Check if images exist
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source image not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target image not found: {target_path}")
        
        # Load grayscale image
        source = Image.open(source_path).convert('RGB')
        # Load color image
        target = Image.open(target_path).convert('RGB')
        
        # Apply transformations
        source_tensor = self.transform(source)
        target_tensor = self.transform(target)
        
        # Normalize
        source_tensor = self.normalize(source_tensor)
        target_tensor = self.normalize(target_tensor)
        
        return {
            "jpg": target_tensor,
            "txt": item['prompt'],
            "hint": source_tensor
        }
        
    def get_sample_path(self, idx):
        """Helper function to get image paths for visualization"""
        item = self.data[idx]
        return {
            'source': os.path.join(self.data_root, item['source']),
            'target': os.path.join(self.data_root, item['target']),
            'prompt': item['prompt']
        }
