import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    def __init__(self):
        self.data = []
        self.data_root = "data/training/"
        with open(f'{self.data_root}prompts.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread(self.data_root + source_filename)
        target = cv2.imread(self.data_root + target_filename)
        source = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_CUBIC)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

