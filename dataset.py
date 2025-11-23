"""Simple face dataset loader for folder-based structure."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2


class SimpleFaceDataset(Dataset):
    """Simple dataset: each folder = person_id, contains images."""
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        
        # Scan all subdirectories
        for person_dir in sorted(self.root_dir.iterdir()):
            if person_dir.is_dir():
                person_id = person_dir.name
                # Get all images in folder
                for img_path in sorted(person_dir.glob('*.jpg')) + sorted(person_dir.glob('*.png')):
                    self.samples.append((str(img_path), person_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, person_id = self.samples[idx]
        
        # Load image with cv2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img_tensor, person_id


def get_dataloader(dataset, batch_size=16, shuffle=True):
    """Create DataLoader for dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

