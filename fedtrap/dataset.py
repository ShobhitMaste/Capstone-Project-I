import os
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class InsectDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}. Error: {e}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def load_split_metadata(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def create_dataloaders(split_metadata_path: str, batch_size: int = 32, img_size: int = 224, num_workers: int = 2) -> Dict[str, DataLoader]:
    metadata = load_split_metadata(split_metadata_path)
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        if split in metadata:
            paths = metadata[split]['images']
            labels = metadata[split]['labels']
            transform = get_transforms(train=(split == 'train'), img_size=img_size)
            dataset = InsectDataset(paths, labels, transform=transform)
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            
    return dataloaders
