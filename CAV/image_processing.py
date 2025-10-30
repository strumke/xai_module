import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image

random.seed(1337)

def select_concept_images(folder_path, concept, n):
    """
    Returns a list with up to `n` random .jpg fiel paths from `folder_path` that start with `concept`.
    """
    matching_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if (f.lower().startswith(concept.lower()) and f.lower().endswith('.jpg'))
    ]
    
    # Shuffle and select
    if len(matching_files) < n:
        print(f"Only found {len(matching_files)} files matching concept '{concept}'. Returning all.")
        return matching_files
    else:
        return random.sample(matching_files, n)

def sample_images(folder_path, n):
    """
    Returns a list with up to `n` random files from `folder_path`.
    """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]

    # Shuffle and select
    if len(files) < n:
        print(f"Only found {len(files)} files in folder. Returning all.")
        return files
    else:
        return random.sample(files, n)

class ResNetImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image 

def get_resnet_dataloader(image_paths, transform, batch_size=16):
    dataset = ResNetImageDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


