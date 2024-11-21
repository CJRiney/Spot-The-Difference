from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class SNNDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, idx):
        input_image = Image.open(self.image_paths[idx]['input']).convert('RGB')
        target_image = Image.open(self.image_paths[idx]['target']).convert('RGB')
        negative_images = [Image.open(p).convert('RGB') for p in self.image_paths[idx]['negatives']]

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
            negative_images = [self.transform(img) for img in negative_images]

        return {
            'input_image': input_image,
            'target_image': target_image,
            'negative_images': torch.stack(negative_images) if negative_images else None
        }

    def __len__(self):
        return len(self.image_paths)
