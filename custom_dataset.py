from torch.utils.data import Dataset
import os
from torchvision.io import read_image

"""
A custom dataset for the Watermark Removal model. 
"""

class WatermarkRemovalData(Dataset):
    def __init__(self, watermark_dirs, no_watermark_dirs, transform=None, target_transform=None):
        self.watermark_files = watermark_dirs
        self.no_watermark_files = no_watermark_dirs
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.no_watermark_files)

    def __getitem__(self, idx):
        watermarked_image_path = self.watermark_files[idx]
        no_watermark_image_path = self.no_watermark_files[idx]
        
        watermarked_image = read_image(watermarked_image_path)
        no_watermark_image = read_image(no_watermark_image_path)

        if self.transform:
            watermarked_image = self.transform(watermarked_image)
        if self.target_transform:
            no_watermark_image = self.target_transform(no_watermark_image)
        return watermarked_image, no_watermark_image