import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WindowDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.image_size = image_size

        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        mask = (mask > 0.5).float()
        return image, mask
