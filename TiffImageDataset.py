from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
from random import random

class TiffImageDataset(Dataset):
    def __init__(self, imagePaths, masksPaths):
        self.imagePaths = imagePaths
        self.masksPaths = masksPaths

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        image = Image.open(self.imagePaths[index])
        image = ToTensor()(image)

        mask = Image.open(self.masksPaths[index])
        mask = ToTensor()(mask)[:1, :, :]

        if random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        if random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return (image, mask)
