
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def get_image_and_label_paths(src_dir):
    """Extract image and corresponding label file paths from the source directory."""
    image_paths = []
    label_paths = []

    subjects = os.listdir(src_dir)

    for subject in subjects:
        image_dir = os.path.join(src_dir, subject, "image")
        label_dir = os.path.join(src_dir, subject, "mask")

        slices = os.listdir(image_dir)

        for slice_file in slices:
            image_path = os.path.join(image_dir, slice_file)
            label_path = os.path.join(label_dir, slice_file.replace(".npy", "_gt.npy"))

            image_paths.append(image_path)
            label_paths.append(label_path)

    return image_paths, label_paths


class ImgSegDataset(Dataset):
    def __init__(self, image_paths, label_paths, aug_seq=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.aug_seq = aug_seq

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load the image and label
        image = np.load(image_path)
        label = np.load(label_path)

        # Apply augmentations, if augmentation sequence is provided
        if self.aug_seq:
            image, label = self.augment(image, label)

        # Expand dimension 0 to add a channel dimension and convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label

    def augment(self, image, label):
        # Set a random seed to ensure different augmentations across data loader workers
        # https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
        random_seed = torch.randint(0, 1000000, (1,)).item()
        imgaug.seed(random_seed)

        # Create a segmentation map for the label
        segmap = SegmentationMapsOnImage(label, shape=image.shape)

        # Apply the augmentations
        image_aug, label_aug = self.aug_seq(image=image, segmentation_maps=segmap)

        # Convert the segmentation map back to an array
        label_aug = label_aug.get_arr()

        return image_aug, label_aug
