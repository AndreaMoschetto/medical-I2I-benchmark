from torch.utils.data import Dataset
import os
import random
import torch
from PIL import Image


class UnifiedBrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", seed=42):
        assert split in ["train", "val",
                         "test"], "split must be 'train', 'val' or 'test'"
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.seed = seed
        self.samples = self._create_file_pairs()
        self._split_dataset()

    def _create_file_pairs(self):
        t1_dir = os.path.join(self.root_dir, "t1")
        t2_dir = os.path.join(self.root_dir, "t2")

        t1_files = set(os.listdir(t1_dir))
        t2_files = set(os.listdir(t2_dir))
        common_files = list(t1_files.intersection(t2_files))
        common_files.sort()

        pairs = [(os.path.join(t1_dir, fname), os.path.join(t2_dir, fname))
                 for fname in common_files]
        return pairs

    def _split_dataset(self):
        random.seed(self.seed)
        random.shuffle(self.samples)

        n_total = len(self.samples)
        n_train = int(n_total * 0.80)
        n_val = int(n_total * 0.05)

        if self.split == "train":
            self.samples = self.samples[:n_train]
        elif self.split == "val":
            self.samples = self.samples[n_train:n_train + n_val]
        elif self.split == "test":
            self.samples = self.samples[n_train + n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t1_path, t2_path = self.samples[idx]
        t1_image = Image.open(t1_path).convert("L")
        t2_image = Image.open(t2_path).convert("L")

        if self.transform:
            t1_image = self.transform(t1_image)
            t2_image = self.transform(t2_image)

        return {
            "t1": t1_image,
            "t2": t2_image,
            "filename": os.path.basename(t1_path)
        }


class PredictionDataset(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.files = sorted([
            f for f in os.listdir(directory) if f.endswith('.pt')
        ])
        if not self.files:
            raise ValueError(f"No .pt files found in directory: {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        data = torch.load(file_path)
        # expected shape: [1, H, W] or [C, H, W]
        pred = data["prediction"]
        gt = data["target"]
        return pred, gt
