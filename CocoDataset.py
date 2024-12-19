from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
import os
import numpy as np

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, oversample='N'):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.oversample = oversample
        self.data = []  # To store processed (img_path, label) pairs

        # Preprocess and load data to handle oversampling
        for img_id in self.ids:
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            if not img_path.endswith(('.jpg', '.png', '.jpeg')):
                continue  # Skip non-image files

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Determine label
            label = 0
            for ann in anns:
                category_id = ann['category_id']
                if category_id != 0:  # Non-zero category ID indicates defect
                    label = 1
                    break

            self.data.append((img_path, label))

        # Perform oversampling if required
        if self.oversample == 'Y':
            self._perform_oversampling()

    def _perform_oversampling(self):
        # Extract image paths and labels separately
        img_paths, labels = zip(*self.data)

        # Convert to numpy arrays for compatibility with RandomOverSampler
        img_paths = np.array(img_paths).reshape(-1, 1)
        labels = np.array(labels)

        # Apply RandomOverSampler
        ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
        img_paths_resampled, labels_resampled = ros.fit_resample(img_paths, labels)

        # Reconstruct the data list with oversampled image paths and labels
        self.data = [(img_path[0], label) for img_path, label in zip(img_paths_resampled, labels_resampled)]

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)