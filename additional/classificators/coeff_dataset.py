from torch.utils.data import Dataset
import os
import torch
import numpy as np

class FeatureDataset(Dataset):
    def __init__(self, coeff_root, max_root):

        classes_path = os.path.join(coeff_root, 'classes.pt')
        self.classes = torch.load(classes_path)

        labels_path = os.path.join(max_root, "..", 'labels')

        num_batches = len([f for f in os.listdir(max_root) if f.endswith('.pt')])

        # First pass: determine total size and shape
        first_feat = torch.load(os.path.join(max_root, '0.pt')).cpu()
        first_label = torch.load(os.path.join(labels_path, '0.pt'))

        images_per_batch = first_feat.shape[0]
        last_batch_path = os.path.join(max_root, f'{num_batches - 1}.pt')
        images_in_last_batch = torch.load(last_batch_path).shape[0]
        total_samples = (num_batches - 1) * images_per_batch + images_in_last_batch

        # Pre-allocate numpy arrays
        feat_dtype = first_feat.dtype
        label_dtype = first_label.dtype
        self.features = torch.empty((total_samples, *first_feat.shape[1:]), dtype=feat_dtype)
        self.labels = torch.empty((total_samples, *first_label.shape[1:]), dtype=label_dtype)

        # Second pass: load directly into pre-allocated arrays
        offset = 0
        for i in range(num_batches):
            feat = torch.load(os.path.join(max_root, f'{i}.pt')).cpu()
            label = torch.load(os.path.join(labels_path, f'{i}.pt'))
            n = feat.shape[0]
            self.features[offset:offset + n] = feat
            self.labels[offset:offset + n] = label
            offset += n

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]