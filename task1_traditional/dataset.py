import os
import torch
import numpy as np
import random

def load_data(data_root, split='train', num_classes=10, selected_classes=None):

    split_dir = os.path.join(data_root, split)
    all_classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    if selected_classes is None:
        random.seed(42)
        k = min(num_classes, len(all_classes))
        selected_classes = sorted(random.sample(all_classes, k))
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}
    
    features = []
    labels = []
    
    print(f"Loading {split} data for {len(selected_classes)} classes...")
    
    for cls_name in selected_classes:
        cls_dir = os.path.join(split_dir, cls_name)
        cls_idx = class_to_idx[cls_name]
        
        for fname in os.listdir(cls_dir):
            if fname.endswith('.pt'):
                fpath = os.path.join(cls_dir, fname)
                feat = torch.load(fpath)
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy().flatten()
                features.append(feat)
                labels.append(cls_idx)
                    
    X = np.array(features)
    y = np.array(labels)
    
    return X, y, selected_classes
