import os
from PIL import Image
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):

        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        self.classes = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.split_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label
