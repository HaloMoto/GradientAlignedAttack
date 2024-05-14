from torchvision import transforms
import numpy as np
import os

class SelectedImagenet():
    def __init__(self, data_dir="../data/"):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.data = np.load(os.path.join(data_dir, "data.npy"))
        self.labels = np.load(os.path.join(data_dir, "labels.npy")).tolist()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = self.data[item]
        target = self.labels[item]

        img = self.transform(img)

        return img, target