from torchvision import transforms
import numpy as np
import os

class SelectedImagenet2Attack():
    def __init__(self, predict_status_arr, data_dir="../data/"):
        # 都分类正确的样本
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.data = np.load(os.path.join(data_dir, "data.npy"))[predict_status_arr]
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))[predict_status_arr].tolist()

        # 选择前1000个样本进行评估
        self.data = self.data[:1000]
        self.labels = self.labels[:1000]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = self.data[item]
        target = self.labels[item]

        img = self.transform(img)

        return img, target

class SelectedImagenet2Attack_INS_R1():
    def __init__(self, predict_status_arr, batch_id=1, data_dir="../data/"):
        # 都分类正确的样本
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.data = np.load(os.path.join(data_dir, "data.npy"))[predict_status_arr]
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))[predict_status_arr].tolist()

        # 选择前1000个样本进行评估
        self.data = self.data[(batch_id-1)*700:batch_id*700]
        self.labels = self.labels[(batch_id-1)*700:batch_id*700]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img = self.data[item]
        target = self.labels[item]

        img = self.transform(img)

        return img, target
