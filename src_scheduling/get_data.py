from torchvision import datasets, transforms
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DataSet(object):
    def __init__(self, client_id):
        # whole data
        self.test_data = None
        self.test_label = None

        # 加载mnist数据集
        data_train = datasets.MNIST(root="datasets/", download=False, train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        data_test = datasets.MNIST(root="datasets/", download=False, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        # split MNIST (training set) into non-iid data sets
        self.client_non_iid = []
        user_dict = mnist_noniid(data_train, clients_num)
        for i in range(clients_num):
            idx = user_dict[i]
            d = data_train.data[idx].float().unsqueeze(1)
            targets = data_train.targets[idx].float()
            self.client_non_iid.append((d, targets))
        self.test_data = torch.tensor(data_test.data.float().unsqueeze(1)).to(dev)
        self.test_label = torch.tensor(data_test.targets.float()).to(dev)

    def get_test_dataset(self):
        return self.test_data, self.test_label

    def get_train_batch(self, client_id):
        return TensorDataset(torch.tensor(self.client_non_iid[client_id][0]),
                             torch.tensor(self.client_non_iid[client_id][1]))


if __name__ == "__main__":
    client_dataset = DataSet(10)
    # test_data, test_labels = client_dataset.get_test_dataset()
    train_data, train_labels = client_dataset.get_train_batch(1)
    print("type(train_data): ", type(train_data))
    print("type(train_labels): ", type(train_labels))
    print("len(train_data): ", len(train_data))
    print("len(train_labels): ", len(train_labels))
