from torchvision import datasets, transforms
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users * 2)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    np.random.seed(2022)
    # rand_set_list = [{0, 19}, {1, 7}, {11, 4}, {16, 17}, {9, 2}, {13, 14}, {8, 3}, {10, 5}, {18, 6}, {12, 15}]
    # rand_list = np.random.permutation(num_shards)
    # print(rand_list)
    # exit()

    for i in range(num_users):
        # rand_set = rand_set_list[i]
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # idx_shard = list(set(idx_shard) - rand_set)
        # print("i: ", i)
        # print("rand_set: ", rand_set)
        # print("idx_shard: ", idx_shard)
        # print(rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


class DataSet(object):
    def __init__(self, clients_num):
        # whole data
        self.test_data = None
        self.test_label = None
        self.clients_num = clients_num

        # 加载mnist数据集
        data_train = datasets.MNIST(root="../datasets/", download=False, train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        data_test = datasets.MNIST(root="../datasets/", download=False, train=False, transform=transforms.Compose([
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
    client_dataset = DataSet(5)
    # test_data, test_labels = client_dataset.get_test_dataset()
    train_data, train_labels = client_dataset.get_train_batch(1)
    print("type(train_data): ", type(train_data))
    print("type(train_labels): ", type(train_labels))
    print("len(train_data): ", len(train_data))
    print("len(train_labels): ", len(train_labels))
