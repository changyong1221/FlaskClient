import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from src.net_core import MnistCNN
import torch.nn.functional as F
from torch import optim
import copy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedClient(nn.Module):
    def __init__(self):
        super(FedClient, self).__init__()
        self.model = MnistCNN().to(dev)
        self.learning_rate = 0.01
        self.clip = 32      # 裁剪系数
        self.q = 0.03
        self.eps = 16.0
        self.delta = 1e-5
        self.tot_T = 100
        self.E = 1
        self.batch_size = 128
        # self.sigma = compute_noise(1, self.q, self.eps, self.E*self.tot_T, self.delta, 1e-5)      # 高斯分布系数
        # self.sigma = 24.831
        self.sigma = 0.9054

    def train(self, client_dataset, epoches):
        loss_func = F.cross_entropy
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.0)

        self.model.train()
        train_loss = 0
        num = 0
        for epoch in range(epoches):
            idx = np.where(np.random.rand(len(client_dataset[:][0])) < self.q)[0]

            sampled_dataset = TensorDataset(client_dataset[idx][0], client_dataset[idx][1])
            train_dl = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            optimizer.zero_grad()

            for data, label in train_dl:
                data, label = data.to(dev), label.to(dev)
                preds = self.model(data.float())

                # loss = loss_func(preds, label.long())
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                # for name, param in self.model.named_parameters():
                #     clipped_grads[name] += param.grad
                # self.model.zero_grad()

                loss = criterion(preds, label.long())
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad / len(idx)
                    self.model.zero_grad()
            # add gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += torch.normal(0, self.sigma*self.clip, clipped_grads[name].shape).to(dev) / len(idx)
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]

            optimizer.step()
        #     if epoch == epoches - 1:
        #         num += 1
        #         train_loss += float(loss.item())
        # return train_loss / num
        return loss.mean().item()

    def evaluate(self, test_data, test_labels):
        self.model.eval()
        correct = 0
        tot_sample = 0
        t_pred_y = self.model(test_data)
        _, predicted = torch.max(t_pred_y, 1)
        correct += (predicted == test_labels).sum().item()
        tot_sample += test_labels.size(0)
        acc = correct / tot_sample
        return acc

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path), strict=True)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


class FedServer(nn.Module):
    def __init__(self):
        super(FedServer, self).__init__()
        self.model = MnistCNN().to(dev)

    def fed_avg(self, model_path_list):
        # FL average
        model_par = [torch.load(model_path) for model_path in model_path_list]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(dev)
        for idx, par in enumerate(model_par):
            # w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            w = 0.1
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                # new_par[name] += par[name] * (w / self.C)
                new_par[name] += par[name] * w
        self.model.load_state_dict(copy.deepcopy(new_par))

    def evaluate(self, test_data, test_labels):
        self.model.eval()
        correct = 0
        tot_sample = 0
        t_pred_y = self.model(test_data)
        _, predicted = torch.max(t_pred_y, 1)
        correct += (predicted == test_labels).sum().item()
        tot_sample += test_labels.size(0)
        acc = correct / tot_sample
        return acc

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    pass