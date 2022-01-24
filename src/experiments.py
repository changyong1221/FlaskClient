from random import random
from src_torch_distributed.get_data import DataSet
from src_torch_distributed.fed_core import FedClient, FedServer
from src_torch_distributed.utils import *
import numpy as np
import time
import shutil

import warnings
warnings.filterwarnings("ignore")

# global settings
clients_num = 10
epoch = 1
dataset = DataSet(clients_num)
x_test, y_test = dataset.get_test_dataset()


def train_one_model(client_id):
    check_and_build_dir("../models/train")
    sub_model_path = f"../models/train/{client_id}.pkl"
    client_dataset = dataset.get_train_batch(client_id)

    model = FedClient()
    global_model_path = "../models/global/global.pkl"
    if os.path.exists(global_model_path):
        model.load_model(global_model_path)
    loss = model.train(client_dataset, epoch)
    # acc = model.evaluate(x_test, y_test, batch_size)
    acc = 0

    model.save_model(sub_model_path)
    # print(f"Client-ID:{client_id}, loss:{loss}, acc:{acc}")
    # print("training done.")
    return loss, acc


def test_one_model():
    sub_model_path = "../models/global/global.pkl"
    server_model = FedServer()
    server_model.load_model(sub_model_path)
    acc = server_model.evaluate(x_test, y_test)
    print(f'model_acc:{acc}')


def test_federated_model():
    sub_model_paths = []
    sub_model_acc = []
    for i in range(clients_num):
        path = f"../models/train/{i}.pkl"
        # client_model = FedClient(net=Mnist_2NN(), ID=client_id)
        # client_model.load_model(path, weight=True)
        # acc = client_model.evaluate(x_test, y_test, batch_size)
        # sub_model_acc.append(acc)
        sub_model_paths.append(path)
    # print(f"mean of sub_model_acc: {np.mean(sub_model_acc)}")

    global_model = FedServer()
    global_model.fed_avg(sub_model_paths)

    global_model_dir = "../models/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"{global_model_dir}/global.pkl"
    global_model.save_model(global_model_path)

    acc = global_model.evaluate(x_test, y_test)
    print(f'clients_num:{clients_num}, global_acc:{acc}')
    return acc


def test_federated():
    # initialization
    start_time = time.time()
    federated_rounds = 100
    init_federated_model()

    # federated main
    clients_loss_list = [[] for x in range(clients_num)]
    clients_acc_list = [[] for x in range(clients_num)]
    global_acc_list = []
    clients_avg_loss_list = []
    clients_avg_acc_list = []
    for epoch in range(federated_rounds):
        print(f"Round {epoch + 1}:")
        clients_loss_sum = 0
        clients_acc_sum = 0
        for client_id in range(clients_num):
            client_loss, client_acc = train_one_model(client_id)
            clients_loss_list[client_id].append(round(client_loss, 4))
            clients_acc_list[client_id].append(round(client_acc, 4))
            clients_loss_sum += client_loss
            clients_acc_sum += client_acc
        clients_avg_loss_list.append(clients_loss_sum / clients_num)
        clients_avg_acc_list.append(clients_acc_sum / clients_num)
        global_acc = test_federated_model()
        global_acc_list.append(round(global_acc, 4))
    save_results(clients_loss_list, clients_acc_list, clients_avg_loss_list, clients_avg_acc_list, global_acc_list)
    save_pics(clients_num)

    end_time = time.time()
    print("Time used: %.2f s" % (end_time - start_time))


def init_federated_model():
    source_path = "../initial_model/global_model.pkl"
    check_and_build_dir("../models/global")
    dest_path = "../models/global/global.pkl"
    shutil.copyfile(source_path, dest_path)

    # init_model = FedServer()
    # init_model_dir = "../initial_model"
    # check_and_build_dir(init_model_dir)
    # init_model_path = f"{init_model_dir}/global_model.pkl"
    # init_model.save_model(init_model_path)


if __name__ == '__main__':
    test_federated()
    # init_federated_model()
    # train_one_model(1)
    # test_one_model()
    # test_federated_model(10)

    # batch training
    # for i in range(clients_num):
    #     train_one_model(i + 1)

    # batch federated
    # for i in range(clients_num):
    #     test_federated_model(i + 1)

    # train one model 1000 rounds
    # train_one_model_roundly(1, 1000)
