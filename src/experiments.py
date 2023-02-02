from random import random
from src.get_data import DataSet
from src.fed_core import FedClient, FedServer
from utils import *
import numpy as np
import time
import shutil
import os
from src.utils import check_and_build_dir
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# global settings
clients_num = 5
epoch = 1
dataset = DataSet(clients_num)
x_test, y_test = dataset.get_test_dataset()


def train_one_model(client_id):
    check_and_build_dir("../models/train")
    sub_model_path = f"../models/train/{client_id}.pkl"
    client_dataset = dataset.get_train_batch(client_id - 1)

    model = FedClient()
    global_model_path = "../models/global/global.pkl"
    if os.path.exists(global_model_path):
        model.load_model(global_model_path)
    loss = model.train(client_dataset, epoch)
    acc = 0

    model.save_model(sub_model_path)
    acc = model.evaluate(x_test, y_test)

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
        client_model = FedClient()
        client_model.load_model(path)
        acc = client_model.evaluate(x_test, y_test)
        sub_model_acc.append(acc)
        sub_model_paths.append(path)
    print(f"mean of sub_model_acc: {np.mean(sub_model_acc)}")

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


def save_results(clients_loss_list, clients_acc_list, clients_avg_loss_list, clients_avg_acc_list, global_acc_list):
    clients_num = len(clients_loss_list)
    # 1. save results of each client to a single file
    for client_id in range(clients_num):
        client_dir = f"../results/clients/client_{client_id}"
        check_and_build_dir(client_dir)
        save_to_file(f"{client_dir}/client_{client_id}_loss.txt", clients_loss_list[client_id])
        save_to_file(f"{client_dir}/client_{client_id}_acc.txt", clients_acc_list[client_id])
    # 2. save results of federated model to file
    global_dir = f"../results/global"
    check_and_build_dir(global_dir)
    save_to_file(f"{global_dir}/global_acc.txt", global_acc_list)
    # 3. save results of clients average loss and average accuracy
    clients_avg_dir = "../results/clients_avg"
    check_and_build_dir(clients_avg_dir)
    save_to_file(f"{clients_avg_dir}/clients_avg_loss.txt", clients_avg_loss_list)
    save_to_file(f"{clients_avg_dir}/clients_avg_acc.txt", clients_avg_acc_list)

    print("all results have been saved.")

def save_to_file(file_path, content_list):
    with open(file_path, 'w+') as f:
        for line in content_list:
            f.write(str(line) + '\n')
    f.close()

def save_pics(clients_num):
    # plt config
    plt_config = {
        "title" : "",
        "xlabel" : "federated rounds",
        "ylabel" : "",
    }
    clients_data_dir = "../results/clients"
    global_data_dir = "../results/global"
    clients_avg_data_dir = "../results/clients_avg"

    for client_id in range(clients_num):
        # 1. process loss data
        client_loss_file_path = f"{clients_data_dir}/client_{client_id}/client_{client_id}_loss.txt"
        client_pic_dir = f"../pic/clients/client_{client_id}"
        check_and_build_dir(client_pic_dir)
        client_loss_pic_path = f"{client_pic_dir}/client_{client_id}_loss.png"
        plt_config["title"] = f"loss of client-{client_id}"
        plt_config["ylabel"] = "loss"
        save_to_pic(client_loss_file_path, client_loss_pic_path, plt_config)

        # 2. process acc data
        client_acc_file_path = f"{clients_data_dir}/client_{client_id}/client_{client_id}_acc.txt"
        client_acc_pic_path = f"{client_pic_dir}/client_{client_id}_acc.png"
        plt_config["title"] = f"accuracy of client-{client_id}"
        plt_config["ylabel"] = "acc"
        save_to_pic(client_acc_file_path, client_acc_pic_path, plt_config)

    # 3. process global acc data
    global_acc_file_path = f"{global_data_dir}/global_acc.txt"
    global_pic_dir = f"../pic/global"
    check_and_build_dir(global_pic_dir)
    global_acc_pic_path = f"{global_pic_dir}/global_acc.png"
    plt_config["title"] = f"accuracy of federated model"
    plt_config["ylabel"] = "acc"
    save_to_pic(global_acc_file_path, global_acc_pic_path, plt_config)

    # 4. process clients average loss data
    clients_avg_loss_file_path = f"{clients_avg_data_dir}/clients_avg_loss.txt"
    clients_avg_pic_dir = "../pic/clients_avg"
    check_and_build_dir(clients_avg_pic_dir)
    clients_avg_loss_pic_path = f"{clients_avg_pic_dir}/clients_avg_loss.png"
    plt_config["title"] = f"average loss of client models"
    plt_config["ylabel"] = "loss"
    save_to_pic(clients_avg_loss_file_path, clients_avg_loss_pic_path, plt_config)

    # 5. process clients average acc data
    clients_avg_acc_file_path = f"{clients_avg_data_dir}/clients_avg_acc.txt"
    clients_avg_acc_pic_path = f"{clients_avg_pic_dir}/clients_avg_acc.png"
    plt_config["title"] = f"average accuracy of client models"
    plt_config["ylabel"] = "accuracy"
    save_to_pic(clients_avg_acc_file_path, clients_avg_acc_pic_path, plt_config)

    print("all pictures have been saved.")


def save_to_pic(data_dir, dest_dir, plt_config):
    # 1. read data
    loss_data = pd.read_csv(data_dir, header=None)

    # 2. plt configure
    plt.figure(figsize=(10, 6))
    plt.title(plt_config["title"])
    plt.xlabel(plt_config["xlabel"])
    plt.ylabel(plt_config["ylabel"])
    y_axis_data = loss_data[0].tolist()
    clients_num = len(y_axis_data)
    x_axis_data = [i for i in range(clients_num)]
    plt.plot(x_axis_data, y_axis_data)
    plt.savefig(dest_dir)
    plt.close()

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
