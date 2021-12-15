from get_data import DataSet
from net_core import Mnist_2NN
from fed_core import FedClient, FedServer
# from utils import *
import torch.nn.functional as F
from torch import optim
import time

# global settings
clients_num = 10
client_id = 1
epoch = 10
batch_size = 64
learning_rate = 0.01
# dataset = DataSet(clients_num, False)
# x_test, y_test = dataset.get_test_dataset()


def initialize_model():
    save_dir = "../initialization_model/global_model.pkl"
    model = FedClient(net=Mnist_2NN(), ID=0)
    model.set_model_settings(loss_func=F.cross_entropy, optimizer=optim.SGD(model.net.parameters(), lr=learning_rate))
    model.save_model(save_dir, weight=True)

    
def train_one_model(client_id):
    check_and_build_dir("../models/train")
    sub_model_path = f"../models/train/{client_id}.pkl"
    x_train, y_train = dataset.get_train_batch(client_id, batch_size*10)

    model = FedClient(net=Mnist_2NN(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    model.set_model_settings(loss_func=F.cross_entropy, optimizer=optim.SGD(model.net.parameters(), lr=learning_rate))
    global_model_path = "../models/global/global.pkl"
    if os.path.exists(global_model_path):
        model.load_model(global_model_path, weight=True)
    loss = model.train(x_train, y_train, epoch, batch_size)
    acc = model.evaluate(x_test, y_test, batch_size)

    model.save_model(sub_model_path, weight=True)
    model.upload()
    print(f"Client-ID:{client_id}, loss:{loss}, acc:{acc}")
    print("training done.")
    return loss, acc


def test_one_model():
    sub_model_path = "../models/global/global.pkl"
    client_model = FedClient(net=Mnist_2NN(), ID=client_id)
    client_model.load_model(sub_model_path, weight=True)
    acc = client_model.evaluate(x_test, y_test, batch_size)
    print(f'client({client_id})_acc:{acc}')


def test_federated_model():
    sub_model_paths = []
    sub_model_acc = []
    for i in range(clients_num):
        path = f"../models/train/{i + 1}.pkl"
        # client_model = FedClient(net=Mnist_2NN(), ID=client_id)
        # client_model.load_model(path, weight=True)
        # acc = client_model.evaluate(x_test, y_test, batch_size)
        # sub_model_acc.append(acc)
        sub_model_paths.append(path)
    # print(f"mean of sub_model_acc: {np.mean(sub_model_acc)}")

    global_model = FedServer(net=Mnist_2NN())
    global_model.load_client_weights(sub_model_paths)
    global_model.fed_avg()
    acc = global_model.evaluate(x_test, y_test, batch_size)
    print(f'clients_num:{clients_num}, global_acc:{acc}')

    global_model_dir = "../models/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"{global_model_dir}/global.pkl"
    global_model.save_model(global_model_path, weight=True)
    return acc


def test_federated():
    # initialization
    start_time = time.time()
    federated_rounds = 1000
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
            client_loss, client_acc = train_one_model(client_id + 1)
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
    global_model = FedServer(net=Mnist_2NN())
    global_model_dir = "../models/global"
    check_and_build_dir(global_model_dir)
    global_model_path = f"{global_model_dir}/global.pkl"
    global_model.save_model(global_model_path, weight=True)


def train_one_model_roundly(client_id, rounds):
    check_and_build_dir("../models/train")
    sub_model_path = f"../models/train/{client_id}.pkl"
    client_loss_list = []
    client_acc_list = []

    for iter in range(rounds):
        x_train, y_train = dataset.get_train_batch(client_id, batch_size*10)

        model = FedClient(net=Mnist_2NN(), ID=client_id)
        model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
        model.set_model_settings(loss_func=F.cross_entropy, optimizer=optim.SGD(model.net.parameters(), lr=learning_rate))
        if os.path.exists(sub_model_path):
            model.load_model(sub_model_path, weight=True)
        loss = model.train(x_train, y_train, epoch, batch_size)
        acc = model.evaluate(x_test, y_test, batch_size)

        model.save_model(sub_model_path, weight=True)
        model.upload()
        client_loss_list.append(round(loss, 4))
        client_acc_list.append(round(acc, 4))
        print(f"Round {iter}: loss:{loss}, acc:{acc}")
    print("training done.")

    # save results
    client_dir = f"../results/clients/client_{client_id}"
    check_and_build_dir(client_dir)
    save_to_file(f"{client_dir}/client_{client_id}_loss.txt", client_loss_list)
    save_to_file(f"{client_dir}/client_{client_id}_acc.txt", client_acc_list)
    print("all results have been saved.")

    # save pictures
    plt_config = {
        "title" : "",
        "xlabel" : "federated rounds",
        "ylabel" : "",
    }
    # 1. process loss data
    client_loss_file_path = f"{client_dir}/client_{client_id}_loss.txt"
    client_pic_dir = f"../pic/clients/client_{client_id}"
    check_and_build_dir(client_pic_dir)
    client_loss_pic_path = f"{client_pic_dir}/client_{client_id}_loss.png"
    plt_config["title"] = f"loss of client-{client_id}"
    plt_config["ylabel"] = "loss"
    save_to_pic(client_loss_file_path, client_loss_pic_path, plt_config)

    # 2. process acc data
    client_acc_file_path = f"{client_dir}/client_{client_id}_acc.txt"
    client_acc_pic_path = f"{client_pic_dir}/client_{client_id}_acc.png"
    plt_config["title"] = f"accuracy of client-{client_id}"
    plt_config["ylabel"] = "acc"
    save_to_pic(client_acc_file_path, client_acc_pic_path, plt_config)
    print("all pictures have been saved.")


if __name__ == '__main__':
    initialize_model()
    # test_federated()
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
    # train_one_model_roundly(1, 10)


