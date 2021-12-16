import os
import time
import shutil
import src.globals as glo
from src.log import print_log
from src.net_core import Mnist_2NN
from src.fed_core import FedClient, FedServer
from src.utils import save_results
import torch.nn.functional as F
from torch import optim


def initialize_global_model():
    global_model_path = glo.get_global_var("global_model_path")
    initial_model_path = "initial_model/global_model.pkl"
    shutil.copyfile(initial_model_path, global_model_path)


def preheat_for_first_round(dataset):
    client_id = glo.get_global_var("client_id")

    # training settings
    epoch = 10
    batch_size = 64
    learning_rate = 0.01

    x_test, y_test = dataset.get_test_dataset()
    x_train, y_train = dataset.get_train_batch(client_id, batch_size * 10)

    model = FedClient(net=Mnist_2NN(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    model.set_model_settings(loss_func=F.cross_entropy, optimizer=optim.SGD(model.net.parameters(), lr=learning_rate))
    loss = model.train(x_train, y_train, epoch, batch_size)
    acc = model.evaluate(x_test, y_test, batch_size)
    print_log("preheat finished.")


def train_one_model(dataset):
    startTime = time.time()

    # training settings
    epoch = 10
    batch_size = 64
    learning_rate = 0.01

    global_model_path = glo.get_global_var("global_model_path")
    sub_model_path = glo.get_global_var("sub_model_path")
    client_id = glo.get_global_var("client_id")

    x_test, y_test = dataset.get_test_dataset()
    x_train, y_train = dataset.get_train_batch(client_id, batch_size * 10)

    model = FedClient(net=Mnist_2NN(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    model.set_model_settings(loss_func=F.cross_entropy, optimizer=optim.SGD(model.net.parameters(), lr=learning_rate))
    if os.path.exists(global_model_path):
        model.load_model(global_model_path, weight=True)
    loss = model.train(x_train, y_train, epoch, batch_size)
    acc = model.evaluate(x_test, y_test, batch_size)

    model.save_model(sub_model_path, weight=True)
    model.upload()

    save_results(round(loss, 4), "LOSS", is_global=False)
    save_results(round(acc, 4), "ACC", is_global=False)
    print_log(f"Client-ID:{client_id} , loss:{loss} , acc:{acc} , Time:{time.time() - startTime}")
    print_log("training done.")


def has_submodel():
    sub_model_path = glo.get_global_var("sub_model_path")
    if os.path.exists(sub_model_path):
        return True
    else:
        return False


# run test dataset for a list of submodels and return test scores
def submodels_test():
    pass


def merge_models_and_test(dataset):
    client_id = glo.get_global_var("client_id")
    merge_clients_num = glo.get_global_var("merge_clients_num")
    merge_clients_id_list = glo.get_global_var("merge_clients_id_list")
    global_model_path = glo.get_global_var("global_model_path")

    x_test, y_test = dataset.get_test_dataset()
    models_path_list = []
    for merge_idx in merge_clients_id_list:
        print_log(f'models/downloads/client-{client_id}/{merge_idx}.pkl')
        models_path_list.append(f'models/downloads/client-{client_id}/{merge_idx}.pkl')

    # get test scores of submodels
    client_acc_list = []
    client_score_list = []
    for i in range(merge_clients_num):
        client_model = FedClient(net=Mnist_2NN(), ID=client_id)
        client_model.load_model(models_path_list[i], weight=True)
        acc = client_model.evaluate(x_test, y_test, batch_size=64)
        client_acc_list.append(acc)
        client_score_list.append(int(acc*1000))
        print_log(f'client({merge_clients_id_list[i]})_model_acc: {acc}')

    # merge global model and test
    global_model = FedServer(net=Mnist_2NN())
    print_log("start loading global model...")
    global_model.load_client_weights(models_path_list)
    print_log("global model loaded.")
    print_log("start doing weights average...")
    global_model.fed_avg()
    print_log("weights average done.")
    print_log("start evaluating global model...")
    global_acc = global_model.evaluate(x_test, y_test, batch_size=64)
    print_log(f'global_model_acc: {global_acc}')

    global_model.save_model(global_model_path, weight=True)
    global_model_score = global_acc * 1000
    save_results(round(global_acc, 4), "ACC", is_global=True)
    retSet = {"clients_scores": client_score_list, "global_score": int(global_model_score)}
    print_log(retSet)
    return retSet


# update local model to the newest global model
def update_model(dataset):
    global_model_path = glo.get_global_var("global_model_path")
    sub_model_path = glo.get_global_var("sub_model_path")
    shutil.copyfile(global_model_path, sub_model_path)
    # client_id = glo.get_global_var("client_id")
    # x_test, y_test = dataset.get_test_dataset()
    #
    # # get test score of sub_model
    # client_model = FedClient(net=Mnist_2NN(), ID=client_id)
    # client_model.load_model(sub_model_path, weight=True)
    # client_acc = client_model.evaluate(x_test, y_test, batch_size=64)
    # sub_model_score = client_acc * 1000
    #
    # # get test score of global_model
    # global_model = FedServer(net=Mnist_2NN())
    # global_model.load_model(global_model_path, weight=True)
    # global_acc = global_model.evaluate(x_test, y_test, batch_size=64)
    # global_model_score = global_acc * 1000
    #
    # print_log(f"sub_model_score: {sub_model_score}")
    # print_log(f"global_model_score: {global_model_score}")
    # if global_model_score > sub_model_score:
    #     print_log("global model is better.")
    #     print_log("updating local model...")
    #     if os.path.exists(global_model_path):
    #         shutil.copyfile(global_model_path, sub_model_path)
    #     print_log("local model updated.")
    # else:
    #     print_log("local model is better.")
    #     os.remove(global_model_path)
    #     print_log("global model dropped.")
    print_log("update process finished.")


if __name__ == '__main__':
    update_model()
