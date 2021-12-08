import os
import time
import shutil
import src.globals as glo
from src.log import printLog
from src.net_core import Mnist_2NN
from src.fed_core import FedClient, FedServer
import torch.nn.functional as F
from torch import optim


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

    printLog(f"Client-ID:{client_id} , loss:{loss} , acc:{acc} , Time:{time.time() - startTime}")
    printLog("training done.")


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
        printLog(f'models/downloads/client-{client_id}/{merge_idx}.pkl')
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
        printLog(f'client({merge_clients_id_list[i]})_model_acc: {acc}')

    # merge global model and test
    global_model = FedServer(net=Mnist_2NN())
    printLog("start loading global model...")
    global_model.load_client_weights(models_path_list)
    printLog("global model loaded.")
    printLog("start doing weights average...")
    global_model.fed_avg()
    printLog("weights average done.")
    printLog("start evaluating global model...")
    global_acc = global_model.evaluate(x_test, y_test, batch_size=64)
    printLog(f'global_model_acc: {global_acc}')

    global_model.save_model(global_model_path, weight=True)
    global_model_score = global_acc * 1000
    retSet = {"clients_scores": client_score_list, "global_score": int(global_model_score)}
    printLog(retSet)
    return retSet


# update local model to the newest global model
def update_model(dataset):
    global_model_path = glo.get_global_var("global_model_path")
    sub_model_path = glo.get_global_var("sub_model_path")
    client_id = glo.get_global_var("client_id")
    x_test, y_test = dataset.get_test_dataset()

    # get test score of sub_model
    client_model = FedClient(net=Mnist_2NN(), ID=client_id)
    client_model.load_model(sub_model_path, weight=True)
    client_acc = client_model.evaluate(x_test, y_test, batch_size=64)
    sub_model_score = client_acc * 1000

    # get test score of global_model
    global_model = FedServer(net=Mnist_2NN())
    global_model.load_model(global_model_path, weight=True)
    global_acc = global_model.evaluate(x_test, y_test, batch_size=64)
    global_model_score = global_acc * 1000

    printLog(f"sub_model_score: {sub_model_score}")
    printLog(f"global_model_score: {global_model_score}")
    if global_model_score > sub_model_score:
        printLog("global model is better.")
        printLog("updating local model...")
        if os.path.exists(global_model_path):
            shutil.copyfile(global_model_path, sub_model_path)
        printLog("local model updated.")
    else:
        printLog("local model is better.")
        os.remove(global_model_path)
        printLog("global model dropped.")
    printLog("update process finished.")


if __name__ == '__main__':
    update_model()
