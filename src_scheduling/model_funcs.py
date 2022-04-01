import os
import time
import shutil
import src_scheduling.globals as glo
from src_scheduling.log import print_log
from src_scheduling.fed_core import FedClient, FedServer
from src_scheduling.utils import save_results

import torch
import math
from core.domain import create_domains, create_multi_domain
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from analyzer.results_analyzer import compute_avg_task_process_time_by_name
from utils.load_data import load_machines_from_file, load_task_batches_from_file, sample_tasks_from_file, \
    load_tasks_from_file, sample_task_batches_from_file
from utils.file_check import check_and_build_dir
from utils.state_representation import get_machine_kind_list


def initialize_global_model():
    machine_num = 20
    scheduler = DQNScheduler(multidomain_id=1, machine_num=machine_num, task_batch_num=1,
                             machine_kind_num_list=[], machine_kind_idx_range_list=[],
                             )
    global_model_path = glo.get_global_var("global_model_path")
    scheduler.DRL.save_initial_model(global_model_path)


def preheat_for_first_round(dataset):
    client_id = glo.get_global_var("client_id")

    # training settings
    epoch = 1

    x_test, y_test = dataset.get_test_dataset()
    client_dataset = dataset.get_train_batch(client_id - 1)

    model = FedClient()
    loss = model.train(client_dataset, epoch)
    acc = model.evaluate(x_test, y_test)
    print_log("preheat finished.")


def train_one_model():
    # Header
    startTime = time.time()

    # training settings
    n_batches = 5

    client_id = glo.get_global_var("client_id")
    glo.set_global_var("current_round", glo.get_global_var("current_round") + 1)

    # training stage
    model = FedClient()
    print_log("client model training...")
    model.train(n_batches)
    print_log("client model training finished.")

    # Footer
    print_log(f"Client-ID:{client_id}, Time:{time.time() - startTime}")
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


def merge_models_and_test():
    client_id = glo.get_global_var("client_id")
    merge_clients_num = glo.get_global_var("merge_clients_num")
    merge_clients_id_list = glo.get_global_var("merge_clients_id_list")
    global_model_path = glo.get_global_var("global_model_path")
    glo.set_global_var("is_federated_test", True)

    round_list = glo.get_global_var("clients_merge_rounds_list")
    models_path_list = []
    for merge_idx in merge_clients_id_list:
        print_log(f'models/downloads/client-{client_id}/{merge_idx}.pkl')
        models_path_list.append(f'models/downloads/client-{client_id}/{merge_idx}.pkl')
        round_list[merge_idx - 1] += 1


    # models_path_list = [f"models/downloads/client-{client_id}/1.pkl",
    #                    f"models/downloads/client-{client_id}/2.pkl"]
    # set merge rounds list
    glo.set_global_var("clients_merge_rounds_list", round_list)

    # merge global model and test
    global_model = FedServer()
    print_log("global model loaded.")
    print_log("start doing weights average...")
    global_model.fed_avg(models_path_list, merge_clients_id_list)
    print_log("weights average done.")
    print_log("start evaluating global model...")
    global_processing_time = global_model.evaluate()
    # global_processing_time = 90000
    print_log(f'global_processing_time: {global_processing_time}')
    save_results(global_processing_time, "TIME", True, 10000)

    # test only one client model
    client_score_list = []
    client_model = FedClient()
    client_processing_time = client_model.evaluate(client_id)
    # client_processing_time = 92000
    client_processing_time = int(client_processing_time)
    print_log(f'client({client_id})_model_processing_time: {client_processing_time}')
    save_results(client_processing_time, "TIME", False, client_id)
    for merge_idx in merge_clients_id_list:
        client_score_list.append(client_processing_time)

    # get test scores of submodels
    # client_acc_list = []
    # client_score_list = []
    # for merge_idx in merge_clients_id_list:
    #     client_model = FedClient()
    #     client_processing_time = client_model.evaluate(merge_idx)
    #     client_acc_list.append(client_processing_time)
    #     client_score_list.append(client_processing_time)
    #     print_log(f'client({merge_idx})_model_processing_time: {client_processing_time}')

    global_model_score = global_processing_time
    retSet = {"clients_scores": client_score_list, "global_score": int(global_model_score)}
    print_log(f"merge results-> {retSet}")
    glo.set_global_var("is_federated_test", False)
    return retSet


# update local model to the newest global model
def update_model():
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
