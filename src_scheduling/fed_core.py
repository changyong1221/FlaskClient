import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import optim
import copy
from src_scheduling.log import print_log
import math
import time
from core.domain import create_domains, create_multi_domain
from scheduler.RoundRobinScheduler import RoundRobinScheduler
from scheduler.DQNScheduler import DQNScheduler
from analyzer.results_analyzer import compute_client_avg_task_process_time
from utils.load_data import load_machines_from_file, load_task_batches_from_file, sample_tasks_from_file, \
    load_tasks_from_file, sample_task_batches_from_file
from utils.file_check import check_and_build_dir
from utils.state_representation import get_machine_kind_list
import src_scheduling.globals as glo


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedClient(nn.Module):
    def __init__(self):
        super(FedClient, self).__init__()

    def train(self, n_batches):
        """Perform inter-domain task scheduling
        """
        # prepare work
        client_id = glo.get_global_var("client_id")

        # 1. create multi-domain system
        multi_domain_system_location = "北京市"
        multi_domain = create_multi_domain(client_id, multi_domain_system_location)

        # 2. create domains
        domain_num = 5
        location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
        domain_list = create_domains(location_list)

        # 3. add machines to domain
        machine_file_path = "dataset/machine/machine.txt"
        machine_list = load_machines_from_file(machine_file_path)
        machine_num_per = len(machine_list) // domain_num
        for domain_id in range(domain_num):
            for i in range(machine_num_per):
                machine = machine_list[i + domain_id * machine_num_per]
                machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
                domain_list[domain_id].add_machine(machine)

        # 4. clustering machines in each domain
        cluster_num = 3
        for domain in domain_list:
            domain.clustering_machines(cluster_num)

        # 5. add domain to multi-domain system
        for domain in domain_list:
            multi_domain.add_domain(domain)

        # 6. load tasks
        task_file_path = f"dataset/Alibaba/client/Alibaba-Cluster-trace-100000-client-{client_id - 1}.txt"
        task_batch_list = sample_task_batches_from_file(task_file_path, batch_num=n_batches, delimiter='\t')

        # compute tasks num in batches
        train_tasks_num = 0
        for batch in task_batch_list:
            train_tasks_num += len(batch)
        print_log(f"train_tasks_num: {train_tasks_num}")
        train_tasks_accu = glo.get_global_var("current_data_scale")
        train_tasks_accu += train_tasks_num
        glo.set_global_var("current_data_scale", train_tasks_accu)
        print_log(f"train_tasks_accu: {train_tasks_accu}")

        # 7. set scheduler for multi-domain system
        machine_num = len(machine_list)
        task_batch_num = len(task_batch_list)
        print_log(f"training batches number: {task_batch_num}")
        # scheduler = RoundRobinScheduler(machine_num)
        machine_kind_num_list, machine_kind_idx_range_list = get_machine_kind_list(machine_list)

        epoch = glo.get_global_var("current_round")
        balance_prob_target = 0.3
        balance_prob_init = 1.0
        epochs = 100
        diff = balance_prob_init - balance_prob_target
        balance_prob_step = diff / epochs
        if epoch < 1:
            epoch = 1
        if epoch > epochs:
            epoch = epochs
        balance_prob = balance_prob_init - balance_prob_step * (epoch - 1)
        scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, machine_kind_num_list,
                                 machine_kind_idx_range_list, balance_prob=balance_prob)
        multi_domain.set_scheduler(scheduler)

        # 8. commit tasks to multi-domain system, training
        for batch in task_batch_list:
            multi_domain.commit_tasks(batch)
        print_log("return to reset")
        # 9. reset multi-domain system
        multi_domain.reset()

    def evaluate(self, client_id):
        """Perform inter-domain task scheduling
        """
        # 1. create multi-domain system
        multi_domain_system_location = "北京市"
        multi_domain = create_multi_domain(client_id, multi_domain_system_location)

        # 2. create domains
        domain_num = 5
        location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
        domain_list = create_domains(location_list)

        # 3. add machines to domain
        machine_file_path = "dataset/machine/machine.txt"
        machine_list = load_machines_from_file(machine_file_path)
        machine_num_per = len(machine_list) // domain_num
        for domain_id in range(domain_num):
            for i in range(machine_num_per):
                machine = machine_list[i + domain_id * machine_num_per]
                machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
                domain_list[domain_id].add_machine(machine)

        # 4. clustering machines in each domain
        cluster_num = 3
        for domain in domain_list:
            domain.clustering_machines(cluster_num)

        # 5. add domain to multi-domain system
        for domain in domain_list:
            multi_domain.add_domain(domain)

        # 6. load tasks
        # task_file_path = f"dataset/GoCJ/GoCJ_Dataset_5000records_50concurrency_test.txt"
        task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-100000-test.txt"
        # task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-5000-test.txt"
        tasks_for_test = load_task_batches_from_file(task_file_path, delimiter='\t')

        # 7. set scheduler for multi-domain system
        machine_num = len(machine_list)
        task_batch_num = len(tasks_for_test)
        # scheduler = RoundRobinScheduler(machine_num)
        machine_kind_num_list, machine_kind_idx_range_list = get_machine_kind_list(machine_list)

        epoch = glo.get_global_var("current_round")
        balance_prob_target = 0.3
        balance_prob_init = 1.0
        epochs = 100
        diff = balance_prob_init - balance_prob_target
        balance_prob_step = diff / epochs
        if epoch > epochs:
            epoch = epochs
        balance_prob = balance_prob_init - balance_prob_step * epoch
        balance_prob = balance_prob * 2
        if balance_prob > 1.0:
            balance_prob = 1.0
        print_log(f"client model testing...")
        print_log(f"epoch: {epoch}")
        print_log(f"balance_prob: {balance_prob}")
        scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, machine_kind_num_list,
                                 machine_kind_idx_range_list, balance_prob=balance_prob)

        multi_domain.set_scheduler(scheduler)

        # 8. commit tasks to multi-domain system, training
        for batch in tasks_for_test:
            multi_domain.commit_tasks(batch)
        # multi_domain.commit_tasks(tasks_for_test)

        # 9. reset multi-domain system
        multi_domain.reset()

        processing_time = compute_client_avg_task_process_time(client_id, is_test=True)
        return processing_time


    def load_model(self, file_path):
        pass

    def save_model(self, save_path):
        pass


class FedServer(nn.Module):
    def __init__(self):
        super(FedServer, self).__init__()

    def fed_avg(self, model_path_list, merge_clients_id_list):
        # FL average
        # load client weights and do weights processing
        client_id = glo.get_global_var("client_id")
        
        clients_weights_sum = None
        clients_num = len(model_path_list)
        round_list = glo.get_global_var("clients_merge_rounds_list")
        all_rounds = glo.get_global_var("current_round")
        data_scale_list = glo.get_global_var("clients_data_scale_list")
        print_log(f"merge_clients_id_list: {merge_clients_id_list}")
        print_log(f"federated round: {all_rounds}")
        print_log(f"round_list: {round_list}")
        print_log(f"data_scale_list: {data_scale_list}")
        all_data_scale = 0
        all_merge_rounds = 0
        for merge_id in merge_clients_id_list:
            all_data_scale += data_scale_list[merge_id - 1]
            all_merge_rounds += round_list[merge_id - 1]
        print_log(f"all_data_scale: {all_data_scale}")
        print_log(f"all_merge_rounds: {all_merge_rounds}")
        for i, model_path in enumerate(model_path_list):
            # compute weights
            merge_idx = merge_clients_id_list[i] - 1;
            print_log(f"merge_idx: {merge_idx}")
            round_weight = round_list[merge_idx] / all_merge_rounds
            data_scale_weight = data_scale_list[merge_idx] / all_data_scale
            print_log(f"round_weight: {round_weight}")
            print_log(f"data_scale_weight: {data_scale_weight}")
            weight = round_weight * 0.5 + data_scale_weight * 0.5
            print_log(f"weight: {weight}")
            if weight <= 0:
                print_log("weight adjust")
                weight = 1 / len(merge_clients_id_list)
                print_log(f"adjusted weight: {weight}")

            cur_parameters = torch.load(model_path)
            if clients_weights_sum is None:
                clients_weights_sum = {}
                for key, var in cur_parameters.items():
                    clients_weights_sum[key] = var.clone() * weight
            else:
                for var in cur_parameters:
                    clients_weights_sum[var] = clients_weights_sum[var] + cur_parameters[var] * weight

        # fed_avg
        global_weights = {}
        for var in clients_weights_sum:
            global_weights[var] = clients_weights_sum[var]
        global_model_path = glo.get_global_var("global_model_path")
        torch.save(global_weights, global_model_path)

    def evaluate(self):
        """Perform inter-domain task scheduling
        """
        # 1. create multi-domain system
        multi_domain_system_location = "北京市"
        client_id = 10000  # federated server
        multi_domain = create_multi_domain(client_id, multi_domain_system_location)

        # 2. create domains
        domain_num = 5
        location_list = ["北京市", "上海市", "莫斯科", "新加坡市", "吉隆坡"]
        domain_list = create_domains(location_list)

        # 3. add machines to domain
        machine_file_path = "dataset/machine/machine.txt"
        machine_list = load_machines_from_file(machine_file_path)
        machine_num_per = len(machine_list) // domain_num
        for domain_id in range(domain_num):
            for i in range(machine_num_per):
                machine = machine_list[i + domain_id * machine_num_per]
                machine.set_location(domain_list[domain_id].longitude, domain_list[domain_id].latitude)
                domain_list[domain_id].add_machine(machine)

        # 4. clustering machines in each domain
        cluster_num = 3
        for domain in domain_list:
            domain.clustering_machines(cluster_num)

        # 5. add domain to multi-domain system
        for domain in domain_list:
            multi_domain.add_domain(domain)

        # 6. load tasks
        # task_file_path = f"dataset/GoCJ/GoCJ_Dataset_5000records_50concurrency_test.txt"
        task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-100000-test.txt"
        # task_file_path = f"dataset/Alibaba/Alibaba-Cluster-trace-5000-test.txt"
        tasks_for_test = load_task_batches_from_file(task_file_path, delimiter='\t')

        # 7. set scheduler for multi-domain system
        machine_num = len(machine_list)
        task_batch_num = len(tasks_for_test)
        # scheduler = RoundRobinScheduler(machine_num)
        machine_kind_num_list, machine_kind_idx_range_list = get_machine_kind_list(machine_list)

        epoch = glo.get_global_var("current_round")
        balance_prob_target = 0.3
        balance_prob_init = 0.8
        epochs = 100
        diff = balance_prob_init - balance_prob_target
        balance_prob_step = diff / epochs
        if epoch > epochs:
            epoch = epochs
        balance_prob = balance_prob_init - balance_prob_step * epoch
        print_log(f"global model testing...")
        print_log(f"epoch: {epoch}")
        print_log(f"balance_prob: {balance_prob}")
        scheduler = DQNScheduler(multi_domain.multidomain_id, machine_num, task_batch_num, machine_kind_num_list,
                                 machine_kind_idx_range_list, balance_prob=balance_prob)

        multi_domain.set_scheduler(scheduler)

        # 8. commit tasks to multi-domain system, training
        for batch in tasks_for_test:
            multi_domain.commit_tasks(batch)
        # multi_domain.commit_tasks(tasks_for_test)

        # 9. reset multi-domain system
        multi_domain.reset()

        processing_time = compute_client_avg_task_process_time(10000, is_test=True)
        return processing_time


    def save_model(self, save_path):
        pass


if __name__ == '__main__':
    pass