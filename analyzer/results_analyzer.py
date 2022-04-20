import math
import numpy as np
import pandas as pd
import src_scheduling.globals as glo
from utils.create_pic import save_to_pic_from_list, save_to_histogram_from_list, save_to_pie_from_list
from utils.plt_config import PltConfig
from src_scheduling.log import print_log


# 分析联邦学习场景下的平均任务处理时间（算法对比）
def analyze_federated_task_processing_time_results_comp():
    # 1. settings
    path = "../backup/processing_time/"
    path_list = [
            path + "global/global_processing_time.txt",
            path + "client1/client_1_processing_time.txt"
        ]

    show_vector = []
    # 1. read data
    for data_path in path_list:
        data = pd.read_csv(data_path, header=None)
        data.columns = ['processing_time']
        print(len(data))
        show_vector.append(data['processing_time'].to_list())

    # 2. 保存图片
    output_path = f"pics/federated_task_processing_time_comp_version.png"
    plt_config = PltConfig()
    plt_config.title = "task processing time in federated learning"
    plt_config.xlabel = "federated round"
    plt_config.ylabel = "task processing time"
    labels = ["global", "client"]
    save_compare_pic_from_vector(show_vector, labels, output_path, plt_config, show=False)



# 计算客户端任务调度用时
def compute_client_avg_task_process_time(client_id, is_test=False, is_global=False):
    """Compute average task process time of different scheduling algorightm
    """
    scheduler_name = glo.get_global_var("current_scheduler")
    if is_test:
        if is_global:
            data_path = f"results/task_run_results/global/client-{client_id}/test/{scheduler_name}/{glo.get_global_var('current_round')}/{scheduler_name}_task_run_results.txt"
        else:
            data_path = f"results/task_run_results/client-{client_id}/test/{scheduler_name}/{glo.get_global_var('current_round')}/{scheduler_name}_task_run_results.txt"
    else:
        data_path = f"results/task_run_results/client-{client_id}/train/{scheduler_name}/{glo.get_global_var('current_round')}/{scheduler_name}_task_run_results.txt"
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                    'execute_time', 'process_time']
    transfer_time_mean = data['transfer_time'].mean()
    wait_time_mean = data['wait_time'].mean()
    execute_time_mean = data['execute_time'].mean()
    process_time_mean = data['process_time'].mean()
    print_log(f"{scheduler_name}_transfer_time_mean: {transfer_time_mean}")
    print_log(f"{scheduler_name}_wait_time_mean: {wait_time_mean}")
    print_log(f"{scheduler_name}_execute_time_mean: {execute_time_mean}")
    print_log(f"{scheduler_name}_process_time_mean: {process_time_mean}")
    return round(process_time_mean, 2)


# 分析联邦学习场景下的机器工作时间
def analyze_federated_machine_results():
    # 1. settings
    path = "../results/machine_status_results/client-10000/DQNScheduler/"
    machine_num = 20
    federated_rounds = 10
    repeat_rounds = 10

    # 1. machine work time
    avg_machine_work_time_list = []
    for epoch in range(federated_rounds):
        for machine_id in range(machine_num):
            data_path = f"{path}/{epoch}/{machine_id}_status_test.txt"
            machine_data = pd.read_csv(data_path, header=None, delimiter='\t')
            machine_data.columns = ['work_time', 'cpu_uti', 'mem_uti', 'band_uti']


def analyze_machine_results():
    # 1. settings
    idx = 4
    schedulers = ["RR", "Random", "Earliest", "GA", "DQN", "DDPG"]
    path_list = [
        "../results/machine_status_results/RoundRobinScheduler",
        "../results/machine_status_results/RandomScheduler",
        "../results/machine_status_results/EarliestScheduler",
        "../results/machine_status_results/HeuristicScheduler",
        "../results/machine_status_results/DQNScheduler",
        "../results/machine_status_results/DDPGScheduler",
    ]
    path = path_list[idx]
    machine_num = 20

    # 1. 机器工作时间（machine work time）
    # 2. 机器分配任务数（machine assigned tasks）
    machine_work_time_list = []
    machine_assigned_tasks_num_list = []
    for machine_id in range(machine_num):
        data_path = f"{path}/{machine_id}_status.txt"
        machine_data = pd.read_csv(data_path, header=None, delimiter='\t')
        machine_data.columns = ['work_time', 'cpu_uti', 'mem_uti', 'band_uti']
        machine_work_time = machine_data['work_time'].tolist()[-1]
        machine_work_time_list.append(machine_work_time)
        machine_assigned_tasks_num_list.append(len(machine_data))
    # 展示并保存机器工作时间图
    dest_path = f"../pic/machine_status_results/machine_work_time_{schedulers[idx]}_20machine_100000tasks.png"
    plt_config = PltConfig()
    plt_config.title = f"machine work time using {schedulers[idx]}"
    plt_config.xlabel = "machine id"
    plt_config.ylabel = "machine work time"
    plt_config.x_axis_data = [str(i) for i in range(machine_num)]
    save_to_histogram_from_list(machine_work_time_list, dest_path, plt_config, show=True, show_text=False)
    # 展示并保存机器分配任务数图
    dest_path = f"../pic/machine_assigned_tasks_num/machine_assigned_tasks_num_{schedulers[idx]}_20machine_100000tasks.png"
    plt_config = PltConfig()
    plt_config.title = f"machine assigned tasks number using {schedulers[idx]}"
    plt_config.xlabel = "machine id"
    plt_config.ylabel = "machine assigned tasks number"
    plt_config.x_axis_data = [str(i) for i in range(machine_num)]
    save_to_histogram_from_list(machine_assigned_tasks_num_list, dest_path, plt_config, show=True, show_text=True)


# 分析联邦学习场景下的平均任务处理时间
def analyze_federated_task_processing_time_results():
    # 1. settings
    path = "../results/task_run_results/client-10000/DQNScheduler_task_run_results_test.txt"
    machine_num = 20
    federated_rounds = 10
    repeat_rounds = 10
    avg_task_processing_time_list = []

    # 1. task processing time
    data = pd.read_csv(path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                    'execute_time', 'process_time']
    test_records_num = 100000
    for epoch in range(federated_rounds):
        tmp_data = data[epoch*(test_records_num*repeat_rounds): (epoch + 1)*(test_records_num*repeat_rounds)]
        avg_task_processing_time_list.append(tmp_data['process_time'].mean())
    # 2. 保存图片
    output_path = f"../pic/federated/federated_avg_task_processing_time_{federated_rounds}_rounds.png"
    plt_config = PltConfig()
    plt_config.title = "average task processing time in federated learning"
    plt_config.xlabel = "federated round"
    plt_config.ylabel = "average task processing time"
    save_to_pic_from_list(avg_task_processing_time_list, output_path, plt_config, show=True)


def compute_avg_task_process_time():
    """Compute average task process time of different scheduling algorightm
    """
    idx = 5
    schedulers = ["RR", "Random", "Earliest", "GA", "DQN", "DDPG"]
    # dataset = "GoCJ4000"
    dataset = "Alibaba1000000"
    data_path = [
        f"../results/task_run_results/{dataset}/RoundRobinScheduler/RoundRobinScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/RandomScheduler/RandomScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/EarliestScheduler/EarliestScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/HeuristicScheduler/HeuristicScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/DQNScheduler/DQNScheduler_task_run_results.txt",
        # f"../results/task_run_results/{dataset}/client-0/DQNScheduler_task_run_results_test.txt",
        # f"../results/task_run_results/{dataset}/client-10000/DQNScheduler_task_run_results_test.txt",
        f"../results/task_run_results/{dataset}/DDPGScheduler/DDPGScheduler_task_run_results2.txt",
    ]
    data_path = data_path[idx]
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                    'execute_time', 'process_time']
    transfer_time_mean = data['transfer_time'].mean()
    wait_time_mean = data['wait_time'].mean()
    execute_time_mean = data['execute_time'].mean()
    process_time_mean = data['process_time'].mean()
    print(f"{schedulers[idx]}_transfer_time_mean: {transfer_time_mean}")
    print(f"{schedulers[idx]}_wait_time_mean: {wait_time_mean}")
    print(f"{schedulers[idx]}_execute_time_mean: {execute_time_mean}")
    print(f"{schedulers[idx]}_process_time_mean: {process_time_mean}")
    # pie_list = [round(transfer_time_mean, 2), round(wait_time_mean, 2), round(execute_time_mean, 2)]
    # output_path = f"../pic/task_run_results/{schedulers[idx]}_time_distribution_20machine_2000tasks.png"
    # plt_config = PltConfig()
    # plt_config.title = f"Task processing time distribution using {schedulers[idx]}"
    # plt_config.labels = ["transfer_time", "wait_time", "execute_time"]
    # save_to_pie_from_list(pie_list, output_path, plt_config, show=True)


def compute_task_to_machine_map():
    # 1. settings
    idx = 5
    schedulers = ["RR", "Random", "Earliest", "GA", "DQN", "DDPG"]
    # dataset = "GoCJ4000"
    dataset = "Alibaba1000000"
    data_path = [
        f"../results/task_run_results/{dataset}/RoundRobinScheduler/RoundRobinScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/RandomScheduler/RandomScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/EarliestScheduler/EarliestScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/HeuristicScheduler/HeuristicScheduler_task_run_results.txt",
        f"../results/task_run_results/{dataset}/DQNScheduler/DQNScheduler_task_run_results.txt",
        # f"../results/task_run_results/{dataset}/client-0/DQNScheduler_task_run_results_test.txt",
        # f"../results/task_run_results/{dataset}/client-10000/DQNScheduler_task_run_results_test.txt",
        f"../results/task_run_results/{dataset}/DDPGScheduler/DDPGScheduler_task_run_results2.txt",
    ]
    path = data_path[idx]
    data = pd.read_csv(path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time', 'execute_time',
                    'process_time']
    test_records_num = 100000
    machine_num = 20
    total_records_num = len(data)
    epoch_num = total_records_num // test_records_num

    machine_assignment_list = []
    for machine_id in range(machine_num):
        machine_assignment_list.append(len(data[data['machine_id'] == machine_id]))
    dest_path = f"../pic/machine_assignment/{schedulers[idx]}_machine_assignment_20machine_2000tasks.png"
    plt_config = PltConfig()
    plt_config.title = f"task to machine assignment map using {schedulers[idx]}"
    plt_config.xlabel = "machine id"
    plt_config.ylabel = "task number"
    plt_config.x_axis_data = [str(i) for i in range(machine_num)]
    save_to_histogram_from_list(machine_assignment_list, dest_path, plt_config, show=True, show_text=True)


def compute_avg_task_process_time_by_name(scheduler_name):
    """Compute average task process time of different scheduling algorightm
    """
    data_path = glo.results_path_list[scheduler_name]
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time', 'execute_time',
                    'process_time']
    process_time_mean = data['process_time'].mean()
    print(f"{scheduler_name}'s average task processing time: {process_time_mean} s")


if __name__ == '__main__':
    # analyze_task_results()
    # analyze_machine_results()
    # compute_avg_task_process_time()
    # compute_task_to_machine_map()
    # analyze_federated_task_processing_time_results()
    analyze_federated_task_processing_time_results_comp()
    
