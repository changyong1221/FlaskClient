import numpy as np
import pandas as pd

from utils.read_data import read_line_elems_from_file
from utils.plt_config import PltConfig
from utils.create_pic import save_to_histogram_from_list, save_to_pic_from_list


def analyze_dataset(file_path):
    """Analyze the data distribution of given dataset
    """
    # 1. compute range_set
    range_list = [525000, 150000, 101000, 59000, 15000, 0]
    range_name_list = ["Huge", "Extra-Large", "Large", "Medium", "Small", "Very-Small"]
    range_set = {
        "Huge": 0,
        "Extra-Large": 0,
        "Large": 0,
        "Medium": 0,
        "Small": 0,
        "Very-Small": 0
    }

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    data_vector = read_line_elems_from_file(file_path, delimiter='\t')
    data_size = len(data_vector)
    print(data_size)
    for line in data_vector:
        # print(line)
        idx = find_range(range_list, int(line[2]))
        # print(idx)
        range_set[range_name_list[idx]] += 1
    print(range_set)

    # 2. plot data
    x_axis_data = ['Very-Small', 'Small', 'Medium', 'Large', 'Extra-Large', 'Huge']
    y_axis_data = []
    for elem in x_axis_data:
        y_axis_data.append(range_set[elem])
    y_axis_data = np.array(y_axis_data)
    y_axis_data = y_axis_data / y_axis_data.sum()
    y_axis_data = y_axis_data * 100
    print(y_axis_data)
    plt_config = PltConfig()
    plt_config.title = f"Alibaba Cluster Trace dataset distribution (size:{data_size})"
    plt_config.xlabel = "task length"
    plt_config.ylabel = "number of records"
    plt_config.x_axis_data = x_axis_data
    # dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}.png"
    dest_dir = f"../pic/dataset_analysis_results/alibaba/Alibaba_Cluster_Trace_dataset_distribution_size_{data_size}_client_4.png"
    save_to_histogram_from_list(y_axis_data, dest_dir, plt_config, show=True)


def analyze_gocj_dataset(file_path):
    """Analyze the data distribution of given GoCJ dataset
    """
    # 1. compute range_set
    range_list = [525000, 150000, 101000, 59000, 15000]
    range_name_list = ["Huge", "Extra-Large", "Large", "Medium", "Small"]
    range_set = {
        "Huge": 0,
        "Extra-Large": 0,
        "Large": 0,
        "Medium": 0,
        "Small": 0
    }

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    data_vector = read_line_elems_from_file(file_path, delimiter='\t')
    data_size = len(data_vector)
    for line in data_vector:
        idx = find_range(range_list, int(line[2]))
        range_set[range_name_list[idx]] += 1
    print(range_set)

    # 2. plot data
    x_axis_data = ['Small', 'Medium', 'Large', 'Extra-Large', 'Huge']
    y_axis_data = []
    for elem in x_axis_data:
        y_axis_data.append(range_set[elem])
    y_axis_data = np.array(y_axis_data)
    y_axis_data = y_axis_data / y_axis_data.sum()
    y_axis_data = y_axis_data * 100
    print(y_axis_data)
    plt_config = PltConfig()
    plt_config.title = f"GoCJ dataset distribution (size:{data_size})"
    plt_config.xlabel = "task length"
    plt_config.ylabel = "number of records"
    plt_config.x_axis_data = x_axis_data
    # dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}.png"
    dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}_client_5.png"
    save_to_histogram_from_list(y_axis_data, dest_dir, plt_config, show=True)


def analyze_gocj_test_dataset(file_path):
    """Analyze the data distribution of given GoCJ dataset
    """
    # 1. compute range_set
    range_list = [525000, 150000, 101000, 59000, 15000]
    range_name_list = ["Huge", "Extra-Large", "Large", "Medium", "Small"]
    range_set = {
        "Huge": 0,
        "Extra-Large": 0,
        "Large": 0,
        "Medium": 0,
        "Small": 0
    }

    def find_range(lst, val):
        lst_len = len(lst)
        for i in range(lst_len):
            if val >= lst[i]:
                return i

    data_vector = read_line_elems_from_file(file_path, delimiter='\t')
    print(data_vector[:10])
    print(type(data_vector))
    data_size = len(data_vector)
    for line in data_vector:
        idx = find_range(range_list, int(line[2] / line[3]))
        range_set[range_name_list[idx]] += 1
    print(range_set)

    # 2. plot data
    x_axis_data = ['Small', 'Medium', 'Large', 'Extra-Large', 'Huge']
    y_axis_data = []
    for elem in x_axis_data:
        y_axis_data.append(range_set[elem])
    y_axis_data = np.array(y_axis_data)
    y_axis_data = y_axis_data / y_axis_data.sum()
    y_axis_data = y_axis_data * 100
    print(y_axis_data)
    plt_config = PltConfig()
    plt_config.title = f"GoCJ dataset distribution (size:{data_size})"
    plt_config.xlabel = "task length"
    plt_config.ylabel = "number of records"
    plt_config.x_axis_data = x_axis_data
    # dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}.png"
    dest_dir = f"../pic/GoCJ_dataset_distribution_size_{data_size}_real_task_mi.png"
    save_to_histogram_from_list(y_axis_data, dest_dir, plt_config, show=True)


def show_task_commit_distribution(file_path):
    data = pd.read_csv(file_path, delimiter='\t')
    data.columns = ['task_id', 'commit_time', 'length', 'cpu_uti', 'size']
    commit_time_list = data['commit_time'].unique()
    batch_num = len(commit_time_list)
    print("batch_num: ", batch_num)
    commit_tasks_num_list = []
    for i in range(batch_num):
        commit_tasks_num_list.append(len(data[data['commit_time'] == commit_time_list[i]]))
    print(len(commit_tasks_num_list))
    print(commit_tasks_num_list)
    avg_commit_tasks_num = len(data) / batch_num
    print("avg_commit_tasks_num: ", avg_commit_tasks_num)

    plt_config = PltConfig()
    plt_config.title = "task commit concurrency on Alibaba Cluster Trace dataset"
    plt_config.xlabel = "commit time"
    plt_config.ylabel = "task commit concurrency"
    save_path = "../pic/task_commit_concurrency/task_commit_concurrency_on_Alibaba_dataset_100000_test.png"
    save_to_pic_from_list(commit_tasks_num_list, save_path, plt_config, show=True)


if __name__ == '__main__':
    # filepath = "../dataset/GoCJ/GoCJ_Dataset_500batches_60concurrency_test.txt"
    # filepath = "../dataset/GoCJ/client/GoCJ_Dataset_2000_client_9.txt"
    filepath = "../dataset/Alibaba/Alibaba-Cluster-trace-100000-test.txt"
    # filepath = "../dataset/Alibaba/Alibaba-Cluster-trace-5000-test.txt"
    # filepath = "../dataset/Alibaba/client/Alibaba-Cluster-trace-100000-client-0.txt"
    # analyze_dataset(filepath)
    # analyze_gocj_test_dataset(filepath)
    show_task_commit_distribution(filepath)
