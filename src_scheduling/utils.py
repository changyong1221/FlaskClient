import os
import pandas as pd
import matplotlib.pyplot as plt
import src_scheduling.globals as glo



def check_and_build_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_results(data, data_type, is_global, client_id):
    if is_global:
        save_path = f"results/processing_time/client-{client_id}/global_processing_time.txt"
    else:
        if data_type is 'LOSS':
            save_path = f"results/processing_time/client-{client_id}/client_{client_id}_loss.txt"
        elif data_type is 'TIME':
            save_path = f"results/processing_time/client-{client_id}/client_{client_id}_processing_time.txt"
    append_data_to_file(save_path, data)


def append_data_to_file(file_path, data):
    with open(file_path, 'a+') as f:
        f.write(str(data) + '\n')
    f.close()


def append_list_to_file(file_path, content_list):
    with open(file_path, 'a+') as f:
        for line in content_list:
            f.write(str(line) + '\n')
    f.close()


def generate_pics():
    pass


def save_pics(clients_num):
    # plt config
    plt_config = {
        "title" : "",
        "xlabel" : "federated rounds",
        "ylabel" : "",
    }
    clients_data_dir = "../results"
    global_data_dir = f"../results/global"

    for client_id in range(clients_num):
        # 1. process loss data
        client_loss_file_path = f"{clients_data_dir}/client-{client_id}/client_{client_id}_loss.txt"
        client_pic_dir = f"../pic/client-{client_id}"
        check_and_build_dir(client_pic_dir)
        client_loss_pic_path = f"{client_pic_dir}/client_{client_id}_loss.png"
        plt_config["title"] = f"loss of client-{client_id}"
        plt_config["ylabel"] = "loss"
        save_to_pic(client_loss_file_path, client_loss_pic_path, plt_config)

        # 2. process acc data
        client_acc_file_path = f"{clients_data_dir}/client-{client_id}/client_{client_id}_acc.txt"
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
    pass