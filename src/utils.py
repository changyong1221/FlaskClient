import os
import pandas as pd
import matplotlib.pyplot as plt


def check_and_build_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def model_file_wrapper(model_path):
    with open(model_path, 'rb') as f:
        content = f.read()
        f.close()
        head_flag = "start_of_file".encode('utf-8')
        tail_flag = "end_of_file".encode('utf-8')
        wrapped_file = head_flag + content + tail_flag
        return wrapped_file


def save_results(clients_loss_list, clients_acc_list, clients_avg_loss_list, clients_avg_acc_list, global_acc_list):
    clients_num = len(clients_loss_list)
    # 1. save results of each client to a single file
    for client_id in range(clients_num):
        client_dir = f"../results/clients/client_{client_id}"
        check_and_build_dir(client_dir)
        save_to_file(f"{client_dir}/client_{client_id}_loss.txt", clients_loss_list[client_id])
        save_to_file(f"{client_dir}/client_{client_id}_acc.txt", clients_acc_list[client_id])
    # 2. save results of federated model to file
    global_dir = f"../results/global/client_{client_id}"
    check_and_build_dir(global_dir)
    save_to_file(f"{global_dir}/global_acc.txt", global_acc_list)
    # 3. save results of clients average loss and average accuracy
    clients_avg_dir = f"../results/clients_avg/client_{client_id}"
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
    global_data_dir = f"../results/global/client_{client_id}"
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
    model_path = "../models/clients/client-1/sub_model.pkl"
    model_file_wrapper(model_path)