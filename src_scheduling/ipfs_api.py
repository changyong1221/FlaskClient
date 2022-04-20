import requests
import src_scheduling.globals as glo
from src_scheduling.log import print_log


def convert_str_to_list(src_str):
    ret_list = []
    left = 0
    right = src_str.find(b' ')
    while right != -1:
        num = int(src_str[left:right])
        ret_list.append(num)
        left = right + 1
        right = src_str.find(b' ', left)
    return ret_list


def model_file_wrapper(filename, model_path, is_global):
    with open(model_path, 'rb') as f:
        content = f.read()
        f.close()
        print_log(f"wrapper filename: {filename}")

        # 1. global model
        if is_global:
            # serialize process
            # all_rounds
            all_rounds = glo.get_global_var("current_round")
            print_log(f"wrapper all_rounds: {all_rounds}")
            # client rounds list
            round_list = glo.get_global_var("clients_merge_rounds_list")
            print_log(f"wrapper round_list: {round_list}")
            round_list_str = ""
            for elem in round_list:
                round_list_str += f"{elem} "
                
            # init model
            # all_rounds = 0
            # print(f"wrapper all_rounds: {all_rounds}")
            # clients_num = 10
            # round_list = [0 for i in range(clients_num)]
            # print(f"wrapper round_list: {round_list}")
            # round_list_str = ""
            # for elem in round_list:
            #     round_list_str += f"{elem} "

            wrapped_file = f'{filename}\n{all_rounds}\n{round_list_str}\n'.encode('utf-8') + content
            # wrapped_file = f'{filename}\n'.encode('utf-8') + content
        # 2. client model
        else:
            current_data_scale = glo.get_global_var("current_data_scale")
            print_log(f"wrapper current_data_scale: {current_data_scale}")
            wrapped_file = f'{filename}\n{current_data_scale}\n'.encode('utf-8') + content
        return wrapped_file


def model_file_unwrapper(wrapped_data, is_global):
    left = 0
    right = wrapped_data.find(b'\n')
    filename = wrapped_data[left:right]
    print_log(f"unwrapper filename: {filename}")

    data_scale = 0
    if is_global:
        left = right + 1
        right = wrapped_data.find(b'\n', left)
        all_rounds = int(wrapped_data[left:right])
        glo.set_global_var("all_rounds", all_rounds)
        print_log(f'unwrapper all_rounds: {all_rounds}')
        glo.set_global_var("current_round", all_rounds)
        print_log(f'unwrapper current_round: {all_rounds}')

        left = right + 1
        right = wrapped_data.find(b'\n', left)
        round_list_str = wrapped_data[left:right]
        round_list = convert_str_to_list(round_list_str)
        print_log(f"unwrapper round_list: {round_list}")
        glo.set_global_var("clients_merge_rounds_list", round_list)
    else:
        left = right + 1
        right = wrapped_data.find(b'\n', left)
        data_scale_str = wrapped_data[left:right]
        data_scale = int(data_scale_str)
        print_log(f"unwrapper data_scale: {data_scale}")

    original_data = wrapped_data[right + 1:]
    return filename, data_scale, original_data


def cid_wrapper(cid):
    # encode cid of string type to hex type
    hex_cid = cid.encode('utf-8').hex()
    return hex_cid


def cid_unwrapper(hex_cid):
    # decode hex cid to original cid
    cid = bytes.fromhex(hex_cid).decode('utf-8')
    return cid


def upload_to_ipfs(client_id, is_global, data_path):
    if is_global:
        filename = 'global_model.pkl'
    else:
        filename = f'{client_id}.pkl'
    wrapped_model_data = model_file_wrapper(filename, data_path, is_global)
    data = {filename: wrapped_model_data}
    response = requests.post(f'http://127.0.0.1:5001/api/v0/add', files=data)
    cid = response.json()['Hash']
    # wrapped_cid = cid_wrapper(cid)
    return cid


def download_from_ipfs(client_id, is_global, cid):
    if is_global:
        save_path = f'models/global/client-{client_id}/'
    else:
        save_path = f'models/downloads/client-{client_id}/'

    # cid = cid_unwrapper(wrapped_cid)
    print_log(f"downloading: {cid}")
    response = requests.post(f'http://127.0.0.1:5001/api/v0/cat?arg={cid}')
    wrapped_data = response.content
    filename, data_scale, original_data = model_file_unwrapper(wrapped_data, is_global)
    filename = filename.decode('utf-8')
    file_path = save_path + filename
    with open(file_path, 'wb') as f:
        f.write(original_data)
        f.close()

    merge_client_id = 0
    if not is_global:
        merge_client_id_str = filename[:filename.find('.')]
        merge_client_id = (int)(merge_client_id_str)
        data_scale_list = glo.get_global_var("clients_data_scale_list")
        data_scale_list[merge_client_id - 1] = data_scale
        glo.set_global_var("clients_data_scale_list", data_scale_list)
        print_log(f"merge_client_id: {merge_client_id}")
    return merge_client_id


if __name__ == '__main__':
    # pass
    # upload file
    client_id = 1
    # data_path = "../initial_model/global_model.pkl"
    data_path = "../models/global/client-3/global_model.pkl"
    cid = upload_to_ipfs(client_id, True, data_path)
    print(cid)

    # download file
    # cid = "QmQPv43Cg6oRQiPvMxykyAtFsKpAUvqakmL4qfcFcQe7i5"
    # merge_client_id = download_from_ipfs(1, True, cid)
    # print(merge_client_id)