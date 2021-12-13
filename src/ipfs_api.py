import requests


def model_file_wrapper(filename, model_path):
    with open(model_path, 'rb') as f:
        content = f.read()
        f.close()
        wrapped_file = f'{filename}\n'.encode('utf-8') + content
        return wrapped_file


def model_file_unwrapper(wrapped_data):
    split_idx = wrapped_data.find(b'\n')
    filename = wrapped_data[:split_idx]
    original_data = wrapped_data[split_idx + 1:]
    return filename, original_data


def cid_wrapper(cid):
    # extend cid to 64 chars by filling with zeros
    fill_str = "000000000000000000"
    # swarm_id_len = 64
    # cid_len = 46
    # for i in range(swarm_id_len - cid_len):
    #     fill_str += "0"
    wrapped_cid = cid + fill_str
    return wrapped_cid


def cid_unwrapper(wrapped_cid):
    # unwrap wrapped cid to original cid
    cid = wrapped_cid[:46]
    return cid


def upload_to_ipfs(client_id, is_global, data_path):
    if is_global:
        filename = 'global_model.pkl'
    else:
        filename = f'{client_id}.pkl'
    wrapped_model_data = model_file_wrapper(filename, data_path)
    data = {filename: wrapped_model_data}
    response = requests.post(f'http://127.0.0.1:5001/api/v0/add', files=data)
    cid = response.json()['Hash']
    wrapped_cid = cid_wrapper(cid)
    return wrapped_cid


def download_from_ipfs(client_id, is_global, wrapped_cid):
    if is_global:
        save_path = f'models/global/client-{client_id}/'
    else:
        save_path = f'models/downloads/client-{client_id}/'

    cid = cid_unwrapper(wrapped_cid)
    response = requests.post(f'http://127.0.0.1:5001/api/v0/cat?arg={cid}')
    wrapped_data = response.content
    filename, original_data = model_file_unwrapper(wrapped_data)
    filename = filename.decode('utf-8')
    file_path = save_path + filename
    with open(file_path, 'wb') as f:
        f.write(original_data)
        f.close()

    merge_client_id = 0
    if not is_global:
        merge_client_id_str = filename[:filename.find('.')]
        merge_client_id = (int)(merge_client_id_str)
    return merge_client_id


if __name__ == '__main__':
    pass
    # upload file
    # client_id = 1
    # data_path = "../models/train/test.txt"
    # cid = upload_to_ipfs(client_id, False, data_path)
    # print(cid)

    # download file
    # cid = "QmaRXcHprtJeapFQiqLiqz4XWZAUxC86dUK6dwgYqLmCq1"
    # merge_client_id = download_from_ipfs(1, False, cid)
    # print(merge_client_id)