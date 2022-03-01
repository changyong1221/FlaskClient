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
    wrapped_model_data = model_file_wrapper(filename, data_path)
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
    client_id = 1
    data_path = "../initial_model/global_model.pkl"
    cid = upload_to_ipfs(client_id, True, data_path)
    print(cid)

    # download file
    # cid = "QmaRXcHprtJeapFQiqLiqz4XWZAUxC86dUK6dwgYqLmCq1"
    # merge_client_id = download_from_ipfs(1, False, cid)
    # print(merge_client_id)