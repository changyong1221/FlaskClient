import src.globals as glo
glo.__init()

from flask import Flask, render_template, request, Response
from src.model_funcs import train_one_model, merge_models_and_test, has_submodel, update_model
import requests
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
import argparse
import socket
from multiprocessing import Process


app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcdefg'

@app.route("/")
def root():
    """
    :return: Index.html
    """
    return render_template('Index.html')


@app.route("/train", methods=['POST'])
def train():
    if glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training"})
    else:
        executor.submit(train_one_model)
        return json.dumps({"status": "start training..."})


def printLog(str):
    client_id = glo.get_global_var("client_id")
    print(f"client-{client_id}: {str}")

@app.route("/infoSubmit", methods=['POST'])
def submit_submodel():
    money = request.json["money"]
    printLog(f"money: {money}")
    submodel_path = glo.get_global_var("sub_model_path")
    if glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training"})
    if not has_submodel():
        train_one_model()
    data = {'file': open(submodel_path, 'rb')}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=data)

    # printLog(response.text)
    ret = {"swarm_id": response.json()["swarm_id"]}
    printLog(ret)
    return ret


@app.route("/infoWork", methods=['POST'])
def merge_models():
    if glo.get_global_var("merge_status") == "todo":
        model_ids = request.json["models"]
        # model_ids = ["10a1b887db6eb16f1c5e73da51c6b645dd0cd0b4bd17c183bdd205e502e0c29e",
        #              "01d8de78a6e0a975b5f3eed94600d2d92577348cec226776438de507b07eece1",
        #              "441707c3e702a4d61d62122ace5c6712c060d8d0fa7868108c2a9a96678b12aa",
        #              "8ed728a48823854fd87a10265e7364a377be31869d4e211523348851a2e7ac9b",
        #              "9ca9fa2d8b5b14ed64054f9c525200cde39ef7673ab3e8d438d610fba719ea04"]
        executor.submit(merge_models_work, (model_ids))
        return json.dumps({"status": "request received"})
    elif glo.get_global_var("merge_status") == "merging":
        return json.dumps({"status": "merging"})
    else:
        printLog("prepare to return...")
        job_info_path = glo.get_global_var("job_info_path")
        f = open(job_info_path, 'r+')
        job = json.load(f)
        glo.set_global_var("merge_status", "todo")
        printLog("merge results returned.")
        return job


def merge_models_work(model_ids):
    glo.set_global_var("merge_status", "merging")
    glo.set_global_var("merge_clients_num", len(model_ids))
    # get all submodels from swarm
    printLog("start downloading submodels.")
    while glo.get_global_var("download_status") == "downloading":
        time.sleep(3)
    glo.set_global_var("download_status", "downloading")
    for idx, swarm_id in enumerate(model_ids):
        data = {"swarm_id": swarm_id, "client_id": idx, "is_global": False, "client_address": local_address}
        printLog(data)
        response = requests.post(f'http://{swarm_server}/download_from_swarm', json=data)

    # merge models. merge process won't start before all submodels have been downloaded
    overtime_counter = 0
    while glo.get_global_var("download_status") != "finished":
        printLog("waiting downloading")
        time.sleep(3)   # sleep three second
        overtime_counter += 1
        if overtime_counter == 100:
            return json.dumps({'merge': 'download overtime', 'status': 'failure'})
    glo.set_global_var("download_status", "todo")
    printLog("submodels downloading finished.")
    printLog("merge start.")
    scores = merge_models_and_test()
    printLog("merge finished.")

    # upload global model
    printLog("start uploading global model.")
    global_model_path = glo.get_global_var("global_model_path")
    file = {'file': open(global_model_path, 'rb')}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=file)
    printLog("global model uploading finished.")

    # stop training when global model score reached 900
    if scores["global_score"] > 900:
        is_stop = True
    else:
        is_stop = False
    ret = {"models": model_ids, "scores": scores["clients_scores"], "fscore": scores["global_score"],
           "fmodel": response.json()['swarm_id'], "stop": is_stop}
    printLog(ret)
    job_info_path = glo.get_global_var("job_info_path")
    with open(job_info_path, 'w+') as f:
        json.dump(ret, f)

    printLog("all tasks have been done.")
    glo.set_global_var("merge_status", "finished")


# has similar function as "/infoSubmit"
@app.route("/upload")
def upload_model():
    sub_model_path = glo.get_global_var("sub_model_path")
    data = {'file': open(sub_model_path, 'rb')}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=data)

    # printLog(response.text)
    return json.dumps({'status': 'success'})


def download_update_global_model(paras):
    # swarm_id = paras
    swarm_id = paras["model"]
    score = paras['score']
    is_stop = paras['stop']

    data = {"swarm_id": swarm_id, "client_id": client_id, "is_global": True, "client_address": local_address}
    printLog(data)
    while glo.get_global_var("download_status") == "downloading":
        time.sleep(3)
    glo.set_global_var("download_status", "downloading")
    printLog("downloading global model...")
    response = requests.post(f'http://{swarm_server}/download_from_swarm', json=data)
    printLog("global model downloaded.")
    update_model()
    glo.set_global_var("update_status", "todo")


# receive model file from swarm and save file to localhost
@app.route("/receive", methods=['POST'])
def receive():
    client_id = glo.get_global_var("client_id")
    is_global_str = request.values['is_global']
    if is_global_str == "True":
        printLog("true")
        is_global = True
    else:
        printLog('false')
        is_global = False
    merge_client_id = int(request.values['client_id'])

    ff = request.files['file']
    if is_global:
        save_path = f'models/global/client-{client_id}/'
        filename = 'global_model.npy'
        ff.save(os.path.join(save_path, filename))
        glo.set_global_var("download_status", "todo")
    else:
        save_path = f'models/downloads/client-{client_id}/'
        filename = f'{merge_client_id}.npy'
        ff.save(os.path.join(save_path, filename))
        printLog(f"model[{merge_client_id}] downloaded.")
        last_client_id = glo.get_global_var("merge_clients_num") - 1
        if merge_client_id == last_client_id:
            glo.set_global_var("download_status", "finished")
    return json.dumps({'receive': 'finished', 'status': 'success'})


#  get newest global model
@app.route("/get_newest_global")
def get_newest_global_model():
    if glo.get_global_var("update_status") == "updating":
        return json.dumps({"status": "updating"})
    glo.set_global_var("update_status", "updating")
    response = request.post(f"http://{dapp_address}/interface/getNewestModel")
    # swarm_id = "739b52501f986cb8a4de782ca411cc643b3c481416f4e173c2d02bf859bee966"
    printLog(response.json())
    executor.submit(download_update_global_model, (response.json()))
    # executor.submit(download_update_global_model, (swarm_id))
    return Response(json.dumps({'status': 'success'}), mimetype="application/json")


# get ipv4 address of current machine
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('-h', '--host', dest='host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', dest='port', type=str, default='4000')
    parser.add_argument("-i", "--id", dest='id', type=int, default=0)
    parser.add_argument("-n", "--number", dest='clients_num', type=int, default=1)
    args = parser.parse_args()
    # app.run(debug=True, host='10.128.205.41', port='4000')
    # app.run(debug=True, host=local_host, port=local_port)

    # Attention: some parameters should be set in the following before first run
    local_host = get_host_ip()
    # local_port = 4001
    swarm_server_host = "10.112.58.204"
    swarm_server_port = "40000"
    # clients_num = 5
    # client_id = 0
    local_port = args.port
    client_id = args.id
    clients_num = args.clients_num

    # Initialization
    glo.set_global_var("clients_num", clients_num)
    glo.set_global_var("client_id", client_id)
    glo.set_global_var("global_model_path", f"models/global/client-{client_id}/global_model.npy")
    glo.set_global_var("sub_model_path", f"models/clients/client-{client_id}/sub_model.npy")
    glo.set_global_var("job_info_path", f"jobs_info/client-{client_id}/job.json")
    executor = ThreadPoolExecutor(10)
    swarm_server = f"{swarm_server_host}:{swarm_server_port}"
    dapp_port = int(local_port) + 1000
    dapp_address = f"localhost:{dapp_port}"
    local_address = f"{local_host}:{local_port}"
    printLog(local_address)
    # create directory

    path_list = [f"models/global/client-{client_id}",
                 f"models/clients/client-{client_id}",
                 f"models/downloads/client-{client_id}",
                 f"jobs_info/client-{client_id}"]
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    app.run(debug=True, host=local_host, port=args.port)

    # multiprocessing
    # client_nums = 5
    # for i in range(client_nums):
    #     p = Process(target=app_run, args=(local_host, str(local_port+i)))
    #     p.start()
    #     p.join()
