import datetime

import src.globals as glo
glo.__init()

from flask import Flask, render_template, request, Response
from src.model_funcs import train_one_model, merge_models_and_test, has_submodel, update_model
from src.log import printLog
from src.get_data import DataSet
from src.utils import model_file_wrapper
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


@app.route("/train", methods=['GET'])
def train():
    if glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training"})
    else:
        executor.submit(train_one_model, (dataset))
        return json.dumps({"status": "start training..."})


@app.route("/infoSubmit", methods=['GET'])
def submit_submodel():
    # money = request.json["money"]
    # printLog(f"money: {money}")
    if glo.get_global_var("train_status") == "todo":
        executor.submit(train_working)
        return json.dumps({"status": "start training"})
    elif glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training..."})
    else:
        train_job_info_path = f"{glo.get_global_var('job_info_path')}/train_job.json"
        f = open(train_job_info_path, 'r+')
        job = json.load(f)
        printLog("swarm_id returned.")
        glo.set_global_var("train_status", "todo")
        return job


def train_working():
    printLog("start training...")
    glo.set_global_var("train_status", "training")
    train_one_model(dataset)
    printLog("train finished.")
    submodel_path = glo.get_global_var("sub_model_path")
    client_id = glo.get_global_var("client_id")
    wrapped_model_data = model_file_wrapper(submodel_path)
    data = {f'{client_id}.pkl': wrapped_model_data}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=data)

    # printLog(response.text)
    ret = {"swarm_id": response.json()["swarm_id"]}
    printLog(ret)
    printLog("model uploaded.")
    train_job_info_path = f"{glo.get_global_var('job_info_path')}/train_job.json"
    with open(train_job_info_path, 'w+') as f:
        json.dump(ret, f)
    glo.set_global_var("train_status", "finished")


@app.route("/infoWork", methods=['GET'])
def merge_models():
    if glo.get_global_var("merge_status") == "todo":
        # model_ids = request.json["models"]
        model_ids = ["6a1868af9913d472b79ea6db07a0a19f105a700789dc5231d60b5e16b07584fb",
                     "1986edaaf2e2ef29adb310bce2c461fa065e2c6d5be8db08498f4b45fd66c927",
                     "3caa47346a5523df736d5d259e6d82412b87725aaad470ac2c413815406d7325"]
                     # "6a4486afede96d6cdeb7de4f84c8af37bb95ee7dd846918e5a3a86131c3ea2d1",
                     # "8f788788ae661f9e6537f6de699ac47eabcd47c6022098923b1be3c6524e4396",
                     # "c4a3808ff57fb41a23bad2ce1573245d5d741ca9778019c6b1e985baa538cad0",
                     # "7a588b97e5f236c539ec5db147c079600ffc736b450d2440638ccf134d27cd92"]
        executor.submit(merge_models_work, (model_ids))
        return json.dumps({"status": "request received"})
    elif glo.get_global_var("merge_status") == "merging":
        return json.dumps({"status": "merging"})
    else:
        printLog("prepare to return...")
        merge_job_info_path = f"{glo.get_global_var('job_info_path')}/merge_job.json"
        f = open(merge_job_info_path, 'r+')
        job = json.load(f)
        glo.set_global_var("merge_status", "todo")
        printLog("merge results returned.")
        return job


def merge_models_work(model_ids):
    glo.set_global_var("merge_status", "merging")
    glo.set_global_var("merge_clients_num", len(model_ids))
    glo.set_global_var("download_num", len(model_ids))
    # get all submodels from swarm
    printLog("start downloading submodels.")
    while glo.get_global_var("download_status") == "downloading":
        time.sleep(3)
    glo.set_global_var("download_status", "downloading")
    for swarm_id in model_ids:
        data = {"swarm_id": swarm_id, "is_global": False, "client_address": local_address}
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
    scores = merge_models_and_test(dataset)
    printLog("merge finished.")

    # upload global model
    printLog("start uploading global model.")
    global_model_path = glo.get_global_var("global_model_path")
    wrapped_model_data = model_file_wrapper(global_model_path)
    file = {'global_model.pkl': wrapped_model_data}
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
    merge_job_info_path = f"{glo.get_global_var('job_info_path')}/merge_job.json"
    with open(merge_job_info_path, 'w+') as f:
        json.dump(ret, f)

    printLog("all tasks have been done.")
    glo.set_global_var("merge_status", "finished")
    glo.get_global_var("merge_clients_id_list").clear()


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

    data = {"swarm_id": swarm_id, "is_global": True, "client_address": local_address}
    printLog(data)
    while glo.get_global_var("download_status") == "downloading":
        time.sleep(3)
    glo.set_global_var("download_status", "downloading")
    printLog("downloading global model...")
    response = requests.post(f'http://{swarm_server}/download_from_swarm', json=data)
    printLog("global model downloaded.")
    update_model(dataset)
    glo.set_global_var("update_status", "todo")


# receive model file from swarm and save file to localhost
@app.route("/receive", methods=['POST'])
def receive():
    client_id = glo.get_global_var("client_id")
    is_global_str = request.values['is_global']
    if is_global_str == "True":
        printLog("is_global: true")
        is_global = True
    else:
        printLog('is_global: false')
        is_global = False
        merge_client_id = int(request.values['client_id'])
        glo.get_global_var("merge_clients_id_list").append(merge_client_id)
        printLog(f"merge_client_id: {merge_client_id}")

    for elem in request.files:
        filename = elem
    printLog(f"filename: {filename}")
    ff = request.files[filename]
    if is_global:
        save_path = f'models/global/client-{client_id}/'
        ff.save(os.path.join(save_path, filename))
        glo.set_global_var("download_status", "todo")
    else:
        save_path = f'models/downloads/client-{client_id}/'
        ff.save(os.path.join(save_path, filename))
        printLog(f"model[{merge_client_id}] downloaded.")
        if glo.get_global_var("download_num") > 0:
            glo.set_global_var("download_num", glo.get_global_var("download_num") - 1)
        if glo.get_global_var("download_num") == 0:
            glo.set_global_var("download_status", "finished")
    return json.dumps({'receive': 'finished', 'status': 'success'})


#  get newest global model
@app.route("/get_newest_global", methods=['GET'])
def get_newest_global_model():
    if glo.get_global_var("update_status") == "updating":
        return json.dumps({"status": "updating"})
    glo.set_global_var("update_status", "updating")
    # response = request.post(f"http://{dapp_address}/interface/getNewestModel")
    paras = {"model": "09fff06b12bcf744a142437d14f781532cf69f7782c86b1540925402811d8f78", "score": 101, "stop": False}
    # printLog(response.json())
    # executor.submit(download_update_global_model, (response.json()))
    executor.submit(download_update_global_model, (paras))
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
    parser.add_argument('-p', '--port', dest='port', type=str, default='4001')
    parser.add_argument("-i", "--id", dest='id', type=int, default=1)
    parser.add_argument("-n", "--number", dest='clients_num', type=int, default=10)
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
    client_id = 4
    clients_num = args.clients_num
    now = datetime.datetime.now()
    date = f"{now.year}-{now.month}-{now.day}"

    # Initialization
    glo.set_global_var("clients_num", clients_num)
    glo.set_global_var("client_id", client_id)
    glo.set_global_var("global_model_path", f"models/global/client-{client_id}/global_model.pkl")
    glo.set_global_var("sub_model_path", f"models/clients/client-{client_id}/sub_model.pkl")
    glo.set_global_var("job_info_path", f"jobs_info/client-{client_id}")
    glo.set_global_var("log_path", f"logs/{date}/client-{client_id}.log")
    executor = ThreadPoolExecutor(10)
    swarm_server = f"{swarm_server_host}:{swarm_server_port}"
    dapp_port = int(local_port) + 1000
    dapp_address = f"localhost:{dapp_port}"
    local_address = f"{local_host}:{local_port}"

    # create directory
    path_list = [f"models/global/client-{client_id}",
                 f"models/clients/client-{client_id}",
                 f"models/downloads/client-{client_id}",
                 f"jobs_info/client-{client_id}",
                 f"logs/{date}"]
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

    is_iid = False
    dataset = DataSet(clients_num, is_iid)

    app.run(host=local_host, port=args.port)

    # multiprocessing
    # client_nums = 5
    # for i in range(client_nums):
    #     p = Process(target=app_run, args=(local_host, str(local_port+i)))
    #     p.start()
    #     p.join()
