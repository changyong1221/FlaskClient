import src.globals as glo
glo.__init()

from flask import Flask, render_template, request, Response
from src.model_funcs import train_one_model, merge_models_and_test, has_submodel, update_model
from src.log import print_log
from src.get_data import DataSet
from src.ipfs_api import upload_to_ipfs, download_from_ipfs
import os
import json
from concurrent.futures import ThreadPoolExecutor
import argparse
import socket
from multiprocessing import Process
import shutil


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
        executor.submit(train_one_model, (dataset))
        return json.dumps({"status": "start training..."})


@app.route("/infoSubmit", methods=['POST'])
def submit_submodel():
    if glo.get_global_var("update_status") is not "todo":
        return json.dumps({"status": "updating"})
    if glo.get_global_var("merge_status") is not "todo":
        return json.dumps({"status": "merging"})
    money = request.json["money"]
    print_log(f"money: {money}")
    if glo.get_global_var("train_status") == "todo":
        executor.submit(train_working)
        return json.dumps({"status": "start training"})
    elif glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training..."})
    else:
        train_job_info_path = f"{glo.get_global_var('job_info_path')}/train_job.json"
        f = open(train_job_info_path, 'r+')
        job = json.load(f)
        print_log("cid returned.")
        glo.set_global_var("train_status", "todo")
        return job


def train_working():
    print_log("start training...")
    glo.set_global_var("train_status", "training")
    train_one_model(dataset)
    print_log("train finished.")
    submodel_path = glo.get_global_var("sub_model_path")
    client_id = glo.get_global_var("client_id")
    cid = upload_to_ipfs(client_id, False, submodel_path)

    # print_log(response.text)
    ret = {"cid": cid}
    print_log(ret)
    print_log("model uploaded.")
    train_job_info_path = f"{glo.get_global_var('job_info_path')}/train_job.json"
    with open(train_job_info_path, 'w+') as f:
        json.dump(ret, f)
    glo.set_global_var("train_status", "finished")


@app.route("/infoWork", methods=['POST'])
def merge_models():
    if glo.get_global_var("update_status") is not "todo":
        return json.dumps({"status": "updating"})
    if glo.get_global_var("train_status") is not "todo":
        return json.dumps({"status": "training"})
    if glo.get_global_var("merge_status") == "todo":
        model_ids = request.json["models"]
        # model_ids = ["QmQs9BHpYyeNUSkfCBJAxccsCZ8uBh3MFxvtprUTadve66000000000000000000",
        #              "QmTFLTt94Km1A5w2TXSBPBsKu46c64795tFURQGNk13tF7000000000000000000",
        #              "QmP1iU7kZJ7ExJvsHX8ddkN4RjRMRhMShExDdjsfQPhqyH000000000000000000",
        #              "QmeoNqpk4MuLY74CiARbGFcXNwv8T2PUMSkU34mU29rZaD000000000000000000"]
        executor.submit(merge_models_work, (model_ids))
        return json.dumps({"status": "request received, start merging"})
    elif glo.get_global_var("merge_status") == "merging":
        return json.dumps({"status": "merging"})
    else:
        print_log("prepare to return...")
        merge_job_info_path = f"{glo.get_global_var('job_info_path')}/merge_job.json"
        f = open(merge_job_info_path, 'r+')
        job = json.load(f)
        glo.set_global_var("merge_status", "todo")
        print_log("merge results returned.")
        return job


def merge_models_work(model_ids):
    glo.set_global_var("merge_status", "merging")
    glo.set_global_var("merge_clients_num", len(model_ids))
    client_id = glo.get_global_var("client_id")
    # get all submodels from swarm
    print_log("start downloading submodels.")
    for cid in model_ids:
        merge_client_id = download_from_ipfs(client_id, False, cid)
        glo.get_global_var("merge_clients_id_list").append(merge_client_id)

    # merge models. merge process won't start before all submodels have been downloaded
    print_log("submodels downloading finished.")
    print_log("merge start.")
    scores = merge_models_and_test(dataset)
    print_log("merge finished.")

    # upload global model
    print_log("start uploading global model.")
    global_model_path = glo.get_global_var("global_model_path")
    global_cid = upload_to_ipfs(0, True, global_model_path)
    print_log("global model uploading finished.")

    # stop training when global model score reached 900
    if scores["global_score"] > 900:
        is_stop = True
    else:
        is_stop = False
    ret = {"models": model_ids, "scores": scores["clients_scores"], "fscore": scores["global_score"],
           "fmodel": global_cid, "stop": is_stop}
    print_log(ret)
    merge_job_info_path = f"{glo.get_global_var('job_info_path')}/merge_job.json"
    with open(merge_job_info_path, 'w+') as f:
        json.dump(ret, f)

    print_log("all tasks have been done.")
    glo.set_global_var("merge_status", "finished")
    glo.get_global_var("merge_clients_id_list").clear()


# has similar function as "/infoSubmit"
@app.route("/upload")
def upload_model():
    sub_model_path = glo.get_global_var("sub_model_path")
    client_id = glo.get_global_var("client_id")
    cid = upload_to_ipfs(client_id, False, sub_model_path)

    return json.dumps({'status': 'success', 'cid': cid})


def download_and_update_global_model(paras):
    # swarm_id = paras
    cid = paras["model"]
    score = paras['score']
    is_stop = paras['stop']

    print_log("downloading global model...")
    client_id = glo.get_global_var("client_id")
    global_id_tmp = download_from_ipfs(client_id, True, cid)
    print_log("global model downloaded.")
    update_model(dataset)
    glo.set_global_var("update_status", "todo")


#  get newest global model
@app.route("/getNewestModel", methods=['POST'])
def get_newest_global_model():
    if glo.get_global_var("train_status") is not "todo":
        return json.dumps({"status": "training"})
    if glo.get_global_var("merge_status") is not "todo":
        return json.dumps({"status": "merging"})
    if glo.get_global_var("update_status") == "updating":
        return json.dumps({"status": "updating"})
    glo.set_global_var("update_status", "updating")
    # response = request.post(f"http://{dapp_address}/interface/getNewestModel")
    # paras = {"model": request.json["model"], "score": request.json["score"], "stop": request.json["stop"]}
    # print_log(response.json())
    # executor.submit(download_and_update_global_model, (response.json()))
    executor.submit(download_and_update_global_model, (request.json))
    # executor.submit(download_and_update_global_model, (paras))
    return Response(json.dumps({'status': 'start updating'}), mimetype="application/json")


# get ipv4 address of current machine
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
    
    
def initialize_global_model():
    global_model_path = glo.get_global_var("global_model_path")
    initial_model_path = "initial_model/global_model.pkl"
    shutil.copyfile(initial_model_path, global_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('-h', '--host', dest='host', type=str, default='0.0.0.0')
    parser.add_argument('-p', '--port', dest='port', type=str, default='10001')
    parser.add_argument("-i", "--id", dest='id', type=int, default=1)
    parser.add_argument("-n", "--number", dest='clients_num', type=int, default=10)
    args = parser.parse_args()
    # app.run(debug=True, host='10.128.205.41', port='4000')
    # app.run(debug=True, host=local_host, port=local_port)

    # Attention: some parameters should be set in the following before first run
    local_host = get_host_ip()
    # local_port = 4001
    local_port = args.port
    client_id = args.id
    # client_id = 2
    clients_num = args.clients_num


    # Initialization
    glo.set_global_var("clients_num", clients_num)
    glo.set_global_var("client_id", client_id)
    glo.set_global_var("global_model_path", f"models/global/client-{client_id}/global_model.pkl")
    glo.set_global_var("sub_model_path", f"models/clients/client-{client_id}/sub_model.pkl")
    glo.set_global_var("job_info_path", f"jobs_info/client-{client_id}")
    executor = ThreadPoolExecutor(10)
    dapp_port = int(local_port) + 10000
    dapp_address = f"localhost:{dapp_port}"
    local_address = f"{local_host}:{local_port}"

    # create directory
    path_list = [f"models/global/client-{client_id}",
                 f"models/clients/client-{client_id}",
                 f"models/downloads/client-{client_id}",
                 f"jobs_info/client-{client_id}",
                 f"results/client-{client_id}",
                 f"results/global",
                 f"pics",
                 f"logs"]
    
    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    is_iid = False
    dataset = DataSet(clients_num, is_iid)
    initialize_global_model()

    app.run(host=local_host, port=args.port)

    # multiprocessing
    # client_nums = 5
    # for i in range(client_nums):
    #     p = Process(target=app_run, args=(local_host, str(local_port+i)))
    #     p.start()
    #     p.join()
