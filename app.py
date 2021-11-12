from flask import Flask, render_template, request, Response
from src.model_funcs import train_one_model, merge_models_and_test, has_submodel, update_model
import requests
import os, json
import time
import src.globals as glo
from concurrent.futures import ThreadPoolExecutor

# Attention: some parameters should be set in the following before first run
local_host = "127.0.0.1"
local_port = "4000"
swarm_server_host = "10.112.58.204"
swarm_server_port = "40000"
clients_num = 5
client_id = 0

# Initialization
glo.__init()
glo.set_global_var("clients_num", clients_num)
glo.set_global_var("client_id", client_id)
executor = ThreadPoolExecutor(5)
swarm_server = f"{swarm_server_host}:{swarm_server_port}"
dapp_port = int(local_port) + 1000
dapp_address = f"localhost:{dapp_port}"
local_address = f"{local_host}:{local_port}"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcdefg'

@app.route("/")
def root():
    """
    主页
    :return: Index.html
    """
    return render_template('Index.html')


@app.route("/train", methods=['GET'])
def train():
    if glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training"})
    else:
        executor.submit(train_one_model)
        return json.dumps({"status": "start training..."})


@app.route("/infoSubmit", methods=['GET'])
def submit_submodel():
    money = request.json["money"]
    print(f"money: {money}")
    submodel_path = 'models/clients/sub_model.npy'
    if glo.get_global_var("train_status") == "training":
        return json.dumps({"status": "training"})
    if not has_submodel():
        train_one_model()
    data = {'file': open(submodel_path, 'rb')}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=data)

    # print(response.text)
    ret = {"swarm_id": response.json()["swarm_id"]}
    return ret


@app.route("/infoWork", methods=['GET'])
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
        f = open("job.json", 'r+')
        job = json.load(f)
        glo.set_global_var("merge_status", "todo")
        return job


def merge_models_work(model_ids):
    glo.set_global_var("merge_status", "merging")
    glo.set_global_var("merge_clients_num", len(model_ids))
    # get all submodels from swarm
    print("start downloading submodels.")
    while glo.get_global_var("download_status") == "downloading":
        time.sleep(3)
    glo.set_global_var("download_status", "downloading")
    for idx, swarm_id in enumerate(model_ids):
        data = {"swarm_id": swarm_id, "client_id": idx, "is_global": False, "client_address": local_address}
        print(data)
        response = requests.post(f'http://{swarm_server}/download_from_swarm', json=data)

    # merge models. merge process won't start before all submodels have been downloaded
    overtime_counter = 0
    while glo.get_global_var("download_status") != "finished":
        print("waiting downloading")
        time.sleep(3)   # sleep three second
        overtime_counter += 1
        if overtime_counter == 100:
            return json.dumps({'merge': 'download overtime', 'status': 'failure'})
    glo.set_global_var("download_status", "todo")
    print("submodels downloading finished.")
    print("merge start.")
    scores = merge_models_and_test()
    print("merge finished.")

    # upload global model
    print("start uploading global model.")
    file = {'file': open('models/global/global_model.npy', 'rb')}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=file)

    # stop training when global model score reached 900
    if scores["global_score"] > 900:
        is_stop = True
    else:
        is_stop = False
    ret = {"models": model_ids, "scores": scores["clients_scores"], "fscore": scores["global_score"],
           "fmodel": response.json()['swarm_id'], "stop": is_stop}
    print(ret)
    job_name = "job.json"
    with open(job_name, 'w+') as f:
        json.dump(ret, f)

    print("global model uploading finished.")
    print("all tasks have been done.")
    glo.set_global_var("merge_status", "finished")


# has similar function as "/infoSubmit"
@app.route("/upload")
def upload_model():
    data = {'file': open('models/clients/submodel.npy', 'rb')}
    response = requests.post(f'http://{swarm_server}/upload_to_swarm', files=data)

    # print(response.text)
    return json.dumps({'status': 'success'})


def download_update_global_model(paras):
    # swarm_id = paras
    swarm_id = paras["model"]
    score = paras['score']
    is_stop = paras['stop']

    data = {"swarm_id": swarm_id, "client_id": client_id, "is_global": True, "client_address": local_address}
    print(data)
    while glo.get_global_var("download_status") == "downloading":
        time.sleep(3)
    glo.set_global_var("download_status", "downloading")
    print("downloading global model...")
    response = requests.post(f'http://{swarm_server}/download_from_swarm', json=data)
    print("global model downloaded.")
    update_model()
    glo.set_global_var("update_status", "todo")


# receive model file from swarm and save file to localhost
@app.route("/receive", methods=['POST'])
def receive():
    is_global_str = request.values['is_global']
    if is_global_str == "True":
        print("true")
        is_global = True
    else:
        print('false')
        is_global = False
    client_id = int(request.values['client_id'])

    ff = request.files['file']
    if is_global:
        save_path = 'models/global/'
        filename = 'global_model.npy'
        ff.save(os.path.join(save_path, filename))
        glo.set_global_var("download_status", "todo")
    else:
        save_path = 'models/downloads/'
        filename = f'{client_id}.npy'
        ff.save(os.path.join(save_path, filename))
        print(f"client({client_id}) downloaded.")
        last_client_id = glo.get_global_var("merge_clients_num") - 1
        if client_id is last_client_id:
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
    print(response.json())
    executor.submit(download_update_global_model, (response.json()))
    # executor.submit(download_update_global_model, (swarm_id))
    return Response(json.dumps({'status': 'success'}), mimetype="application/json")


if __name__ == '__main__':
    # app.run(debug=True, host='10.128.205.41', port='4000')
    app.run(debug=True, host=local_host, port=local_port)