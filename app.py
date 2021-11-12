from flask import Flask, render_template, request
from src.model_funcs import train_one_model, merge_models_and_test
import requests
import os, json
import time
import src.globals as glo
from concurrent.futures import ThreadPoolExecutor

# Attention: some parameters should be set in globals.py before first run

# executor = ThreadPoolExecutor(5)
glo.__init()
client_id = glo.get_global_var("client_id")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcdefg'

@app.route("/")
def root():
    """
    主页
    :return: Index.html
    """
    return render_template('Index.html')


@app.route("/train", methods=['POST'])
def train():
    print("hahaha")
    train_one_model()
    return json.dumps({"status": "success"})


@app.route("/infoSubmit", methods=['POST'])
def submit_submodel():
    money = request.json["money"]
    print(f"money: {money}")
    submodel_path = 'models/clients/sub_model.npy'
    if glo.get_global_var("train_staus") == "training":
        return json.dumps({"status": "training"})
    if not glo.get_global_var("has_submodel"):
        train_one_model()
    data = {'file': open(submodel_path, 'rb')}
    response = requests.post('http://10.112.58.204:40000/upload_to_swarm', files=data)

    # print(response.text)
    ret = {"swarm_id": response.json()["swarm_id"]}
    return ret


@app.route("/infoWork", methods=['POST'])
def merge_models():
    if glo.get_global_var("merge_status") == "todo":
        return json.dumps({"status": "request received"})
    elif glo.get_global_var("merge_status") == "merging":
        return json.dumps({"status": "merging"})
    glo.set_global_var("merge_status", "merging")
    model_ids = request.json["models"]
    # model_ids = ["10a1b887db6eb16f1c5e73da51c6b645dd0cd0b4bd17c183bdd205e502e0c29e",
    #              "01d8de78a6e0a975b5f3eed94600d2d92577348cec226776438de507b07eece1",
    #              "441707c3e702a4d61d62122ace5c6712c060d8d0fa7868108c2a9a96678b12aa",
    #              "8ed728a48823854fd87a10265e7364a377be31869d4e211523348851a2e7ac9b",
    #              "9ca9fa2d8b5b14ed64054f9c525200cde39ef7673ab3e8d438d610fba719ea04"]
    glo.set_global_var("merge_clients_num", len(model_ids))
    # get all submodels from swarm
    print("start downloading submodels.")
    glo.set_global_var("download_status", "todo")
    for idx, swarm_id in enumerate(model_ids):
        data = {"swarm_id": swarm_id, "client_id": idx, "is_global": False}
        print(data)
        response = requests.post('http://10.112.58.204:40000/download_from_swarm', json=data)
        # executor.submit(receive)

    # merge models. merge process won't start before all submodels have been downloaded
    overtime_counter = 0
    while glo.get_global_var("download_status") is not "finished":
        print("waiting downloading")
        time.sleep(3)   # sleep three second
        overtime_counter += 1
        if overtime_counter == 100:
            return json.dumps({'merge': 'download overtime', 'status': 'failure'})
    print("submodels downloading finished.")
    print("merge start.")
    scores = merge_models_and_test()
    print("merge finished.")

    # upload global model
    print("start uploading global model.")
    file = {'file': open('models/global/global_model.npy', 'rb')}
    response = requests.post('http://10.112.58.204:40000/upload_to_swarm', files=file)

    ret = {"models": model_ids, "scores": scores["clients_scores"], "fscore": scores["global_score"],
           "fmodel": response.json()['swarm_id'], "stop": False}
    print(ret)

    print("global model uploading finished.")
    print("all tasks have been done.")
    glo.set_global_var("merge_status", "todo")
    return json.dumps(ret)


# has similar function as "/infoSubmit"
@app.route("/upload")
def upload_model():
    data = {'file': open('models/clients/submodel.npy', 'rb')}
    response = requests.post('http://10.112.58.204:40000/upload_to_swarm', files=data)

    # print(response.text)
    return json.dumps({'status': 'success'})


# has similar function as "/infoWork"
@app.route("/download")
def download_model():
    swarm_id = "e62e990f3d7eded160af257f296ae32e8eea56aa394ba08c7b322324a0dc0289"   # client0
    #swarm_id = "e06920af76d6bc931a6ad047f91993825f012b9c98306bac578cb439d85f6ebf"
    data = {"swarm_id": swarm_id}
    print(data)
    response = requests.post('http://10.112.58.204:40000/download_from_swarm', json=data)
    return json.dumps({'status': 'success'})


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
    else:
        glo.set_global_var("download_status", "downloading")
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
    swarm_id = "739b52501f986cb8a4de782ca411cc643b3c481416f4e173c2d02bf859bee966"
    data = {"swarm_id": swarm_id, "client_id": -1, "is_global": True}
    print(data)
    response = requests.post('http://10.112.58.204:40000/download_from_swarm', json=data)

    return json.dumps({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='4000')