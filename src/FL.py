import json
import threading
from flask import Blueprint, request, Response
from src.MergeModels import MergeModels

interface = Blueprint("interface", __name__)


def load_job(data):
    f = open('interface/job.json', 'r+')
    job = json.load(f)
    f.close()
    return job


# 作为委员会节点，请对当前模型列表做出评分后合并成为一个新模型，并将新模型打分
# 数据通过json格式传输
# 提供1个参数
# 1.models，是一个列表，内含多个子模型swarmid
# 返回4个参数
# 1.models,是一个列表,内含多个子模型
# 2.socres,是一个列表,内含多个子模型打分,必须与参数1对齐
# 3.fmodel,是一个字符串,代表了合并之后的swarmid
# 4.fscore,是一个三位整数,代表了参数3的打分,例如准确率为89.6%，那么fscore=896
@interface.route('/infoWork', methods=["POST"])
def WorkEvent():
    data = request.json
    res_data = {"state": "ok"}
    job = load_job()
    if job["hasdone"] == 1:
        # try:
        res_data.update(job)
        models = data['models']
        t = threading.Thread(target=MergeModels, args=(models,))
        t.start()
        # except:
        #     res_data["state"] = "error"
    else:
        res_data["state"] = "wt"
    return Response(json.dumps(res_data),  mimetype='application/json')




