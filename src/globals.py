import os

'''
parameters should be set:
* clients_num:  number of clients participate in the system
* client_id:    client id of current client, and each client has an unique client id

'''

def __init():
    global _global_dict
    _global_dict = {}
    submodel_path = 'models/clients/submodel.npy'
    if os.path.exists(submodel_path):
        _global_dict["has_submodel"] = True
    else:
        _global_dict["has_submodel"] = False
    _global_dict["train_status"] = "todo"       # train_status has 3 status { "todo", "training", "finished" }
    _global_dict["merge_status"] = "todo"       # merge_status has 3 status { "todo", "merging", "finished" }
    _global_dict["download_status"] = "todo"        # download_status has 3 status { "todo", "downloading", "finished" }
    _global_dict["clients_num"] = 5
    _global_dict["merge_clients_num"] = 5
    _global_dict["client_id"] = 0


def set_global_var(key, value):
    _global_dict[key] = value


def get_global_var(key):
    try:
        return _global_dict[key]
    except KeyError:
        print("Global dict has no key named %s!".format(key))