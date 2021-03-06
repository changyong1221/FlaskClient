import os

'''
parameters should be set:
* clients_num:  number of clients participate in the system
* client_id:    client id of current client, and each client has an unique client id

'''

def __init():
    global _global_dict
    _global_dict = {}
    _global_dict["train_status"] = "todo"       # train_status has 3 status { "todo", "training", "finished" }
    _global_dict["merge_status"] = "todo"       # merge_status has 3 status { "todo", "merging", "finished" }
    _global_dict["update_status"] = "todo"      # update_status has 2 status { "todo", "updating" }
    _global_dict["clients_num"] = 0
    _global_dict["merge_clients_num"] = 0
    _global_dict["merge_clients_id_list"] = []
    _global_dict["client_id"] = 0
    _global_dict["global_model_path"] = ""
    _global_dict["sub_model_path"] = ""
    _global_dict["job_info_path"] = ""
    _global_dict["current_scheduler"] = "DQNScheduler"
    _global_dict["current_round"] = 0
    _global_dict["all_rounds"] = 0
    _global_dict["clients_merge_rounds_list"] = []
    _global_dict["current_data_scale"] = 0
    _global_dict["clients_data_scale_list"] = []
    _global_dict["is_client_test"] = False
    _global_dict["is_global_test"] = False
    _global_dict["privacy_level"] = 16


def set_global_var(key, value):
    _global_dict[key] = value


def get_global_var(key):
    try:
        return _global_dict[key]
    except KeyError:
        print(f"Global dict has no key named {key}!")