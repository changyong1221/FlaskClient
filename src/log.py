import src.globals as glo

def printLog(strs):
    client_id = glo.get_global_var("client_id")
    log_str = f"client-{client_id}: {strs}"
    print(log_str)
    saveLog(log_str)


def saveLog(strs):
    file_name = glo.get_global_var("log_path")
    with open(file_name, 'a+') as f:
        f.write(strs + "\n")