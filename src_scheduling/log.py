import src_scheduling.globals as glo
from src_scheduling.utils import check_and_build_dir
import time


def print_log(strs):
    client_id = glo.get_global_var("client_id")
    cur_date = time.strftime("%Y-%m-%d", time.localtime())
    log_path = f"logs/client-{client_id}/client-{client_id}-{cur_date}.log"
    cur_time = time.strftime("%H:%M:%S", time.localtime())
    log_str = f"[client-{client_id} {cur_time}]: {strs}"
    print(log_str)
    save_log(log_str, log_path)


def save_log(strs, log_path):
    with open(log_path, 'a+') as f:
        f.write(strs + "\n")
        f.close()


if __name__ == "__main__":
    pass