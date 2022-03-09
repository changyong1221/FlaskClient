

"""Global Variables
"""

# file path to save task run results
results_path_list = {
    "RoundRobinScheduler": "results/task_run_results/RoundRobinScheduler/RoundRobinScheduler_task_run_results.txt",
    "RandomScheduler": "results/task_run_results/RandomScheduler/RandomScheduler_task_run_results.txt",
    "EarliestScheduler": "results/task_run_results/EarliestScheduler/EarliestScheduler_task_run_results.txt",
    "HeuristicScheduler": "results/task_run_results/HeuristicScheduler/HeuristicScheduler_task_run_results.txt",
    # "DQNScheduler": "results/task_run_results/DQNScheduler/DQNScheduler_task_run_results.txt",
    "DQNScheduler": "results/task_run_results/client-0/DQNScheduler_task_run_results_test.txt",
    "DDPGScheduler": "results/task_run_results/DDPGScheduler/DDPGScheduler_task_run_results.txt"
}
task_run_results_path = None

# dataset path
current_dataset = "None"
current_batch_size = 0
machine_file_path = "dataset/create/machine.txt"
# task_file_path = "dataset/Alibaba/Alibaba-Cluster-trace-2000.txt"
# task_file_path = "dataset/GoCJ/GoCJ_Dataset_20000_test.txt"

# global location
location_longitude = 0
location_latitude = 0
line_transmit_speed = 280000000

geographical_location_list = {
    "北京市": [116.41, 39.91],
    "上海市": [121.48, 31.24],
    "深圳市": [114.06, 22.55],
    "莫斯科": [113.85, 23.12],
    "新加坡市": [116.36, 39.95],
    "吉隆坡": [104.62, 28.96]
}


# current scheduler name
current_scheduler = ""

# federated learning settings
is_federated = False
is_test = False
federated_round = 0
records_num = 100000

# log control
is_print_log = True

