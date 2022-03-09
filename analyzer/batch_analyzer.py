import pandas as pd
import globals.global_var as glo
from utils.file_check import check_and_build_dir
from utils.write_file import write_list_to_file
from utils.create_pic import save_compare_pic_from_vector, save_to_histogram_from_list
from utils.plt_config import PltConfig


def batch_analyze_avg_task_processing_time():
    """Compute average task process time of different scheduling algorightm
    """
    idx = 4
    schedulers = ["RoundRobinScheduler", "RandomScheduler", "EarliestScheduler", "HeuristicScheduler",
                  "DQNScheduler", "DDPGScheduler"]
    glo.current_dataset = "Alibaba"

    batch_result_dir = f"../results/task_run_results/{glo.current_dataset}/{schedulers[idx]}/batch_results"
    check_and_build_dir(batch_result_dir)
    output_path = f"../results/task_run_results/{glo.current_dataset}/{schedulers[idx]}/batch_results/" \
                  f"{schedulers[idx]}_batch_results.txt"
    for test_batch_size in range(1000, 11000, 1000):
        data_path = f"../results/task_run_results/{glo.current_dataset}/{schedulers[idx]}/{test_batch_size}/" \
                    f"{schedulers[idx]}_task_run_results.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                        'execute_time', 'process_time']
        transfer_time_mean = data['transfer_time'].mean()
        wait_time_mean = data['wait_time'].mean()
        execute_time_mean = data['execute_time'].mean()
        process_time_mean = data['process_time'].mean()
        print(f"test_batch_size: ", test_batch_size)
        print(f"{schedulers[idx]}_transfer_time_mean: {transfer_time_mean}")
        print(f"{schedulers[idx]}_wait_time_mean: {wait_time_mean}")
        print(f"{schedulers[idx]}_execute_time_mean: {execute_time_mean}")
        print(f"{schedulers[idx]}_process_time_mean: {process_time_mean}")
        print("\n")
        output_list = [transfer_time_mean, wait_time_mean, execute_time_mean, process_time_mean]
        write_list_to_file(output_list, output_path, mode='a+')


def plot_batch_results_avg_task_processing_time():
    schedulers = ["RoundRobinScheduler", "RandomScheduler", "EarliestScheduler", "HeuristicScheduler",
                  "DQNScheduler"]
    labels = ["RR", "Random", "Earliest", "GA", "D3QN"]
    glo.current_dataset = "Alibaba"

    data_vector = []
    for scheduler in schedulers:
        data_path = f"../results/task_run_results/{glo.current_dataset}/{scheduler}/batch_results/" \
                  f"{scheduler}_batch_results.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['transfer_time', 'wait_time', 'exectue_time', 'process_time']
        data['process_time'] /= 1000
        avg_processing_time_data = data['process_time'].tolist()
        data_vector.append(avg_processing_time_data)
    dest_path = f"../pic/task_run_results/batch/average_task_processing_time_comparison_on_{glo.current_dataset}.png"
    plt_config = PltConfig()
    plt_config.title = f"average task processing time on {glo.current_dataset}"
    plt_config.xlabel = "test batch size"
    plt_config.ylabel = "average task processing time"
    plt_config.x_axis_data = [str(i) for i in range(1000, 11000, 1000)]
    save_compare_pic_from_vector(data_vector, labels, dest_path, plt_config, show=True)


def plot_task_batch_results():
    schedulers = ["RoundRobinScheduler", "RandomScheduler", "EarliestScheduler", "HeuristicScheduler",
                  "DDPGScheduler", "DQNScheduler"]
    labels = ["RR", "Random", "Earliest", "GA", "DDPG", "D3QN-OPT"]

    avg_process_time_vector = []
    makespan_vector = []
    avg_work_time_vector = []
    # [200, 300], [200, 250], [300, 350], [350, 400], [450, 500]
    # Alibaba1000000 [370, 420],
    # GoCJ [300, 350],
    slice_start = 4050
    slice_end = 4100
    for scheduler in schedulers:
        data_path = f"../results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler}/task_batches/" \
                    f"task_batches_run_results2.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['avg_process_time', 'makespan', 'avg_work_time']
        avg_processing_time_data = data['avg_process_time'].tolist()[slice_start:slice_end]
        makespan_data = data['makespan'].tolist()[slice_start:slice_end]
        avg_work_time_data = data['avg_work_time'].tolist()[slice_start:slice_end]
        avg_process_time_vector.append(avg_processing_time_data)
        makespan_vector.append(makespan_data)
        avg_work_time_vector.append(avg_work_time_data)

    # 1. task_batch平均任务处理时间
    dest_path = f"../pic/task_run_results/task_batch/{glo.current_dataset}{glo.records_num}/average_task_processing_time_comparison_on_{glo.current_dataset}-2.png"
    plt_config = PltConfig()
    plt_config.title = f"average task processing time comparison on {glo.current_dataset}"
    plt_config.xlabel = "number of task batches"
    plt_config.ylabel = "average task processing time"
    plt_config.x_axis_data = [i for i in range(len(makespan_data))]
    save_compare_pic_from_vector(avg_process_time_vector, labels, dest_path, plt_config, show=True)

    # 2. task_batch makespan
    dest_path = f"../pic/task_run_results/task_batch/{glo.current_dataset}{glo.records_num}/makespan_comparison_on_{glo.current_dataset}-2.png"
    plt_config = PltConfig()
    plt_config.title = f"makespan comparison on {glo.current_dataset}"
    plt_config.xlabel = "number of task batches"
    plt_config.ylabel = "batch makespan"
    plt_config.x_axis_data = [i for i in range(len(makespan_data))]
    save_compare_pic_from_vector(makespan_vector, labels, dest_path, plt_config, show=True)

    # 3. task_batch平均机器工作时间
    dest_path = f"../pic/task_run_results/task_batch/{glo.current_dataset}{glo.records_num}/" \
                f"average_machine_worktime_comparison_on_{glo.current_dataset}.png"
    plt_config = PltConfig()
    plt_config.title = f"average machine worktime on {glo.current_dataset}"
    plt_config.xlabel = "number of task batches"
    plt_config.ylabel = "average machine worktime"
    plt_config.x_axis_data = [i for i in range(len(makespan_data))]
    save_compare_pic_from_vector(avg_work_time_vector, labels, dest_path, plt_config, show=True)


def plot_throughoutput_comparison():
    schedulers = ["RoundRobinScheduler", "RandomScheduler", "EarliestScheduler", "HeuristicScheduler",
                  "DDPGScheduler", "DQNScheduler"]
    labels = ["RR", "Random", "Earliest", "GA", "DDPG", "D3QN-OPT"]

    # 获取总任务数
    data_path = f"../results/task_run_results/{glo.current_dataset}{glo.records_num}/RoundRobinScheduler/" \
                f"RoundRobinScheduler_task_run_results2.txt"
    data = pd.read_csv(data_path, header=None, delimiter='\t')
    task_num = len(data)
    # 计算每个调度算法的单位时间吞吐量
    # 单位时间吞吐量 = 总任务数 / 总任务makespan
    throughoutput_list = []
    for scheduler in schedulers:
        data_path = f"../results/machine_status_results/{glo.current_dataset}{glo.records_num}/{scheduler}/"
        machine_num = 20
        max_makespan = 0
        for machine_idx in range(machine_num):
            tmp_path = data_path + f"{machine_idx}_status2.txt"
            data = pd.read_csv(tmp_path, header=None, delimiter='\t')
            machine_makespan = round(data[-1:][0].tolist()[0], 4)
            max_makespan = max(max_makespan, machine_makespan)
        throughoutput_list.append(max_makespan)
    for i, elem in enumerate(throughoutput_list):
        throughoutput_list[i] = task_num / elem * 1000000
    print(throughoutput_list)
    dest_path = f"../pic/task_run_results/task_batch/{glo.current_dataset}{glo.records_num}/" \
                f"throughoutput_comparison_on_{glo.current_dataset}.png"
    plt_config = PltConfig()
    plt_config.title = f"throughoutput comparison on {glo.current_dataset}"
    plt_config.xlabel = "scheduling algorithms"
    plt_config.ylabel = "number of tasks"
    plt_config.x_axis_data = labels
    save_to_histogram_from_list(throughoutput_list, dest_path, plt_config, show=True, show_text=True)


def plot_average_task_processing_time_comparison():
    schedulers = ["RoundRobinScheduler", "RandomScheduler", "EarliestScheduler", "HeuristicScheduler",
                  "DDPGScheduler", "DQNScheduler"]
    labels = ["RR", "Random", "Earliest", "GA", "DDPG", "D3QN-OPT"]

    # 获取总任务数
    # 计算每个调度算法的单位时间吞吐量
    # 单位时间吞吐量 = 总任务数 / 总任务makespan
    total_avg_task_processing_time_list = []
    for scheduler in schedulers:
        data_path = f"../results/task_run_results/{glo.current_dataset}{glo.records_num}/{scheduler}/" \
                    f"{scheduler}_task_run_results2.txt"
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['task_id', 'task_mi', 'machine_id', 'machine_mips', 'transfer_time', 'wait_time',
                        'execute_time', 'process_time']
        task_num = len(data)
        print("task_num: ", task_num)
        transfer_time_mean = data['transfer_time'].mean()
        wait_time_mean = data['wait_time'].mean()
        execute_time_mean = data['execute_time'].mean()
        process_time_mean = data['process_time'].mean()
        print(f"{scheduler}_transfer_time_mean: {transfer_time_mean}")
        print(f"{scheduler}_wait_time_mean: {wait_time_mean}")
        print(f"{scheduler}_execute_time_mean: {execute_time_mean}")
        print(f"{scheduler}_process_time_mean: {process_time_mean}")
        total_avg_task_processing_time_list.append(process_time_mean)
    for i, elem in enumerate(total_avg_task_processing_time_list):
        total_avg_task_processing_time_list[i] /= 1000
    dest_path = f"../pic/task_run_results/task_batch/{glo.current_dataset}{glo.records_num}/" \
                f"total_average_task_processing_time_comparison_on_{glo.current_dataset}.png"
    plt_config = PltConfig()
    plt_config.title = f"total average task processing time comparison on {glo.current_dataset}"
    plt_config.xlabel = "scheduling algorithms"
    # plt_config.ylabel = "total average task processing time"
    plt_config.ylabel = "total average task processing time (e3)"
    plt_config.x_axis_data = labels
    save_to_histogram_from_list(total_avg_task_processing_time_list, dest_path, plt_config, show=True, show_text=True)


if __name__ == "__main__":
    # glo.current_dataset = "GoCJ"
    # glo.records_num = 2
    glo.current_dataset = "Alibaba"
    glo.records_num = 1000000
    save_dir = f"../pic/task_run_results/task_batch/{glo.current_dataset}{glo.records_num}"
    check_and_build_dir(save_dir)
    # batch_analyze_avg_task_processing_time()
    # plot_batch_results_avg_task_processing_time()
    plot_task_batch_results()
    # plot_throughoutput_comparison()
    # plot_average_task_processing_time_comparison()
