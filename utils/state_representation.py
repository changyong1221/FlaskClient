import sys
import math
import numpy as np
import globals.global_var as glo


# 通过任务和机器获取状态
def get_state(task_list, machine_list):
    commit_time = task_list[0].commit_time  # 当前批次任务的开始时间
    machines_state = []
    for machine in machine_list:
        machines_state.append(machine.get_mips())
        machines_state.append(max(machine.get_finish_time() - commit_time, 0))  # 等待时间
        # if (machine.next_start_time - start_time > 0):
        #     print_log("machines_state: ", machines_state)
    # print("machines_state: ", machines_state)
    tasks_state = []
    for i, task in enumerate(task_list):
        task_state = []
        task_state.append(task.get_task_mi())
        task_state.append(task.get_task_cpu_utilization())
        task_state.append(task.get_task_mi() / machine_list[0].get_bandwidth())  # 传输时间
        task_state += machines_state  # 由于是DQN，所以一个任务状态加上多个虚拟机状态
        tasks_state.append(task_state)
    # 返回值 [[[153.0, 0.79, 0.34, 600, 0, 600, 0, 500, 0, 500, 0, 400, 0, 400, 0, 300, 0, 300, 0, 200, 0, 200, 0]... ]]
        #           任务长度，任务利用率，任务传输时间，vm1_mips, vm1_waitTime, vm2....
    # print("tasks_state: ", tasks_state)
    return tasks_state



# 精简状态
# def get_state(task_list, machine_list):
#     tasks_state = []
#     for task in task_list:
#         task_state = []
#         def get_task_mi_kind(task_mi):
#             if task_mi >= 525000:
#                 return 5
#             elif task_mi >= 150000:
#                 return 4
#             elif task_mi >= 101000:
#                 return 3
#             elif task_mi >= 59000:
#                 return 2
#             elif task_mi >= 15000:
#                 return 1
#         # task_state.append(get_task_mi_kind(task.get_task_mi()))
#         task_state.append(task.get_task_mi())
#         tasks_state.append(task_state)
#     return tasks_state


# 通过任务和机器获取状态
# 状态定义：[任务利用率，vm1_utilization, vm2_utilization, ...]
def get_ddpg_state(task_list, machine_list):
    commit_time = task_list[0].commit_time  # 当前批次任务的开始时间
    machines_state = []
    for machine in machine_list:
        machines_state.append(machine.get_mips())
        machines_state.append(max(machine.get_finish_time() - commit_time, 0))  # 等待时间
        # if (machine.next_start_time - start_time > 0):
        #     print_log("machines_state: ", machines_state)
    # print("machines_state: ", machines_state)
    tasks_state = []
    for i, task in enumerate(task_list):
        task_state = []
        task_state.append(task.get_task_mi())
        task_state.append(task.get_task_cpu_utilization())
        task_state.append(task.get_task_mi() / machine_list[0].get_bandwidth())  # 传输时间
        task_state += machines_state  # 由于是DQN，所以一个任务状态加上多个虚拟机状态
        tasks_state.append(task_state)
    # 返回值 [[[153.0, 0.79, 0.34, 600, 0, 600, 0, 500, 0, 500, 0, 400, 0, 400, 0, 300, 0, 300, 0, 200, 0, 200, 0]... ]]
        #           任务长度，任务利用率，任务传输时间，vm1_mips, vm1_waitTime, vm2....
    # print("tasks_state: ", tasks_state)
    return tasks_state


# 根据机器性能分配任务数
def get_vm_tasks_capacity(machine_list):
    # 定义数组用于存储最终结果
    vm_tasks_capacity = []
    # 所有机器的总mips
    total_mips = 0
    for machine in machine_list:
        vm_tasks_capacity.append(0)
        total_mips += machine.mips
    # print("total_mips: ", total_mips)
    # 计算每个机器mips占总mips的比例
    for i, machine in enumerate(machine_list):
        vm_tasks_capacity[i] = math.ceil(((float)(machine.mips) / total_mips) * glo.records_num)
    # print("vm_tasks_capacity: ", vm_tasks_capacity)
    return vm_tasks_capacity


# 根据机器性能设置机器性能阶梯表
def get_machine_kind_list(machine_list):
    """定义五种性能种类：
    very low [0, 800),
    low      [800, 2000),
    common   [2000, 6000),
    high     [6000, 12000),
    very high [12000, 24000),
    extremely high [24000, +oo)

    """
    # 定义数组用于存储最终结果
    num_of_machine_kind = 6
    machine_kind_range_list = [24000, 12000, 6000, 2000, 800, 0]
    machine_kind_num_list = [0 for i in range(num_of_machine_kind)]      # 记录每类机器的数目
    machine_kind_idx_range_list = []    # 记录每类机器的索引范围
    # 机器性能阶梯表
    for machine in machine_list:
        for i, left_border in enumerate(machine_kind_range_list):
            if machine.get_mips() >= left_border:
                machine_kind_num_list[num_of_machine_kind - i - 1] += 1
                break
    start_idx = 0
    for i in range(num_of_machine_kind):
        machine_kind_idx_range_list.append((start_idx, start_idx + machine_kind_num_list[i]))
        start_idx = start_idx + machine_kind_num_list[i]
    return machine_kind_num_list, machine_kind_idx_range_list


# 根据机器id获取机器所属的种类索引
def get_machine_kind_idx(machine_id, num_of_machine_kind, machine_kind_idx_range_list):
    for i in range(num_of_machine_kind):
        if machine_kind_idx_range_list[i][0] <= machine_id < machine_kind_idx_range_list[i][1]:
            return i
    return num_of_machine_kind - 1


# 阶梯平衡因子
def balancing(machine_kind_avg_task_map, num_of_machine_kind, machine_kind_idx_range_list, balance_factor_min,
              balance_factor_max):
    # 先进行一轮分配，使每类机器的平均任务数大于等于1
    for i in range(num_of_machine_kind):
        kind_idx = num_of_machine_kind - i - 1
        if machine_kind_avg_task_map[kind_idx] < 1:
            return np.random.randint(machine_kind_idx_range_list[kind_idx][0], machine_kind_idx_range_list[kind_idx][1])
    for i in range(num_of_machine_kind - 1):
        kind_idx = num_of_machine_kind - i - 2
        if machine_kind_avg_task_map[kind_idx] / machine_kind_avg_task_map[kind_idx + 1] < balance_factor_min:
            return np.random.randint(machine_kind_idx_range_list[kind_idx][0], machine_kind_idx_range_list[kind_idx][1])
        elif machine_kind_avg_task_map[kind_idx] / machine_kind_avg_task_map[kind_idx + 1] > balance_factor_max:
            return np.random.randint(machine_kind_idx_range_list[kind_idx + 1][0],
                                     machine_kind_idx_range_list[kind_idx + 1][1])
    return np.random.randint(machine_kind_idx_range_list[-1][0], machine_kind_idx_range_list[-1][1])


# 判断是否负载均衡
def is_balanced():
    pass


# 使用任务亲和度优化
def task_adapting(state, num_of_machine_kind, machine_kind_idx_range_list):
    task_mi = state[0]
    task_mi_list = [150000, 8000, 4000, 2000, 1000, 0]
    # task_mi_list = [525000, 150000, 101000, 59000, 15000]
    for i, task_mi_border in enumerate(task_mi_list):
        if task_mi >= task_mi_border:
            kind_idx = num_of_machine_kind - i - 1
            return np.random.randint(machine_kind_idx_range_list[kind_idx][0],
                                     machine_kind_idx_range_list[kind_idx][1])
    print("task_mi: ", task_mi)
