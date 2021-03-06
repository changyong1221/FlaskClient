import math
import numpy as np
from scheduler.scheduler import Scheduler
from model.dqn.dqn import DQN
from utils.state_representation import get_state
from src_scheduling.log import print_log


class DQNScheduler(Scheduler):
    def __init__(self, multidomain_id, machine_num, task_batch_num, machine_kind_num_list, machine_kind_idx_range_list,
                 balance_prob=0.5):
        """Initialization

        input : a list of tasks
        output: scheduling results, which is a list of machine id
        """
        self.task_dim = 3
        self.machine_dim = 2

        self.state_all = []  # 存储所有的状态 [None,2+2*20]
        self.action_all = []  # 存储所有的动作 [None,1]
        self.reward_all = []  # 存储所有的奖励 [None,1]
        self.machine_kind_idx_range_list = machine_kind_idx_range_list

        self.double_dqn = True
        self.dueling_dqn = True
        self.optimized_dqn = False
        self.prioritized_memory = False
        self.DRL = DQN(multidomain_id, self.task_dim, machine_num, self.machine_dim, machine_kind_num_list,
                       self.machine_kind_idx_range_list,
                       self.double_dqn, self.dueling_dqn, self.optimized_dqn, self.prioritized_memory, balance_prob=balance_prob)
        self.DRL.max_step = task_batch_num
        self.cur_step = 0
        self.alpha = 0.5
        self.beta = 0.5
        self.C = 10
        print_log("DQN网络初始化成功！")

    def schedule(self, task_instance_batch, machine_list):
        # print_log("enter schedule")
        task_num = len(task_instance_batch)

        states = get_state(task_instance_batch, machine_list)
        self.state_all += states
        # self.state_all.append(states)
        machines_id = self.DRL.choose_action(np.array(states))  # 通过调度算法得到分配 id
        machines_id = machines_id.astype(int).tolist()
        return machines_id
        # if (step == 1): print_log("machines_id: " + str(machines_id))

    def learn(self, task_instance_batch, machines_id, makespan):
        # print_log(f"enter learn...")
        for idx, task in enumerate(task_instance_batch):  # 便历新提交的一批任务，记录动作和奖励
            self.action_all.append([machines_id[idx]])
            # if (task.get_task_mi() < 3000 and machines_id[idx] < 6) or \
            #         (task.get_task_mi() < 8000 and machines_id[idx] < 12) or \
            #         (task.get_task_mi() >= 8000 and machines_id[idx] < 19):
            #     reward = 1
            # if (task.get_task_mi() >= 525000 and machines_id[idx] >= 17) or \
            #     (task.get_task_mi() >= 150000 and machines_id[idx] >= 13) or \
            #         (task.get_task_mi() >= 101000 and machines_id[idx] >= 9) or \
            #         (task.get_task_mi() >= 59000 and machines_id[idx] >= 4) or \
            #         (task.get_task_mi() >= 15000 and machines_id[idx] >= 0):
            #     reward = 1
            # else:
            #     reward = 0

            # print_log(f"task.get_task_processing_time(): {task.get_task_processing_time()}")
            # print_log(f"makespan: {makespan}")
            task_item = 2 if task.get_task_processing_time() <= 2 else task.get_task_processing_time()
            makespan_item = 2 if makespan <= 2 else makespan
            reward = self.C / (self.alpha * math.log(task_item, 10) +
                               self.beta * math.log(makespan_item, 10))
            # reward = task.get_task_mi() / task.get_task_processing_time() / 100
            # if task.get_task_mi() > 100000 and machines_id[idx] > 15:
            #     reward = 100
            # print("machine_id: ", machines_id[idx])
            # print("task_mi: ", task.get_task_mi())
            # print("task_processing_time: ", task.get_task_processing_time())
            # print("reward: ", reward)
            self.reward_all.append([reward])  # 计算奖励
        # 减少存储数据量
        if len(self.state_all) > 20000:
            self.state_all = self.state_all[-10000:]
            self.action_all = self.action_all[-10000:]
            self.reward_all = self.reward_all[-10000:]
        # 如果使用prioritized memory
        if self.prioritized_memory:
            for i in range(len(task_instance_batch)):
                self.DRL.append_sample([self.state_all[-2 + i]], [self.action_all[-1 + i]],
                                  [self.reward_all[-1 + i]], [self.state_all[-1 + i]])

        # 先学习一些经验，再学习
        print_log(f"cur_step: {self.cur_step}")
        if self.cur_step > 2:
            # 截取最后10000条记录
            # print_log(type(self.state_all))
            # print_log(self.state_all)
            # array = np.array(self.state_all)
            # print_log(array)
            # print_log(type(array))
            new_state = np.array(self.state_all, dtype=np.float32)[-10000:-1]
            new_action = np.array(self.action_all, dtype=np.float32)[-10000:-1]
            new_reward = np.array(self.reward_all, dtype=np.float32)[-10000:-1]
            self.DRL.store_memory(new_state, new_action, new_reward)
            self.DRL.step = self.cur_step
            loss = self.DRL.learn()
            print_log(f"step: {self.cur_step}, loss: {loss}")
        self.cur_step += 1
