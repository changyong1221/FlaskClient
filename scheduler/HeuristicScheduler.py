import sys
import numpy as np
from scheduler.scheduler import Scheduler


class HeuristicScheduler(Scheduler):
    def __init__(self, machine_list):
        """Initialization, using Genetic algorithm
        """
        # GA调度算法的参数
        self.popsize = 10        # 种群大小
        self.gmax = 100          # 迭代次数
        self.crossover_prob = 0.8    # 交叉概率
        self.mutation_rate = 0.01    # 变异概率
        self.machine_list = machine_list
        self.task_list = []

    def schedule(self, task_instance_batch):
        self.task_list.clear()
        self.task_list = task_instance_batch
        pop = self.init_pops_randomly(len(self.task_list), len(self.machine_list))
        final_pop = self.GA_main(pop)
        final_schedule = self.find_best_schedule(final_pop)
        return final_schedule

    def init_pops_randomly(self, task_num, machine_num):
        """随机初始化种群
        """
        schedules = []
        for i in range(self.popsize):
            schedule = []
            for j in range(task_num):
                schedule.append(np.random.randint(0, machine_num))
            schedules.append(schedule)
        return schedules

    def find_best_schedule(self, pop):
        """从最终种群中找到最优的调度策略
        """
        best_fitness = sys.maxsize
        best_idx = 0
        for i in range(self.popsize):
            fitness = self.get_fitness(pop[i])
            if fitness < best_fitness:
                best_fitness = fitness
                best_idx = i
        return pop[best_idx]

    def GA_main(self, pop):
        """GA算法核心
        """
        segment_foreach = self.calc_selection_probs(pop)
        children = []       # 一轮遗传算法产生的子代
        temp_parents = []   # 临时对象，是对parents的拷贝
        while len(children) < self.popsize:
            # 1. 选择阶段，选择两个亲代
            for i in range(2):
                prob = np.random.random()
                for j in range(self.popsize):
                    if self.is_between(prob, segment_foreach[j]):
                        temp_parents.append(pop[j])
                        break
            # 2. 染色体交叉阶段
            p1 = temp_parents[0].copy()
            p1temp = temp_parents[0].copy()
            p2 = temp_parents[1].copy()
            p2temp = temp_parents[1].copy()
            if np.random.random() < self.crossover_prob:
                task_num = len(pop[0])
                cross_position = np.random.randint(0, task_num)
                # 交叉操作
                for i in range(cross_position + 1, task_num):
                    temp = p1temp[i]
                    p1temp[i] = p2temp[i]
                    p2temp[i] = temp
            # 在下一轮迭代中，如果子代更优秀则保留子代，否则保留亲代
            children.append(p1temp if self.get_fitness(p1temp) < self.get_fitness(p1) else p1)
            children.append(p2temp if self.get_fitness(p2temp) < self.get_fitness(p2) else p2)
            # 3. 变异阶段
            if np.random.random() < self.mutation_rate:
                maxidx = len(children) - 1
                self.operate_mutation(children[maxidx])
        self.gmax -= 1
        return self.GA_main(children) if self.gmax > 0 else children

    def operate_mutation(self, child):
        """在某个个体上变异
        """
        mutation_idx = np.random.randint(0, len(self.task_list))
        new_machine_idx = np.random.randint(0, len(self.machine_list))
        while (child[mutation_idx] == new_machine_idx):
            new_machine_idx = np.random.randint(0, len(self.machine_list))
        child[mutation_idx] = new_machine_idx

    def calc_selection_probs(self, pop):
        """计算选择概率
        """
        total_fitness = 0
        fits = []
        probs = {}      # probs保存每个种群的适应度占总适应度的占比
        for i in range(self.popsize):
            fitness = self.get_fitness(pop[i])
            fits.append(fitness)
            total_fitness += fitness
        for i in range(self.popsize):
            probs[i] = fits[i] / total_fitness
        segment_foreach = self.get_segments(probs)
        return segment_foreach

    def get_segments(self, probs):
        """计算概率片段
        """
        prob_segments = {}      # prob_segments保存每个个体选择概率的起点、终点，以便选择作为交配元素
        start = 0
        end = 0
        for i in range(self.popsize):
            end = start + probs[i]
            segment = [start, end]
            prob_segments[i] = segment
            start = end
        return prob_segments

    def is_between(self, prob, segment):
        """判断某个(0,1)范围内的浮点数是否介于segment表示的范围内
        """
        if segment[0] <= prob <= segment[1]:
            return True
        return False

    def get_fitness(self, schedule):
        """计算个体适应度，依据makespan计算适应度
        """
        fitness = 0
        machine_assigned_tasks = {}
        task_num = len(self.task_list)
        for i in range(task_num):
            if machine_assigned_tasks.get(schedule[i]) is None:
                task_list = []
                task_list.append(i)
                machine_assigned_tasks[schedule[i]] = task_list
            else:
                machine_assigned_tasks[schedule[i]].append(i)
        # 计算makespan
        for machine_idx in machine_assigned_tasks.keys():
            length = 0
            for task_idx in machine_assigned_tasks[machine_idx]:
                length += self.task_list[task_idx].get_task_mi()
            runtime = length / self.machine_list[machine_idx].get_mips()
            if runtime > fitness:
                fitness = runtime
        return fitness
