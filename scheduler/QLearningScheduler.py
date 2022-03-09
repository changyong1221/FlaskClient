from scheduler.scheduler import Scheduler


class DQNScheduler(Scheduler):
    def __init__(self, machine_list):
        """Initialization
        """
        self.epoch = 200  # 迭代次数
        self.gamma = 0.9  # 为未来reward的衰减值, 0~1之间, 0 looks in the near future, 1 looks in the distant future
        self.epsilon = 0.1  # Epsilon greedy 是用在决策上的一种策略, 比如 epsilon = 0.9 时, 就说明有90% 的情况我会按照 Q 表的最优值选择行为, 10% 的时间使用随机选行为.
        self.alpha = 0.1  # 学习率，0~1之间，来决定这次的误差有多少是要被学习的
        self.machine_list = machine_list
        self.task_list = []

    def schedule(self, task_instance_batch):
        self.task_list.clear()
        self.task_list = task_instance_batch
        Q = {}  # Q表
        task_num = len(task_instance_batch)  # 任务数目
        machine_num = len(self.machine_list)  # 机器数目
        action_num = machine_num  # 动作空间大小

        """
        一个状态是一个整型数组，如（2,1,3,4)，数组索引代表任务id，索引对应值代表分配给第几号机器
        所以对于5个机器，20个任务的例子，状态空间的大小是5^20
        动作分为单步动作和整步动作，单步动作只把一个任务分配给一个机器，整步动作把所有任务分配给机器
        奖励函数是计算所有任务全部执行完成的花费时间
        Q表用一个HashMap来实现，Key是状态，Value是一个double数组，数组大小即动作空间大小，每个值对应
        在该状态下采取该动作的Q值
        所有任务全部执行完成的花费时间为每个机器执行完各自任务的终止时间的最大值，即makespan
        初始状态所有机器的任务完成时间设置为无穷大，即Double.MAX_VALUE
        """
        pass
