import numpy as np
from scheduler.scheduler import Scheduler


class RandomScheduler(Scheduler):
    def __init__(self, machine_num):
        """Initialization
        """
        self.machine_num = machine_num

    def schedule(self, task_num):
        """Schedule tasks in random way

        @:return scheduling results, which is a list of machine id
        """
        schedule_results = []
        for task in range(task_num):
            schedule_results.append(np.random.randint(0, self.machine_num))
        return schedule_results
