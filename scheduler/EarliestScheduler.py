import numpy as np
from scheduler.scheduler import Scheduler


class EarliestScheduler(Scheduler):
    def __init__(self):
        """Initialization
        """
        pass

    def schedule(self, task_num, machine_num, idle_machine_list):
        """Schedule using the earliest idle machines
        """
        schedule_results = []
        idle_machine_id_list = []
        idle_machine_num = len(idle_machine_list)
        for idle_machine in idle_machine_list:
            idle_machine_id_list.append(idle_machine.get_machine_id())
        idle_machine_id_list = np.array(idle_machine_id_list)
        np.random.shuffle(idle_machine_id_list)
        for idx in range(task_num):
            if idle_machine_num > 0:
                schedule_results.append(idle_machine_id_list[idx])
                idle_machine_num -= 1
            else:
                schedule_results.append(np.random.randint(0, machine_num))
        return schedule_results
