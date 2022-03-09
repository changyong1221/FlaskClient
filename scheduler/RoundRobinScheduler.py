from scheduler.scheduler import Scheduler


class RoundRobinScheduler(Scheduler):
    def __init__(self, machine_num):
        """Initialization
        """
        self.machine_num = machine_num
        self.cur_machine_id = 0

    def schedule(self, task_num):
        """Schedule tasks in round-robin way

        :return scheduling results, which is a list of machine id
        """
        scheduling_results = []
        for task in range(task_num):
            scheduling_results.append(self.cur_machine_id)
            self.cur_machine_id = (self.cur_machine_id + 1) % self.machine_num
        return scheduling_results
