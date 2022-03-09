from utils.write_file import write_list_to_file
from utils.file_check import check_and_build_dir
from src_scheduling.log import print_log
import src_scheduling.globals as glo


class Machine(object):
    def __init__(self, machine_id, mips, memory, bandwidth):
        """Initialization
        """
        self.machine_id = machine_id
        self.ip_address = None      # 跟 flask客户端对应
        self.port = None            # 跟 flask客户端对应
        self.longitude = 0
        self.latitude = 0
        self.mips = mips
        self.memory = memory
        self.bandwidth = bandwidth
        self.realtime_cpu_utilization = 0
        self.realtime_memory_utilization = 0
        self.realtime_bandwidth_utilization = 0
        self.finish_time = 0
        self.work_time = 0
        self.batch_makespan = 0
        self.task_waiting_queue = []

    def add_task(self, task):
        """Add task run instance to task_waiting_queue
        """
        self.task_waiting_queue.append(task)
        print_log(f"task({task.task_id}) ---> machine({self.machine_id})")

    def execute_tasks(self, multidomain_id):
        """Execute tasks in the task_waiting_queue
        """
        self.batch_makespan = 0
        for task in self.task_waiting_queue:
            task.run_on_machine(self, multidomain_id)
            self.work_time += task.get_task_processing_time()
            self.batch_makespan += task.get_task_processing_time()
            self.realtime_cpu_utilization = task.get_task_cpu_utilization()
            self.realtime_memory_utilization = round(task.get_task_size() / self.memory, 4)
            self.realtime_bandwidth_utilization = 1
            scheduler_name = glo.get_global_var("current_scheduer")
            output_dir = f"results/machine_status_results/train/client-{multidomain_id}/{scheduler_name}/{glo.get_global_var('current_round')}"
            check_and_build_dir(output_dir)
            output_path = output_dir + f"/{self.machine_id}_status.txt"
            if glo.get_global_var("is_federated_test"):
                output_dir = f"results/machine_status_results/test/client-{multidomain_id}/{scheduler_name}/{glo.get_global_var('current_round')}"
                check_and_build_dir(output_dir)
                output_path = output_dir + f"/{self.machine_id}_status.txt"
            output_list = [self.work_time, self.realtime_cpu_utilization, self.realtime_memory_utilization,
                           self.realtime_bandwidth_utilization]
            write_list_to_file(output_list, output_path, mode='a+')

        self.task_waiting_queue.clear()

    def set_location(self, longitude, latitude):
        """Set longitude and latitude
        """
        self.longitude = longitude
        self.latitude = latitude

    def reset(self):
        """Reset machine to initial state
        """
        self.realtime_cpu_utilization = 0
        self.realtime_memory_utilization = 0
        self.realtime_bandwidth_utilization = 0
        self.finish_time = 0
        self.task_waiting_queue.clear()

    def get_machine_id(self):
        """Return machine_id
        """
        return self.machine_id

    def get_mips(self):
        """Return mips
        """
        return self.mips

    def get_memory(self):
        """Return memory
        """
        return self.memory

    def get_bandwidth(self):
        """Return bandwidth
        """
        return self.bandwidth

    def get_finish_time(self):
        """Return finish_time
        """
        return self.finish_time

    def set_finish_time(self, new_finish_time):
        """Set a new finish time
        """
        self.finish_time = new_finish_time

    def get_batch_makespan(self):
        """Return batch_makespan
        """
        return self.batch_makespan
