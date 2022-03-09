from scheduler.RoundRobinScheduler import RoundRobinScheduler
from utils.get_position import get_position_by_name
from src_scheduling.log import print_log
from utils.write_file import write_list_to_file
from utils.file_check import check_and_build_dir
import globals.global_var as glo


class Domain(object):
    def __init__(self, domain_id, location, longitude=None, latitude=None, auto_locate=True):
        """Initialization
        """
        self.domain_id = domain_id
        self.location = location        # city name, for example "北京市" "莫斯科"
        self.longitude = longitude      # longitude of geographical position
        self.latitude = latitude        # latitude of geographical position
        self.machine_list = []          # all the machines in this domain
        self.idle_machine_list = []     # idle machine list
        self.cluster_list = []          # node clusters in this domain
        self.scheduler = None

        if (self.longitude is None or self.latitude is None) and auto_locate:
            self.longitude, self.latitude = get_position_by_name(location)
        print_log(f"domain({self.domain_id}) is created.")

    def add_machine(self, machine):
        """Add machine to domain
        """
        self.machine_list.append(machine)
        print_log(f"machine({machine.machine_id}) is added to domain({self.domain_id})")

    def delete_machine(self, machine):
        """Delete machine from domain
        """
        for node in self.machine_list:
            if node is machine:
                self.machine_list.remove(node)
                break

    def set_scheduler(self, scheduler):
        """Set task scheduling strategy
        """
        self.scheduler = scheduler

    def commit_tasks(self, task_list):
        """Commit tasks to cluster
        """
        if self.scheduler is None:
            print_log(f"scheduler of domain({self.cluster_id}) is not set, use RoundRobinScheduler on default.")
            self.set_scheduler(RoundRobinScheduler(len(self.machine_list)))
        if self.scheduler.__class__.__name__ == "RoundRobinScheduler":
            schedule_ret = self.scheduler.schedule(len(task_list))
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "DQNScheduler":
            schedule_ret = self.scheduler.schedule(task_list, self.machine_list)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()

            self.scheduler.learn(task_list, schedule_ret)

    def run_tasks(self):
        """Run the committed tasks
        """
        for machine in self.machine_list:
            machine.execute_tasks()

    def reset(self):
        """Reset cluster
        """
        for machine in self.machine_list:
            machine.reset()

    def clustering_machines(self, cluster_num):
        """Clustering all the machines into several clusters
        """
        pass

    def get_cluster_list(self):
        """Return cluster list
        """
        return self.cluster_list


class MultiDomain(object):
    def __init__(self, multidomain_id, location):
        """Initialization
        """
        self.multidomain_id = multidomain_id
        self.location = location
        self.longitude = None
        self.latitude = None
        self.domain_list = []       # all the domains in the multi-domain system
        self.cluster_list = []      # all the clusters in the multi-domain system
        self.machine_list = []      # all the machines in the multi-domain system
        self.idle_machine_list = []  # idle machine list
        self.scheduler = None
        self.is_using_clustering_optimization = False
        self.version = "v1.0"
        self.print_version()
        print_log("multi-domain scheduling system is created.")

    def auto_locate(self):
        """Auto locate the longitude and latitude
        """
        self.longitude, self.latitude = get_position_by_name(self.location)

    def add_domain(self, domain):
        """Add domain to multi-domain system
        """
        self.domain_list.append(domain)
        for cluster in domain.cluster_list:
            self.cluster_list.append(cluster)
        for machine in domain.machine_list:
            self.machine_list.append(machine)
        print_log(f"domain({domain.domain_id}) is add to multi-domain scheduling system.")

    def delete_domain(self, domain):
        """Delete domain from multi-domain system
        """
        for domain_ in self.domain_list:
            if domain_ is domain:
                self.domain_list.remove(domain_)
                break
        for cluster in domain.cluster_list:
            for cluster_ in self.cluster_list:
                if cluster is cluster_:
                    self.cluster_list.remove(cluster_)
        for machine in domain.machine_list:
            for machine_ in self.machine_list:
                if machine is machine_:
                    self.machine_list.remove(machine_)

    def set_scheduler(self, scheduler):
        """Set task scheduling strategy
        """
        self.scheduler = scheduler
        print_log(f"{scheduler.__class__.__name__} task scheduler is set for multi-domain scheduling system.")

    def commit_tasks(self, task_list):
        """Commit tasks to multi-domain system
        """
        if self.scheduler is None:
            print_log(f"scheduler of multidomain is not set, use RoundRobinScheduler on default.")
            self.set_scheduler(RoundRobinScheduler(len(self.machine_list)))
        if self.scheduler.__class__.__name__ == "RoundRobinScheduler":
            schedule_ret = self.scheduler.schedule(len(task_list))
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "RandomScheduler":
            schedule_ret = self.scheduler.schedule(len(task_list))
            print("RandomScheduler schedule_ret: ", schedule_ret)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "EarliestScheduler":
            task_batch_commit_time = task_list[0].get_task_commit_time()
            self.get_idle_machine_list(task_batch_commit_time)
            schedule_ret = self.scheduler.schedule(len(task_list), len(self.machine_list), self.idle_machine_list)
            print("EarliestScheduler schedule_ret: ", schedule_ret)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "HeuristicScheduler":
            schedule_ret = self.scheduler.schedule(task_list)
            print("HeuristicScheduler schedule_ret: ", schedule_ret)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "DQNScheduler":
            schedule_ret = self.scheduler.schedule(task_list, self.machine_list)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
            makespan = 0
            for machine in self.machine_list:
                makespan = max(makespan, machine.get_batch_makespan())
            self.scheduler.learn(task_list, schedule_ret, makespan)
        elif self.scheduler.__class__.__name__ == "DDPGScheduler":
            schedule_ret = self.scheduler.schedule(task_list, self.machine_list)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.run_tasks()
            self.scheduler.learn(task_list, schedule_ret)

        # save results
        self.save_batch_results(task_list)

    def run_tasks(self):
        """Run the committed tasks
        """
        print_log("committed tasks is running...")
        for machine in self.machine_list:
            machine.execute_tasks(self.multidomain_id)

    def save_batch_results(self, task_list):
        """Save batch results
        """
        # 计算平均任务处理时间
        batch_avg_task_processing_time = 0
        for task_instance in task_list:
            batch_avg_task_processing_time += task_instance.get_task_processing_time()
        batch_avg_task_processing_time = round(batch_avg_task_processing_time / len(task_list), 4)

        # 计算makespan和平均worktime
        makespan = 0
        worktime = 0
        for machine in self.machine_list:
            makespan = max(makespan, machine.get_batch_makespan())
            worktime += machine.get_batch_makespan()
        makespan = round(makespan, 4)
        avg_worktime = round(worktime / len(self.machine_list), 4)

        if glo.is_federated:
            output_dir = f"results/task_run_results/federated/batch/client-{self.multidomain_id}/{glo.federated_round}"
            check_and_build_dir(output_dir)
            output_path = output_dir + "/task_batches_run_results2.txt"
            if glo.is_test:
                output_dir = f"results/task_run_results/federated/batch/federated_test/{glo.federated_round}"
                check_and_build_dir(output_dir)
                output_path = output_dir + "/task_batches_run_results2.txt"
            output_list = [batch_avg_task_processing_time, makespan, avg_worktime]
            write_list_to_file(output_list, output_path, mode='a+')
        else:
            output_dir = f"results/task_run_results/{glo.current_dataset}{glo.records_num}/{glo.current_scheduler}/task_batches/"
            check_and_build_dir(output_dir)
            output_path = output_dir + "task_batches_run_results2.txt"
            output_list = [batch_avg_task_processing_time, makespan, avg_worktime]
            write_list_to_file(output_list, output_path, mode='a+')

    def get_idle_machine_list(self, task_commit_time):
        """Get idle machines according to the commit tasks
        """
        self.idle_machine_list.clear()
        for machine in self.machine_list:
            if machine.get_finish_time() <= task_commit_time:
                self.idle_machine_list.append(machine)

    def reset(self):
        """Reset cluster
        """
        for machine in self.machine_list:
            machine.reset()
        print_log("multi-domain scheduling system is reset.")

    def print_version(self):
        """Print version at initialization
        """
        print_log("--------------------------------------------------------")
        print_log("|                                                      |")
        print_log("|       Cross-domain task scheduling system v1.0       |")
        print_log("|                                                      |")
        print_log("--------------------------------------------------------")


def create_domain(domain_id, location_name):
    """Create one domain with domain_id 0, default location is "北京市"
    """
    return Domain(domain_id, location_name)


def create_domains(location_list):
    """Create multiple domains with given location_list
    """
    domain_list = []
    for i, location in enumerate(location_list):
        domain_list.append(Domain(i, location))
    return domain_list


def create_multi_domain(multidomain_id, location):
    """Create multi-domain system using singleton pattern
    """
    multi_domain = MultiDomain(multidomain_id, location)
    multi_domain.auto_locate()
    glo.location_longitude = multi_domain.longitude
    glo.location_latitude = multi_domain.latitude
    return multi_domain
