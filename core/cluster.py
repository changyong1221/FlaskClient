from scheduler.RoundRobinScheduler import RoundRobinScheduler
from src_scheduling.log import print_log


class Cluster(object):
    def __init__(self, domain_id, cluster_id):
        """Initialization
        """
        self.domain_id = domain_id      # domain id
        self.cluster_id = cluster_id    # cluster id
        self.machine_list = []          # machine list
        self.idle_machine_list = []     # idle machine list
        self.scheduler = None
        self.state = "IDLE"             # state in {"INIT", "COMMITTED", "RUNNING"}

    def add_machine(self, machine):
        """Add machine to cluster
        """
        self.machine_list.append(machine)

    def delete_machine(self, machine):
        """Delete machine from cluster
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
            print_log(f"scheduler of cluster({self.cluster_id}) is not set, use RoundRobinScheduler on default.")
            self.set_scheduler(RoundRobinScheduler(len(self.machine_list)))
        if self.scheduler.__class__.__name__ == "RoundRobinScheduler":
            schedule_ret = self.scheduler.schedule(len(task_list))
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.state = "COMMITTED"
            self.run_tasks()
        elif self.scheduler.__class__.__name__ == "DQNScheduler":
            schedule_ret = self.scheduler.schedule(task_list, self.machine_list)
            for idx, machine_id in enumerate(schedule_ret):
                self.machine_list[machine_id].add_task(task_list[idx])

            self.state = "COMMITTED"
            self.run_tasks()

            self.scheduler.learn(task_list, schedule_ret)

    def run_tasks(self):
        """Run the committed tasks
        """
        if self.state == "IDLE":
            print_log(f"please commit tasks to cluster({self.cluster_id}) first before running tasks!")
        elif self.state == "RUNNING":
            print_log(f"please wait for the last round finished.")
        else:
            self.state = "RUNNING"
            for machine in self.machine_list:
                machine.execute_tasks()
            self.state = "IDLE"

    def reset(self):
        """Reset cluster
        """
        for machine in self.machine_list:
            machine.reset()


def create_cluster(domain_id, cluster_id):
    """Create one cluster with cluster_id 0
    """
    return Cluster(domain_id, cluster_id)


def create_clusters(domain_id, cluster_num):
    """Create a cluster list
    """
    cluster_list = []
    for idx in range(cluster_num):
        cluster_list.append(Cluster(domain_id, idx))
    return cluster_list
