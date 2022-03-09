from abc import ABCMeta


class Scheduler(metaclass=ABCMeta):
    """Abstract class
    """
    @classmethod
    def schedule(cls):
        pass
