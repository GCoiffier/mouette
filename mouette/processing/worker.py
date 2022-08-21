from ..utils import Logger
from abc import ABC, abstractmethod

class Worker(Logger, ABC):

    def __init__(self, name:str, verbose:bool = False):
        super().__init__(name, verbose)

    @abstractmethod
    def run(self, *args, **kwargs):
        """A M.Worker implements a `.run()` method where the algorithm is run on an input mesh"""
        pass

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return self