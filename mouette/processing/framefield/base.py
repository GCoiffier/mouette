from abc import abstractmethod
from ..worker import Worker

class FrameField(Worker):
    """
    Abstract base class for any frame field implementations.
    A frame field is a set of vector fields defined on a surface or volume mesh.
    """

    def __init__(self, name="FrameField", verbose=True):
        super().__init__(name, verbose)
        self.initialized = False
        self.var = None

    def _check_init(self):
        if not self.initialized:
            raise Exception("FrameField was not initialized properly. Call `initialize()` before `optimize()`")

    def __getitem__(self, i):
        if self.var is not None:
            return self.var[i]
        return None

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    def run(self):
        self.log("Initialize")
        self.initialize()
        self.log("Optimize")
        self.optimize()
        self.log("Done.")

    def normalize(self):
        if self.var is None: return
        for i in range(self.var.size):
            if abs(self.var[i])>1e-8: 
                self.var[i] /= abs(self.var[i])

    @abstractmethod
    def export_as_mesh(self, *args, **kwargs):
        pass

    @abstractmethod
    def flag_singularities(self, *args, **kwargs):
        pass