from abc import abstractmethod
from ..worker import Worker
import numpy as np

class FrameField(Worker):
    """
    Abstract base class for any frame field implementations.
    A frame field is a set of vector fields defined on a surface or volume mesh.
    """

    def __init__(self, element:str, name:str="FrameField", verbose=True):
        super().__init__(name, verbose)
        self.initialized : bool = False
        self.smoothed : bool = False
        self._element : str = element
        self.var = None

    @property
    def element(self) -> str:
        """The element of a mesh on which the frame field is defined. Either vertices, edges, faces or cells"""
        return self._element

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
        if not self.initialized:
            self.log("Initialize")
            self.initialize()
            self.initialized = True
        if not self.smoothed:
            self.log("Optimize")
            self.optimize()
            self.smoothed = True

    def normalize(self):
        if self.var is None: return
        self.var /= np.maximum(np.abs(self.var), 1e-20)

    @abstractmethod
    def export_as_mesh(self, *args, **kwargs):
        pass

    @abstractmethod
    def flag_singularities(self, *args, **kwargs):
        pass