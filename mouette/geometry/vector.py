import numpy as np

class Vec(np.ndarray):
    
    ###### Constructors ######
    def __new__(cls, *a):
        if len(a)==1:
            obj = np.asarray(a[0]).view(cls)
        else:
            obj = np.asarray(a).view(cls)
        return obj
    
    @classmethod
    def from_complex(cls, c : complex):
        return cls(c.real, c.imag)

    @classmethod
    def random(cls, n : int):
        return Vec(np.random.random(n))

    @classmethod
    def zeros(cls, n : int):
        return Vec(np.zeros(n))
    
    @classmethod
    def X(cls): 
        return Vec(1.,0.,0.)

    @classmethod
    def Y(cls): 
        return Vec(0.,1.,0.)

    @classmethod
    def Z(cls): 
        return Vec(0.,0.,1.)

    ###### Accessors ######
    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def xy(self):
        return self[:2]

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    ###### Methods ######
    def norm(self, which="l2") -> float :
        if which=="l2":
            return np.sqrt(np.dot(self, self))
        elif which=="l1":
            return np.sum(np.abs(self))
        elif which=="linf":
            return np.max(np.abs(self))

    def dot(self, other) -> float:
        return np.dot(self, other)

    def outer(self,other):
        return np.outer(self,other)

    def normalize(self, which="l2"):
        self /= self.norm(which)

    @staticmethod
    def normalized(vec, which="l2"):
        nrm = Vec.norm(vec, which)
        np.seterr(all='raise')
        # we want the following to fail when a division by zero is encountered
        out = Vec(vec/nrm)
        np.seterr(all='warn')
        return out