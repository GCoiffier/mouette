import numpy as np

class Vec(np.ndarray):
    """A simple class to manipulate vectors in mouette. 
    Basically, it inherits from a numpy array and implements some quality of life features for 2D and 3D vectors especially.
    """
    ###### Constructors ######
    def __new__(cls, *a):
        if len(a)==1:
            obj = np.asarray(a[0]).view(cls)
        else:
            obj = np.asarray(a).view(cls)
        return obj
    
    @classmethod
    def from_complex(cls, c : complex):
        """2D vector from complex number

        Args:
            c (complex):

        Returns:
            Vec: `Vec(c.real, c.imag)`
        """
        return cls(c.real, c.imag)

    @classmethod
    def random(cls, n : int):
        """Generates a random vector of size `n` with coefficients sampled uniformly and independently in [0;1)

        Args:
            n (int): size

        Returns:
            Vec:
        """
        return cls(np.random.random(n))

    @classmethod
    def zeros(cls, n : int):
        """Generates a vector of size `n` full of zeros.

        Args:
            n (int): size

        Returns:
            Vec:
        """
        return cls(np.zeros(n))
    
    @classmethod
    def X(cls):
        """The [1,0,0] vector

        Returns:
            Vec: [1,0,0]
        """
        return Vec(1.,0.,0.)

    @classmethod
    def Y(cls):
        """The [0,1,0] vector

        Returns:
            Vec: [0,1,0]
        """
        return Vec(0.,1.,0.)

    @classmethod
    def Z(cls):
        """The [0,0,1] vector

        Returns:
            Vec: [0,0,1]
        """
        return Vec(0.,0.,1.)

    ###### Accessors ######
    @property
    def x(self) -> float: 
        """First coordinate of the vector

        Returns:
            float: `vec[0]`
        """
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        """Second coordinate of the vector

        Returns:
            float: `vec[1]`
        """
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def xy(self):
        """First two coordinates of the vector

        Returns:
            float: `vec[:2]`
        """
        return self[:2]

    @property
    def z(self):
        """Third coordinate of the vector

        Returns:
            float: `vec[3]`
        """
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    ###### Methods ######
    def norm(self, which="l2") -> float :
        """Vector norm. Three norms are implemented: the Euclidean l2 norm, the l1 norm or the l-infinite norm:

        l2 : $ \\sqrt{ \\sum_i v[i]^2 } $

        l1 : $ \\sum_i |v[i]| $

        linf : $ \\max_i |v[i]| $

        Args:
            which (str, optional): which norm to compute. Choices are "l2", "l1" and "linf". Defaults to "l2".

        Returns:
            float: the vector's norm
        """
        if which=="l2":
            return np.sqrt(np.dot(self, self))
        elif which=="l1":
            return np.sum(np.abs(self))
        elif which=="linf":
            return np.max(np.abs(self))

    def dot(self, other) -> float:
        """Dot product between two vectors:

        $ a \\cdot b = \\sum_i a[i]b[i]$

        Args:
            other (Vec): other vector to dot with

        Returns:
            float: the dot product
        """
        return np.dot(self, other)

    def outer(self,other) -> np.ndarray:
        """Outer product between two vectors:

        $a \\otimes b = c \\in \\mathbb{R}^{n \\times n}$ such that $c[i,j] = a[i]b[j]$.

        Args:
            other (Vec): the second vector

        Returns:
            np.array: an array of shape (n,n)
        """
        return np.outer(self,other)

    def normalize(self, which="l2"):
        """Normalizes the vector to have unit norm.

        Args:
            which (str, optional): which norm to compute. Choices are "l2", "l1" and "linf". Defaults to "l2".
        """
        self /= self.norm(which)

    @staticmethod
    def normalized(vec, which="l2"):
        """Computes and returns a normalized vector of the input vector `vec`.

        Args:
            vec (Vec): the input vector
            which (str, optional): which norm to compute. Choices are "l2", "l1" and "linf". Defaults to "l2".

        Returns:
            Vec: the normalized vector
        """
        nrm = Vec.norm(vec, which)
        np.seterr(all='raise')
        # we want the following to fail when a division by zero is encountered
        out = Vec(vec/nrm)
        np.seterr(all='warn')
        return out