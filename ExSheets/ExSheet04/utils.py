import numpy
import numpy as np
import numpy.linalg

def getstationary(P: np.ndarray) -> np.ndarray:
    "Return the stationary distribution of a markov chain with transition matrix P"
    A = numpy.concatenate([(P.T - numpy.identity(8)), numpy.ones([1, 8])], axis=0)
    b = numpy.array([0] * 8 + [1])
    b, b[-1] = numpy.zeros(9), 1
    return numpy.linalg.lstsq(A, b, rcond=None)[0]


def mcstep(X: np.ndarray, Ppad: np.ndarray, seedval: int) -> np.ndarray:
    "Performs a Markov chain transition for multiple particles in parallel"
    Xp = numpy.dot(X, Ppad)
    Xc = numpy.cumsum(Xp, axis=1)
    L, H = Xc[:, :-1], Xc[:, 1:]
    R = numpy.random.mtrand.RandomState(seedval).uniform(0, 1, [len(Xp), 1])
    return (R > L) * (R < H) * 1.0

def getinitialstate() -> np.ndarray:
    "# Initial position of particles in the lattice"
    X = numpy.random.mtrand.RandomState(123).uniform(0, 1, [1000, 8])
    X = (X == numpy.max(X, axis=1)[:, numpy.newaxis]) * 1.0
    return X
