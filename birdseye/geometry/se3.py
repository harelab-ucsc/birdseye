import numpy as np
from scipy.linalg import expm


class SE3:

    @staticmethod
    def skew(v):
        v = np.asarray(v)

        if v.ndim == 1:
            return np.array([
                [0,      -v[2],  v[1]],
                [v[2],    0,    -v[0]],
                [-v[1],   v[0],  0],
            ])

        S = np.zeros((len(v),3,3), dtype=v.dtype)

        S[:,0,1] = -v[:,2]
        S[:,0,2] =  v[:,1]
        S[:,1,0] =  v[:,2]
        S[:,1,2] = -v[:,0]
        S[:,2,0] = -v[:,1]
        S[:,2,1] =  v[:,0]

        return S

    @staticmethod
    def hat(xi):
        rho = xi[:3]
        phi = xi[3:]
        Xi = np.zeros((4,4))
        Xi[:3,:3] = SE3.skew(phi)
        Xi[:3,3] = rho
        return Xi

    @staticmethod
    def apply_se3_update(T, xi):
        # left-sided update
        return expm(SE3.hat(xi)) @ T
