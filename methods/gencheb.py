import torch
from gpytorch.kernels import Kernel
import numpy as np
import pandas as pd
class gencheb(Kernel):
    
    def __init__(self,**kwargs):
        super(gencheb, self).__init__(**kwargs)


    
    def forward(self, x, y, active_dims=6,**params):
        x_tmp = x.detach().numpy()
        y_tmp = y.detach().numpy()
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(x, dtype=torch.float)
        n = active_dims
        m = len(x)
        if n == 0:
            K = 1
        elif n == 1:
            K = np.dot(x, y)
        else:
            K = 0
            T1matrix = self.Tgenerate(x, n, tmp = 1)
            T2matrix = self.Tgenerate(y, n, tmp = 1)
            for i in range(n + 1):
                if not i % 2:
                    K = K + T1matrix[i, 0] * T2matrix[i, 0]
                else:
                    K = K + np.dot(T1matrix[i, :], T2matrix[i, :])
        #K = K / torch.sqrt(m - torch.dot(x.reshape(-1),y.reshape(-1)))
        return K

    def Tgenerate(self, x, ni, tmp):
        vectorsize = x.shape[tmp]
        T_matrix = np.zeros((ni+1, vectorsize))
        T_matrix[0,:] = 1
        T_matrix[1,:] = x
        for i in range(2, ni+1):
            if not i % 2:
                T_1 = T_matrix[i-1,:].T
                T_2 = T_matrix[i-2,0]
            else:
                T_1 = T_matrix[i-1,0].T
                T_2 = T_matrix[i-2,:]
            T_matrix[i,:] = 2 * np.dot(x , T_1) - T_2
        return  torch.tensor(T_matrix, dtype=torch.float)    