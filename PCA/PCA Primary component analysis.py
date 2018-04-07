# Primary Component Analysis
# If the data are rows and a dimension is held by each column then axis = 0
# If the data are columns and a dimension is held by each row then axis = 1

import numpy as np
import random
import time
from matplotlib import pyplot as plt


def PCA_EVD(X, axis):
    t = time.time()
    Xmean = np.mean(X, axis=axis)

    if axis == 0:
        X = X - Xmean
    else:
        X = np.transpose(X) - Xmean
        X = np.transpose(X)


    if axis == 0:
        new_X = np.dot(np.transpose(X), X)
    else:
        new_X = np.dot(X, np.transpose(X))


    eignvalue, s = (np.linalg.eig(new_X))

    index = eignvalue.argsort(-1)
    s = np.transpose(s)

    new_s = np.array([s[index[i]] for i in range(len(index)-1,-1,-1)])
    new_s = np.transpose(new_s)

    # Manually truncate the new_s according o the size of the eignvalue to get a smaller dimension
    if axis == 0:
        Y = np.dot( X, new_s)
    else:
        Y = np.dot(np.transpose(new_s), X)


    return Y

'''
print("Method 1")
print(PCA_SVD([[3,2,2], [2,3,-2]]))
print("Method 2")
print(PCA_EVD([[3, 2, 2], [2, 3, -2]]))
'''

a = [[7.7998,5.9417,1.4767],[7.5617,5.2167,1.3433],[7.2800,4.1267,1.3400],[7.0083,2.9983,1.3300],[6.6483,2.3000,1.3150],[7.3333,2.8150,2.0067],[7.7433,3.9783,3.6833],[5.8750,3.0267,2.4633],[5.3950,4.2567,4.1267],[5.8850,4.9300,3.6283],[6.4700,5.8067,3.3550],[7.2850,6.4267,1.93673]]


a = np.array(a)
a = np.transpose(a)

z = PCA_EVD(a,1)
print(z)



