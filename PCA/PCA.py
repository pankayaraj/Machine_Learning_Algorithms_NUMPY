import theano.tensor as T
from theano import function
import numpy as np

# Data should be given in the fromat such that
    #Columns : Features/Dimensions
    #Rows : Samples

   #    1,1,1
   #    2,2,2       here 1,1,1 is a sample
   #    3,3,3

#Each row is a sample

def PCA_EVD(X, centered = None):

    x = T.dmatrix('x')
    y = T.matrix('x')
    z = T.mean(x, axis=0, dtype='float64')
    d = T.dot(x,y)
    e = T.nlinalg.eig(x)

    mean = function([x], z)
    dot  = function([x,y], d)
    eig  = function([x], e)


    if centered != True:
        Xmean = mean(X)
        X = X - Xmean

    X_new = dot(np.transpose(X), X)
    eignvalue, s = eig(X_new)

    index = eignvalue.argsort(-1)
    s = np.transpose(s)
    new_s = np.array([s[index[i]] for i in range(len(index) - 1, -1, -1)])
    new_s = np.transpose(new_s)

    Y = dot(X, new_s)

    return Y




a = [[7.7998,5.9417,1.4767],[7.5617,5.2167,1.3433],[7.2800,4.1267,1.3400],[7.0083,2.9983,1.3300],[6.6483,2.3000,1.3150],[7.3333,2.8150,2.0067],[7.7433,3.9783,3.6833],[5.8750,3.0267,2.4633],[5.3950,4.2567,4.1267],[5.8850,4.9300,3.6283],[6.4700,5.8067,3.3550],[7.2850,6.4267,1.93673]]

z = PCA_EVD(a, centered= False)
print(z)
