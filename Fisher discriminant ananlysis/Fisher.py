import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab


#m is a list of matrices where each matrix belongs to a paticular class
# In each matrix each dimension of the data are given by a column
# A single data is along the a row

def fisher(m):
    no_classes = len(m)
    no_data =[]
    no_dimension = 0

    for i in range(no_classes):
        r, c = np.shape(np.array(m[i]))
        no_data.append(r)
        no_dimension= c

    total_data = sum(no_data)

    classwise_mean = []
    tot = [0 for _ in range(no_dimension)]
    for i in range(no_classes):
        mean = []
        a = m[i]
        a = np.array(a)
        for j in range(no_dimension):
            mean.append(np.mean(a[:,j]))
            tot[j] += np.sum(a[:,j])
        classwise_mean.append(mean)

    total_mean = [x/total_data for x in tot]

    Sb = np.array([[0.0 for _ in range(no_dimension)] for ___ in range(no_dimension)])

    for i in range(no_classes):
        u = np.array(classwise_mean[i]) - np.array(total_mean)
        u.shape = (no_dimension, 1)
        Sb += np.dot(u, np.transpose(u))*no_data[i]

    Sw = np.array([[0.0 for _ in range(no_dimension)] for ___ in range(no_dimension)])

    for i in range(no_classes):
        c = m[i]
        c = np.array(c)
        cov = np.cov(np.transpose(c))
        Sw += cov*no_data[i]

    eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))

    index = eigval.argsort(-1)
    eigvec = np.transpose(eigvec)
    new_eigvec = np.array([eigvec[index[i]] for i in range(len(index)-1,-1,-1)])


    return new_eigvec



