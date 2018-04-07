# LOGISTIC REGRESSION



import numpy as np

def gradient_descnt_logR( y, x, theta, no_of_iterations, alpha):

    for  _ in xrange(no_of_iterations):
        xtrans = np.transpose(x)
        thetaX = np.dot(theta, xtrans)

        hypothesis = []
        for i in thetax:

            value = 1/(1+np.exp(-i))
            hypothesis.append(value)

        hypothesis = np.array(hypothesis)

        loss = hypothesis - y

        gradient = np.dot(xtrans,loss)/m

        theta = theta - alpha*gradient

    return theta
