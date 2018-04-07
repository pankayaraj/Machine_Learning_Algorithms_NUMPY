
#NOTE IF THE PARAMETERS ARE SMALL AND THE INV IS OBTAINABLE  NORMAL METHOD IS A MORE CONVIENIENT ONE

#sizeofvariable = n
#training examples = m
#y is m*1 array
# NOTE here x shd be a m*(n+1) matrix and theta should be a (n+1)*1 matrix(of inital guess mostly 0s)
import numpy as np

def gradient_descent_LinR(x, y, theta, no_iterations, alpha, m):

    xtrans = np.transpose(x)  
    
    for _ in range(no_iterations):

        hypothesis = np.dot(theta,xtrans) #here hypotheses in a m*1 array where each element is the hypothesis for a paticular set  of training example
        loss = hypothesis - y
        
        #cost = (np.sum(loss**2))/(2*m)
    
        gradient = np.dot(xtrans,loss)/ m # gradient is (n+1)*1 it is computed for the convenience of calculating all the thetas simultaneously(there are n+1)
            
        theta = theta- gradient*alpha

    return theta    

    
# Assuming the end using the variance in the value J(theta) NOT A VERY GOOD METHOD AS WE CAN'T FIND THE EXACT VARIANCE NEEDED

import numpy as np

def gradient_descent_LinR(x, y, theta, variance, alpha, m):

    xtrans = np.transpose(x)
    nonaccuracy = True
    previous = float("inf") #incase all the thetas are zero  as we begin
    
    while nonaccuracy:

        hypothesis = np.dot(theta,xtrans) #here hypotheses in a 1*m array where each element is the hypothesis for a paticular set  of training example
        loss = hypothesis - y

        cost = (np.sum(loss**2))/(2*m)

        gradient = np.dot(xtrans,loss) # gradient is 1*(n+1) it is computed for the convenience of calculating all the thetas simultaneously(there are n+1)
            
        theta = theta- gradient*alpha

        if abs(cost- previous) < variance:
            nonaccuracy = False
        else:                                         #for the easiness using costfunction without square to calculate the variance 
            previous = cost

    return theta
