# Gradient descent algorithm for linear regression with one variable
# Reference ;  Andrew Ng lecture( onlie courese)


# Note : Let's say that y depends on variable x and we have a training set of x
#        x and y aim of this function is to find a best fit linear graph that would
#        that would find a close value 

# Let's say that the function is H = a1*x + a2 (hypothesis function)
# At first cost function J(a1,a2) can be calculated for a set of values a1 and a2 and they
# can be calculated by the algorithm until they coverge(usually we consider 0,0 ans our intial value)

# at each iteration when we change the a1 and a2 we will calculate the function for all training sets
# this is called  BATCH GRADIENT DESCENT

# J(a1,a2) = 1/2*m( summation of ( H(xi) - yi )^2 for i = 0 to m)  where m = size if training set
                                # this is SQUARED ERROR FUNCTION which  must be minimized

# then we update a1 and a2 simultaneously for the function below( partial derivation regarding to a1 and a2 are considered here

#  a2 = a2 - alpha*( 1/m( summation of ( H(xi) - yi ) for i = 0 to m)
#  a1 = a1 - alpha*( 1/m( summation of ( H(xi) - yi )*xi  for i = 0 to m)

# Alpha is the LEARNING RATE which must be choosed carefully as it controls the rate at which the gradient descent comes down

# we get our answer when we get a convergence. Concergence can be assumed by taking the difference to
# a paticular decimal palce or by considering a suitable number of iterations


# NOTE: While claculationg gradient descent we may end up in differnt local minimums but for various guesses
#       But as it turns out linear regresssion always gives a bowl shaped cureve(convex function) with a single local minimum(global optimum). So it is ok to make any guess.
def gradient_descent( x, y, alpha, m, a1, a2, no_of_iterations):
     
    
    for _ in range(no_of_iterations):
        h1 = 0
        h2 = 0
        
        for i in range(m):
            h1 += a1*x[i] + a2 - y[i]
            h2 += (a1*x[i] + a2 - y[i])*x[i]
        a1 = a1 - (alpha*h1)/m
        a2 = a2 - (alpha*h2)/m


    return [a1, a2]

def gradient_descent( x, y, alpha, m, a1, a2, precision):
     
    b1 = float('inf')
    b2 = float('inf')
    while abs(a1-b1) > precision and abs(a2-b2) > precision:
        h1 = 0
        h2 = 0
        
        for i in range(m):
            h1 += a1*x[i] + a2 - y[i]
            h2 += (a1*x[i] + a2 - y[i])*x[i]

        b1 = a1
        b2 = a2
        a1 = a1 - (alpha*h1)/m
        a2 = a2 - (alpha*h2)/m


    return [a1, a2]

