# Pearson's correlational coefficient

#refernce = http://www.srmuniv.ac.in/sites/default/files/downloads/CORRELATION.pdf

# When pearson's coefficient is applied to the sample of the population the i is known as
# SAMPLE PEARSON CORRELATIONAL COEFFICIENT

# given an sample karl pearson's coefficient of correlation says the linear realtionship between
# twovariable x and y


        # Formula is [(n*sum(xi,yi) - sum(xi)*sum(yi))]/[((n*sum(xi^2)-sum(xi)^2)^0.5 )*(n*sum(yi^2)-sum(yi)^2)^0.5)] # most preferable
        # or sum[(xi-xbar)*(yi-ybar)] / [(sum((xi-xbar)^2)*sum((yi-ybar)^2)]^0.5

        #n = sample size
        #xabr,ybar sample mean
#code 1
import numpy as np
r = np.corrcoef(x,y)[0,1]

#code2
import numpy as np

def coff(x,y):

    n = size(x)

    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    for i in range(n):
        a += x[i]*y[i]
        b += x[i]
        c += y[i]
        d += x[i]**2
        e += y[i]**2

    r = (n*a - b*c)/n*pow((d-b**2)*(e-c**2),0.5)
    return r

#code3
import numpy as np

def coeff(x,y):
    
    xbar = np.mean(x)
    ybar = np.mean(y)

    sdevxy = 0
    sdevy = 0
    sdevx = 0

    for i in range(len(x)):
        sdevxy += (x[i]-xbar)*(y[i]-ybar)
        sdevx += (x[i]-xbar)**2
        sdevy += (y[i]-ybar)**2
        
    r = sdevxy/pow((sdevx*sdevy),0.5)

    return r

#Properties of r
'''
    if r = -1 negative perfect corelation
    if r =  1 positive perfect corelation
    1 to 0.75 strong
    0.5 to 0.75 moderate
    0.25 to 0.5 weak
    0.25 < no relation
    the same for - as well

    1. r is independent of the change in scale and orign
    2. It is the geometric mean of two regression coefficient
    3. It is symetric coff(x,y) = coff(y,x)

    LIMITAION
     1 always assueme that there is a linear relationship
     2 interpreting the value of r is difficult
     3 time consuming
     4 affected by extreme rules or stray point
'''

## Coefficint of determination = r^2 or = explained variation/ total variation

# if r = 0.9 then cod is 0.81 which means 81% of the variation in the dependent variable is explained by the independent variable
import numpy as np

coeffecient_of_determination = (np.corrcoef(x,y)[0,1])**2
