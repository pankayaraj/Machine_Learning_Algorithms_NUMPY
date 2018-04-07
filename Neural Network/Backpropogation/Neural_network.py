#Neural networks For classification using sigmoidal function as the activation function

import random
import numpy as np

    #  1 INITALIZING FUNCTION
'''
def initaliztion(no_inputs, no_neurons_HL, no_outputs):
    network = [] 

    hidden_layer = [[0 for i in range(no_inputs+1)] for j in range(no_neurons_HL+1)]
    network.append(hidden_layer)
    output_layer = [[0 for i in range(no_neurons_HL+1)] for j in range(no_outputs)]
    network.append(output_layer)

    return network
'''
# here for no_outputs and no_inputs functions use numerical values and for the hidden layer present a list of the size of the hidden layers and every elemnt should denote
#  the number of neurons present in the hidden layers ingnoring the bias term
def initialization( no_inputs, hiddenlayer, no_outputs):

    network = []
    
    input_layer = [[random.random() for i in range(no_inputs+1)] for j in range(hiddenlayer[0])]                  #here input_layer denotes the thetas that transfor inputs form to HL1
    network.append(input_layer)
    for k in range( len(hiddenlayer)-1):
        hidden_layer = [[random.random() for i in range(hiddenlayer[k]+1)] for j in range(hiddenlayer[k+1])]
        network.append(hidden_layer)

    output_layer = [[random.random() for i in range(hiddenlayer[-1]+1)] for j in range(no_outputs)]
    network.append(output_layer)

    return network

#network = initialization(2,[2],2)
#print (network)

# Network returns a function list with 2 lists which denots the two layes hidden and output
# To acess the 3rd neuron's thetas all we have to do is network[1][3]


# the zeros u get are the weights u get form the inputs including the bias

# Now we have to make  a function to get a theta*X term for each neuron in the next layer

   # 2 FORWARD PROPAGATION FUNCTIONS
def zigma( theta, x):
    
    x = np.array(x)
    theta = np.array(theta)
    # if x is given as a row matrix
    zigma = np.dot(theta,x) # here zigma will be diliverd as column matrix where each element gives the zigmas needed for eaah neuron in each layer
    
    '''
    #else if it is a 2 dimential matrix
    xtrans = np.transpose(x)
    zigma = np.dot(theta, xtrans)
    '''
    return zigma


def transfer(zigma):
    new_inputs = []
    for i in zigma:      # here the obtained weighted inputs are trannsfered as inputs for next layer/ as outputs using sigmid function
        
        a_i = 1/(1+np.exp(-i))
        new_inputs.append(a_i)
    
    return new_inputs

def foward_propagate( network, inputs):
    
    OUT = [inputs]
    input_ = OUT[0]
    for x in range(len(network)):
                
        theta = network[x]
        
        z  = zigma( theta, input_)
        input_ = transfer(z)
        if x != len(network)-1:
            input_.append(1)
        OUT.append(input_)
        
    
    
    return OUT

#outputs =  (foward_propagate( network, [1,1,0]))
    # BACKWARD  PROPAGATION

# first of all we need the derivaative of the output(slope of the output since it is a result if sigmoidal function it can be

def derivative(outputs, len_outputs):
    l = len_outputs
    
    outputs = np.array(outputs)
    I = [1 for i in range(l)]
    I = np.array(I)

    derivative = np.multiply(outputs,(I-outputs)) # doing element wise multiplication
    
    return derivative
    
def backward_propagation( network, OUT, exp_outputs, no_layers):
    errors = []
    exp_outputs = np.array(exp_outputs)
    outputs = OUT[-1]
    outputs = np.array(outputs)
    ini_error = np.multiply((exp_outputs - outputs),derivative(outputs,len(outputs)))   # actually the ini error is 1/2*(exp_out-out)^2 but due to the
                                                                                        # computaional purposes we will end up using this values(partial derivative refer text if confusesd)
    #for cross entropy costfunction
    #ini_error = exp_outputs - outputs                                                                             
    ini_error = ini_error.tolist()
    errors.append(ini_error)

    #error calculation is not made for the input layer so it must be done to the HL since we have finished with the OL
    
    for i in range(no_layers-1): #no layers are considered ignoring the input layer
        
        theta = network[-i-1]
        if i == 0:
            error = errors[i]
            error = np.array(error)
        else:
            error = errors[i][:len(errors[i])-1]
           
            error = np.array(error)
        theta = np.array(theta)
        theta = np.transpose(theta)
        outputs = OUT[-i-2]
        outputs = np.array(outputs)
               
        
        new_error = np.multiply(np.dot(theta, error),derivative(outputs,len(outputs)))
        new_error = new_error.tolist()
        errors.append(new_error)
        
    errors.reverse()
   
    return errors    
'''
print (backward_propagation(network, outputs, [1,1,1,], 2))

print (outputs)
'''
def neuralnetwork( network, INPUT, EX_OUTPUT, no_layers, learning_rate, no_of_iterations):
    R = 0
    W = 0
    for _ in range(no_of_iterations):
        
        for n_th in range(len(INPUT)):
            inputs = INPUT[n_th]
            outputs = foward_propagate(network, inputs)
            
            exp_outputs = EX_OUTPUT[n_th]
            
            errors = backward_propagation(network, outputs, exp_outputs, no_layers)
            
            for t in range(len(network)):
                
                
                theta = network[t]
                for n in range(len(theta)):  #w_t_n : weights comming to a single neuron    input_c_e : input that caused the error
                    w_t_n = theta[n]
                    w_t_n = np.array(w_t_n)
                    
                    error  = errors[t][n]
                    
                    input_c_e =  outputs[t]
                    
                    input_c_e = np.array(input_c_e)
                    
                    error = np.array(error)
                    
                    w_t_n = w_t_n + learning_rate*error*input_c_e
                    w_t_n = w_t_n.tolist()
                    theta[n] = w_t_n
                network[t] = theta
    '''     
            if _ == no_of_iterations-1:
                
                a = 0
                b = 0
                
                x= outputs[-1]
                for i in range(len(x)):
                    if  x[i] > a:
                        a = x[i]
                        b = i
                n  = [0.0]*len(exp_outputs)
                n[b] = 1.0
                if n == exp_outputs:
                    R += 1
                else:
                    W += 1
    print (R,W)
                    
    '''                
    return (network)
X = [[1,0,1]]
Y = [[1,1]]
network = initialization(2 , [2], 2)       
#print (neuralnetwork( network, X, Y, 2,0.1,4))


new_network = neuralnetwork( network, INPUT, EX_OUTPUT, no_layers, learning_rate, no_of_iterations) # this function is for training the network
def predict(new_network, x): # later use this function for predincting the normal data sets

    output = foward_propagate(network, x)[-1]

    return output
    

    

