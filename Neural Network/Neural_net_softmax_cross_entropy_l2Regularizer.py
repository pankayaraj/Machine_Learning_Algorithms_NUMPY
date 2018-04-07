# Neural  network with a final layer softmax function and sigmoidal function for other layers and a cross entropy cost function anlong with L2 REGULARIZTION

import random
import numpy as np

def softmax(x):
    x = np.array(x)
    out = np.exp(x)
    a = sum(out)
    for i in range(len(x)):
        out[i] = out[i]/a
    
    return out

def zigma(theta, x):
    theta = np.array(theta)
    x = np.array(x)
    zigma = np.dot(theta, x)

    return zigma

def sigmoid(x):
    x = np.array(x)
    out = np.exp(-x)
    for i in range(len(x)):
        out[i] = 1/(1+out[i])

    return out

def derivative(x):
    a = len(x)
    x = np.array(x)
    I = np.ones(a)
    der = np.multiply( x, (I-x))
    
    return der

    

class neuralnet:

    def __init__(self, input_layer_size, hidden_layer, output_layer_size):

        self.theta = []
        self.no_of_hidden_layers = len(hidden_layer)
        
        
        in_layer = [[ random.random() for i in range(input_layer_size+1)] for j in range(hidden_layer[0])]
        self.theta.append(in_layer)

        for k in range(1,len(hidden_layer)):

            hid_layer = [[ random.random()  for i in range(hidden_layer[k-1]+1)] for j in range(hidden_layer[k])]
            self.theta.append(hid_layer)

        out_layer = [[ random.random() for i in range(hidden_layer[-1] + 1)] for j in range( output_layer_size)]
        self.theta.append(out_layer)

        self.theta_len = len(self.theta)

    def foward_propogate(self, data):
        output  =  []
        
        output.append(data)
        no_of_transformations = len(self.theta)
        
        input_ = output[0]
        for i in range(no_of_transformations-1):
            z = zigma(self.theta[i], input_)
            new_input = sigmoid(z)
            new_input = new_input.tolist()
            new_input.append(1)
            
            output.append(new_input)
            input_ = new_input

        z = zigma(self.theta[-1], input_)
        new_input = softmax(z)
        new_input = new_input.tolist()
        output.append(new_input)

        return output


    def backpropogation(self, output, actual_y):
        errors = []
        out = output[-1]

        out = np.array(out)
        actual_y = np.array(actual_y)
        
        e = actual_y - out
        e = e.tolist()
        # this is not the inital errror, it is the partial derivative of the  cross entropy cost function(for softmax) with regard to the out

        errors.append(e)

        # backpropogation part

        for i in range(len(output)-2):
            out = output[-i-2]
            out = np.array(out)

            t = self.theta[-i-1]
            t = np.array(t)
            t = np.transpose(t)

            if i == 0:
                e = errors[i]
                e = np.array(e)
            else:
                e = errors[i][:len(errors[i])-1]
                e = np.array(e)
            

            new_error = np.multiply(np.dot(t, e), derivative(out))
            new_error = new_error.tolist()
            errors.append(new_error)

        errors.reverse()
        return errors

    def neural ( self, dataset, actual_ys, learning_rate, regularizing_parameter, no_of_iterations):

        lmda = regularizing_parameter
        len_training_data  = len(dataset)

        for _ in range(no_of_iterations):

            for i in range(len_training_data):
                data = dataset[i]
                outputs = self.foward_propogate(data)
                
                errors = self.backpropogation(outputs, actual_ys[i])

                for j in range(self.theta_len):
                    t = self.theta[j]
                    
                    for k in range(len(t)):

                        w_t_n = t[k]    #w_t_n weight comming to a single neuron
                        w_t_n = np.array(w_t_n)
                        e = errors[j][k]
                        
                        regularizer = learning_rate*(lmda/len_training_data)*w_t_n
                        regularizer[-1] = 0                             #since the last element is considered as bias in a data set

                        in_ = outputs[j]
                        in_ = np.array(in_)
                        
                        w_t_n = w_t_n - regularizer + learning_rate*e*in_
                        w_t_n = w_t_n.tolist()
                        
                        
                        
                        t[k] = w_t_n

                        
                    self.theta[j] = t
                
        return self.theta

    def predict( self, x):
        out = self.foward_propogate(x)
        return out[-1]
                        
        
        
    
'''
A = neuralnet( 3, [2,2], 3)
c = A.neural ( [[0,0,0,1]], [[0,0,1]], 1, 0.1, 100)
print (c)
'''
