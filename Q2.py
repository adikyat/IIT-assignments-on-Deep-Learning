#!/usr/bin/env python
# coding: utf-8

# In[87]:



import numpy as np
import matplotlib.pyplot as plt
import math
import random


# # Q1 e)

# # I assumed (8000,13) as samples and features of the dataset because the execution time with whole original dataset was very high.So, I undersampled the data using RandomUnderSampler function and deleted the features which have more than 10000 zero values because these features had less significant contribution to the model.
# 

# # I also decreased the number of neurons and mini batch size according to the number of features available after preprocessing.

# In[158]:


class Neural_Network():
    def __init__(self, neurons, hidden_acti, output_acti): # arguments: an array "neurons" consist of number of neurons for each layer, activation function to be used in hidden layers and activation function to be used in output layer
        self.inputSize = neurons[0] # Number of neurons in input layer
        self.outputSize = neurons[-1] # Number of neurons in output layer
        self.layers = len(neurons)
        self.w = []
        for i in range(len(neurons)-1): 
            self.w.append(np.random.rand(neurons[i],neurons[i+1])) #weight matrix between layer i and layer i+1
        self.w1=self.w[0]
       # print(self.w[0].shape)
        self.w2=self.w[1]
       # print(self.w[1].shape)
        self.w3=self.w[2]
       # print(self.w[2].shape)
        self.w4=self.w[3]
      #  print(self.w[3].shape)
        
        self.activationHidden = None # Activation funtion to be used in hidden layers
        self.activationOutput = None # Activation funtion to be used in output layer
        self.activationHiddenPrime = None # Derivative of the activation funtion to be used in hidden layers
        self.activationOutputPrime = None # Derivative of the activation funtion to be used in output layer
        
        if(hidden_acti == "sigmoid"):
            self.activationHidden = self.sigmoid
            self.activationHiddenPrime = self.sigmoidPrime
        else:
            self.activationHidden = self.linear
            self.activationHiddenPrime = self.linearPrime
            
        if(output_acti == "sigmoid"):
            self.activationOutput = self.sigmoid
            self.activationOutputPrime = self.sigmoidPrime
        else:
            self.activationOutput = self.linear
            self.activationOutputPrime = self.linearPrime
            
    def sigmoid(self, s): # sigmoid activation function
        return(1/(1+np.exp(-s)))
    
    def sigmoidPrime(self,x): # derivative sigmoid activation function
        return(self.sigmoid(x)*(1-self.sigmoid(x)))
    
    def linear(self, s): # Linear activation function
        return(s)
    
    def linearPrime(self,x): # derivative of linear activation function
        return(np.ones(len(x)))

    
    def forward(self, x): # function of forward pass which will receive input and give the output of final layer
        # Write the forward pass using the weights to find the predicted value and return it.
        Z1 = np.dot(self.w1.T,x)
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.w2.T,A1)
        A2 = self.sigmoid(Z2)
        Z3 = np.dot(self.w3.T,A2)
        
        A3 = self.sigmoid(Z3)
        Z4 = np.dot(self.w4.T,A3)
        A4 = self.sigmoid(Z4)
        self.temp = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
            "Z3": Z3,
            "A3": A3,
            "Z4": Z4,
            "A4": A4
        }
        return(A4)
       
            
    def backward(self, x, y, o,loss_func): # find the loss and return derivative of loss w.r.t every parameter
        # Write the backpropagation algorithm here to find the gradients and return it.
        if loss_func=="MSE":
            L = sum((o-y)**2)#MSE LOSS FUNC
            dL = (2*(o-y))
        else:
            L= -np.sum(np.multiply(np.log(o),y))#CE loss
            dL=-(y/o)
        
        
        Z1 = self.temp["Z1"]
        A1 = self.temp["A1"]
        Z2 = self.temp["Z2"]
        A2 = self.temp["A2"]
        Z3 = self.temp["Z3"]
        A3 = self.temp["A3"]
        Z4 = self.temp["Z4"]
        A4 = self.temp["A4"]
        #FOR last layer:
        
        
        
        dA4_Z4 = self.sigmoidPrime(Z4)
        dZ4_w4 = A3
       
        dL_Z4= np.dot(dL,dA4_Z4)
        #print("dL_Z4"+str(dL_Z4.shape))
       # print(dA4_Z4)
        dw4 = np.dot(dL_Z4,dZ4_w4)
        #print("dw4"+str(dw4.shape))
       
     #  dL_Z4 will be fed back to 3rd layer:
        
        dZ4_A3 = self.w4
        dA3_Z3 = self.sigmoidPrime(Z3).reshape(neurons[3],1)#*****************************
        dZ3_w3 = A2.reshape(neurons[2],1)#500,1)*************************
        dz4a3 =np.dot(dZ4_A3.T,dA3_Z3)
        #print(dz4a3.shape)
        #print(dL_Z4.shape)
        dL_Z3= np.dot(dL_Z4,dz4a3)
       # print("dL_Z3"+str(dL_Z3.shape))      

        dw3 = np.dot(dL_Z3,dZ3_w3.T)
        #print("dw3"+str(dw3.shape))
    # dL_Z3 will be fed back to 2nd layer:
        dZ3_A2 = self.w3
        dA2_Z2 = self.sigmoidPrime(Z2).reshape(neurons[2],1)#*(500,1)***************************
       # print("da2z2 "+str(dA2_Z2.shape))
       # print("dZ3_A2 "+str(dZ3_A2.shape))
        dZ2_w2 = A1.reshape(neurons[1],1)#***********************
        dz3a2=np.dot(dZ3_A2.T,dA2_Z2)
     #   print("dz3a2 "+str(dz3a2.shape))
       # print("dlz3 "+str(dL_Z3.shape))
        dL_Z2= np.dot(dL_Z3,dz3a2.T)
        #print("dL_Z2"+str(dL_Z2.shape))
        dw2 = np.dot(dL_Z2,dZ2_w2)
       # print("dw2"+str(dw2.shape))
    
    # dL_Z2 will be fed back to 1st layer:
        dZ2_A1 = self.w2
        dA1_Z1 = self.sigmoidPrime(Z1).reshape(neurons[1],1)#***********************************
        dZ1_w1 = x# x shud be 1 sample error will come...
       # print("dZ1_w1 "+str(dZ1_w1[0].shape))
        
        dz2a1 = np.dot(dZ2_A1.T,dA1_Z1) #.T(500,1)
        #print("dz2a1 "+str(dz2a1.T.shape))
        #print("dlz2 "+str(dL_Z2.T.shape))
        dL_Z1 = np.dot(dL_Z2.T,dz2a1.T)
      #  print("dZ1_w1[i]"+str(dZ1_w1[0]))
        dw1=[]
        for i in range(x.shape[0]):
            
            dw1.append(np.dot(dL_Z1,dZ1_w1[i]))###
        #dw1=np.asarray(dw1)
        
        #dw1=np.ravel(dw1)
       # print("dw1"+str(dw1))
        # dL_Z1 would have been fed back if there were any more layers
        
        grads = {
            "dw4": dw4,
            "dw3": dw3,
            "dw2": dw2,
            "dw1": dw1,
           # "dw1_2": dw1_2
        }
        return(L, grads)
        
        
        
    def update_parameters(self, grads, learning_rate): # update the parameters using the gradients
        # update each parameter using the gradients and the learning rate
        dw1 = grads["dw1"]
        dw2 = grads["dw2"]
        dw3 = grads["dw3"]
        dw4 = grads["dw4"]
        #self.w1.reshape(784,300,1)
       # self.w1 = np.expand_dims(self.w1, axis=2)
       # self.w1=self.w1.T
       # dw1=dw1.T
        #print(dw1[0][200][0])
        #print(self.w1[0][200])
        self.w1=np.subtract(self.w1,np.multiply(learning_rate,dw1)[:,:,0])
        #for i in range(len(dw1)-1):#dw1.shape[1]
           
            
            #self.w1[i][0] -= learning_rate*dw1[i][0][0]#.T  self.[0]
        self.w2 -= learning_rate*dw2
        self.w3 -= learning_rate*dw3.T
        dw4=dw4.reshape(neurons[3],1)#************************************
        
       # print(dw4.shape)
        
       # print(self.w4.shape)
        dww4=learning_rate*dw4
      #  print(dww4.shape)
        self.w4 -= dww4
        dw1,dw2,dw3,dw4=0,0,0,0 #reset the gradient
                     
    def train(self, X, Y,loss_func): # receive the full training data set
        lr = 1e-6        # learning rate
        epochs = 50  # number of epochs
        batch_size=5#50*************************************************4,5,6,7,8
        points_seen=0
        self.losss = []
        k=0
        index = np.array([ i for i in range(len(X))])
        random.shuffle(index)
        for e in range(epochs):
            loss = 0.0
            k+=1
            for q in index:
           # for (x_batch,y_batch) in mini_batch(X,Y):
               
               # out = self.forward(x_batch) # call of forward pass to get the predicted value
                out = self.forward(X[q])
                #los,grads = self.backward(x_batch, y_batch, out,loss_func) # find the gradients using backward pass
                los,grads = self.backward(X[q], Y[q], out,loss_func)
                points_seen+=1
                if points_seen % batch_size ==0:
                    
                    self.update_parameters(grads, lr)
                    loss += los
            if(k==5):#for every 5 epochs
                k=0
                self.losss.append(loss/len(X))
            
        
    def predict(self,x):
        print ("Input : \n" + str(x))
        print ("Output: \n" + str((self.forward(x))))
        
    def plot_MSE(self):
        plt.plot(self.losss,'r',label='MSE loss')
        plt.legend()
        plt.title("Feed forward ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    def plot_CE(self):
        plt.plot(self.losss,'b',label='CE loss')
        plt.legend()
        plt.title("Feed forward ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

   


# # Generating some training data

# In[159]:


import pandas as pd
from collections import Counter
df = pd.read_csv("C:/Users/Aditya Kyatham/Documents/IE 643 Deep Learning/Assignment3_train_data.csv", index_col=None)
#inputsize = 3
Y = df.pop("label").values
X = df.values
print(X.shape)
print(sorted(Counter(Y).items()))


# In[ ]:





# In[160]:



arr=[]
i=0
for i in range(len(X[1])):
    num=Counter(X[:,i])
    n=num.get(0)
    #print(n)
    
    if n> 10000:
        arr.append(i) 
X=np.delete(X,arr,1)


# In[161]:


#Counter(X[:,603])
X.shape


# In[162]:


from imblearn.under_sampling import RandomUnderSampler
und = RandomUnderSampler(ratio={1:2000,2:2000,3:2000,4:2000})
X,Y = und.fit_resample(X, Y)
print(sorted(Counter(Y).items()))

        


# In[163]:



X.shape


# In[164]:


from sklearn import preprocessing #normalization
mm_scaler = preprocessing.MinMaxScaler()
Y=Y.reshape(-1, 1)
X= mm_scaler.fit_transform(X)
Y= mm_scaler.fit_transform(Y)


# In[ ]:





# In[165]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#random for splitting same data when run again.


# In[166]:


from sklearn.model_selection import train_test_split
X_train_tr, X_val, y_train_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)#random for splitting same data when run again.
#X_val and Y_val are validation sets.


# In[167]:


print(X_train.shape[1])


# # Defining the network

# In[168]:


#D_in is input dimension
# H1 is dimension of first hidden layer 
# H2 is dimension of first hidden layer
#D_out is output dimension.
D_in, H1, H2,H3,D_out =X_train.shape[1] , 3, 5, 3,1#300, 500, 300*******************************************

neurons = [D_in, H1, H2,H3, D_out] # list of number of neurons in the layers sequentially.
print(neurons[-1])
Hidden_activation = "sigmoid"   # activation function of the hidden layers.
Output_activation = "sigmoid"  # activation function of the output layer.
test = Neural_Network(neurons, Hidden_activation, Output_activation )


# In[ ]:





#  Training the network

# In[84]:


test.train(X_train_tr,y_train_tr,"MSE")


# In[ ]:





# In[ ]:





# In[169]:


test.train(X_train_tr,y_train_tr,"CE")


# # Prediction for a data point after the training

# In[170]:


test.predict(X_val.T)


# In[ ]:





# In[171]:


test.plot_CE()


# In[86]:


test.plot_MSE()


# In[ ]:





# In[ ]:




