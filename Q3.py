#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import torch
import numpy as np


# In[2]:


import pandas as pd
from collections import Counter
####do this for private###
df = pd.read_csv("public_data.csv", index_col=None)
#inputsize = 3
Y = df.pop("label").values
X = df.values
print(X.shape)
print(sorted(Counter(Y).items()))


# In[ ]:





# In[3]:


arr=[]
i=0
for i in range(len(X[1])):
    num=Counter(X[:,i])
    n=num.get(0)
    #print(num)
    
    if n > 10000:
        arr.append(i) 
X=np.delete(X,arr,1)


# In[4]:


from imblearn.under_sampling import RandomUnderSampler
und = RandomUnderSampler(ratio={1:2000,2:2000,3:2000,4:2000})
X,Y = und.fit_resample(X, Y)
print(sorted(Counter(Y).items()))

        


# In[5]:


####do this for private######
from sklearn import preprocessing #normalization
mm_scaler = preprocessing.MinMaxScaler()
Y=Y.reshape(-1, 1)
X= mm_scaler.fit_transform(X)
Y= mm_scaler.fit_transform(Y)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#random for splitting same data when run again.


# In[7]:


###give your private test data#####
inp=input("Enter the path to private test data")
df_private = pd.read_csv(inp, index_col=None)
Y_private = df_private.pop("label").values
X_private = df_private.values
print(X_private.shape[0])


# In[8]:


print(X_private.shape[1])
print(X.shape[1])


# In[9]:


####IF THE PRIVATE DATA IS VERY LARGE THEN WE SHOULD UNDERSAMPLE IT OR DELETE THE INSIGNIFICANT FEATURES FOR MODEL######

X_private=np.delete(X_private,arr,1)
print(X_private.shape[1])
#####taking 200 samples in proportion with the training data#####
#####IF THE PRIVATE DATA IS NOT RELATED TO  Assignment3_train_data.csv then CHANGE THE RandomUnderSampler ratioS.
if Y_private.shape[0]>200:
    from imblearn.under_sampling import RandomUnderSampler
    und = RandomUnderSampler(ratio={1:200,2:200,3:200,4:200})
    print(X_private.shape[0])
    X_private,Y_private = und.fit_resample(X_private, Y_private)
    print(sorted(Counter(Y_private).items()))



from sklearn import preprocessing #normalization
mm_scaler_private = preprocessing.MinMaxScaler()
Y_private=Y_private.reshape(-1, 1)
X_private= mm_scaler.fit_transform(X_private)
Y_private= mm_scaler.fit_transform(Y_private)
print(X_private.shape[1])
print(X.shape[1])


# In[10]:


from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader


# In[11]:


x_tensor_train = torch.from_numpy(X_train).float()
y_tensor_train = torch.from_numpy(y_train).float()
x_tensor_test = torch.from_numpy(X_test).float()
y_tensor_test = torch.from_numpy(y_test).float()

x_tensor_private = torch.from_numpy(X_private).float()
y_tensor_private = torch.from_numpy(Y_private).float()

train_dataset = TensorDataset(x_tensor_train, y_tensor_train)
test_dataset = TensorDataset(x_tensor_test, y_tensor_test)
private_dataset = TensorDataset(x_tensor_private, y_tensor_private)

train_loader = DataLoader(dataset=train_dataset, batch_size=8)
test_loader = DataLoader(dataset=test_dataset, batch_size=8)
private_loader = DataLoader(dataset=private_dataset, batch_size=8)



# In[16]:


def train_step(model, loss_fn, optimizer,x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        y_pred = model(x)
        # Computes loss
        loss = loss_fn(y, y_pred)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()



# In[ ]:


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H1,H2,H3, D_out = X_train.shape[1], 3, 5, 3, 1#///////////////

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1, H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, H3),
    torch.nn.ReLU(),
    torch.nn.Linear(H3, D_out),
    torch.nn.Sigmoid()
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
#loss_fn = torch.nn.MSELoss(reduction='sum')


loss_fn = torch.nn.MSELoss()

learning_rate = 0.01#/////////////////
optimizer = torch.optim.SGD(model.parameters(), learning_rate)




# Creates the train_step function for our model, loss function and optimizer
#train_step = make_train_step(model, loss_fn, optimizer)
losses = []
val_losses = []
private_losses = []

n_epochs=500#///////////////
# For each epoch...
acc_val=[]
for epoch in range(n_epochs):
    # Performs one train step and returns the corresponding loss
    for x_batch, y_batch in train_loader:
        
        loss = train_step(model, loss_fn, optimizer,x_batch, y_batch)#/////////////
        losses.append(loss)
    
    
    with torch.no_grad():
        for x_val, y_val in test_loader:
            model.eval()
            yhat = model(x_val)
            val_loss = loss_fn(y_val, yhat)
            
            val_losses.append(val_loss.item())
        #Accuracy
        yhat = (yhat>0.5).float()
        correct = (yhat == y_val).float().sum()
        acc_val.append((correct/yhat.shape[0])*100)
        #print("Validation Accuracy: {:.3f}".format((correct/yhat.shape[0])*100))
print("Average Validation Accuracy: {:.3f}".format(sum(acc_val)/n_epochs))



# In[50]:


acc_private=[]
for epoch in range(n_epochs):
##for private data##
    with torch.no_grad():
        for x_pri, y_pri in private_loader:
            model.eval()
            yhat_private = model(x_pri)
            private_loss = loss_fn(y_pri, yhat_private)
            
            private_losses.append(private_loss.item())
        #Accuracy
        yhat_private = (yhat_private>0.5).float()
        correct_private = (yhat_private == y_pri).float().sum()
        acc_private.append((correct_private/yhat_private.shape[0])*100)
       # print("Validation Accuracy: {:.3f}".format((correct_private/yhat_private.shape[0])*100))
print("Average Private Accuracy: {:.3f}".format(sum(acc_private)/n_epochs))


# In[ ]:




