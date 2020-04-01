#!/usr/bin/env python
# coding: utf-8

# In[19]:


#import necessary packages

import numpy as np
import math
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.special import erf


# In[26]:


# Black scholes exact formuale calculation

def normal_distr(x, mean, sigma):
    
    return (1/sigma*np.sqrt(2*np.pi)) * np.exp((-(x- mean)**2)/ (2*(sigma**2)))
    
def Black_scholes(S, K, r, T, sigma):
        
    d1_num = np.log(S/K) + (r+0.5*(sigma**2))*(T) # Calculate d1
    
    d1_den = sigma*T
    
    d1 = d1_num/d1_den
    d2 = d1 - sigma*(np.sqrt(T)) # calculate d2
    
    call_option = S*erf(d1) - erf(d2)*K*np.exp(-r * (T)) #calculate the call price using d1 & d2
    
    return call_option
    


# In[4]:


data = pd.read_csv("options_trade.csv")
data = data[['quote_datetime', 'expiration','strike', 'option_type','underlying_bid']]
data["value"]=  (data["underlying_bid"]-data["strike"]).apply(lambda x: max(x,0))
data.drop(data[data['value']==0].index, axis=0, inplace=True)
data['quote_datetime'] = pd.to_datetime(data['quote_datetime'])
data['expiration'] = pd.to_datetime(data['expiration'])
data['time diff'] = (data['expiration'] - data['quote_datetime']).dt.days
data['rate'] = data['time diff'].apply(lambda x: np.random.random()/10)


# In[5]:


import scipy.stats as si

def newton_vol_call(S, K, T, C, r, sigma):
    #S: spot price
    #K: strike price
    #T: time to maturity
    #C: Call value
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1   
    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - C) / vega
        
        return abs(xnew)


# In[6]:


data.drop(data[data['underlying_bid']==0].index, axis=0, inplace=True)
data = data[data.option_type=="C"]
data.drop(['quote_datetime', 'expiration', "option_type"], axis =1, inplace=True)
sigma = 0.01
data["volatility"]= newton_vol_call(data['underlying_bid'],data['strike'],data['time diff'],data['value'],
                                                          data['rate'],sigma)


# In[8]:


def generator(Data, n):
    data = copy.deepcopy(Data)
    new_data = []
    slice = 0
    while len(new_data) < n:
        R = list(range(len(data)))
        np.random.shuffle(R)
        data = data.iloc[R]
        i,j = np.random.choice(range(len(data)), size=2)
        list_ = []
        for col in data.columns:
            if col == 'time diff':
                list_.append((data[col].iloc[i]+data[col].iloc[j])//2)
            else:
                list_.append((data[col].iloc[i]+data[col].iloc[j])/2)
        new_data.append(list_)
        if len(new_data)%10 == 0:
            data_to_add = pd.DataFrame(new_data[10*slice:], columns=data.columns)
            data = pd.concat([data,data_to_add], axis = 0)
            slice += 1
        
    return pd.DataFrame(new_data, columns=data.columns)
                


# In[14]:


new_data = pd.concat([data,Big_data], axis = 0)
new_data.to_csv('optionData.csv', index = False)


# # MODELS

# In[6]:


# library & dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[7]:


df = pd.read_csv("optionData.csv")


# # BLACK SCHOLES MODEL

# In[18]:


df["BSValue"]=  Black_scholes(df['underlying_bid'].values, df['strike'].values, df['rate'].values,
                                                df['time diff'].values, df['volatility'].values)
df["diff"] = np.abs(df["value"]-df["BSValue"])


# # MLP  Network for Option Pricing

# In[5]:



X = df[['strike', 'underlying_bid','time diff', 'rate', 'volatility']]
y = df["value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[6]:


import torch
import torch.nn as nn


# In[7]:


class Feedforward(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.sigmoid = nn.Sigmoid()
            self.fc2 = nn.Linear(self.hidden_size, 1)
            self.relu = nn.ReLU()
            
            
        def forward(self, x):
            hidden = self.fc1(x)
            sigmoid = self.sigmoid(hidden)
            output = self.fc2(sigmoid)
            output = self.relu(output)
            return output


# In[8]:


model = Feedforward(5,3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

Inputs =torch.from_numpy(X_train.values).float()
Target = torch.from_numpy(y_train.values).float()
batch_size = 64
Loss = []
Num_epoch = 50
for i in range(Num_epoch):
    los = 0.0
    for batch_idx in range(len(Inputs)//batch_size):
        data, target = Inputs.narrow(dim=0, start = batch_idx*batch_size, length = batch_size), Target.narrow(dim=0, start = batch_idx*batch_size, length = batch_size).view(-1,1)
        prediction = model(data)
        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        los += loss.data.numpy()
#     if i%10 == 0:
#         print(loss.data.numpy())
    
    if i % 5 == 0 and i>0:
        Loss.append(los)
        plt.cla()
#         Prediction = model(Inputs)
#         loss = criterion(Prediction, Target)
       # plot and show learning process
    
        plt.plot(range(len(target)), target.data.numpy())
        plt.plot(range(len(target)), prediction.data.numpy(), 'r-', lw=2)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
        plt.legend(['target', 'prediction'])
        plt.xlabel('Range of batch size', fontsize=20)
        plt.ylabel('Target, prediction',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.pause(0.1)
        plt.savefig('targetpr.jpg',bbox_inches='tight', dpi=150)


plt.show()

#plt.savefig('tagertpred.png')

    


# In[240]:


plt.plot(range(len(Loss)), Loss)
plt.title("Error Curve", fontsize=20)
plt.xlabel("Iterations", fontsize=20)
plt.ylabel("Batch Loss", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



# In[158]:


X_test = torch.from_numpy(X_test.values).float()
y_test = torch.from_numpy(y_test.values).float().view(-1,1)


# In[159]:


prediction = model(X_test)
loss = criterion(prediction,y_test)


# In[160]:


plt.plot(range(len(y_test)), y_test.data.numpy())
plt.plot(range(len(y_test)), prediction.data.numpy(), 'r-', lw=2)
plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
plt.legend(['target', 'prediction'])
plt.xlabel('Range of batch size')
plt.ylabel('Target, prediction')


# In[ ]:


# number of epochs to train the model
n_epochs = 30  # suggest training between 20-50 epochs

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    #for data, target in 
    
    ###################
    # train the model #
    ###################
    for data, target in :
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss


# ## PAIRED SAMPLE TEST

# In[20]:


from scipy.stats import ttest_ind
stat, p = ttest_ind(data['underlying_bid'],  Big_data['underlying_bid'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')


# ### $H_{0} : \mu _{1} == 0$
# ### $H_{1} : \mu _{1} \neq 0$
# Given the pvalue > 0.05 one can then fail to reject the null hypothesis and conclude that truly the two data sets have a $\mu == 0$
# 
