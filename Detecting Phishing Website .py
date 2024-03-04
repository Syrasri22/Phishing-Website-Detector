#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install torch torchvision


# In[3]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option("display.max_columns",None)
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

import os
for dirname,_,filenames in os.walk("/LENOVO/Desktop"):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# In[5]:


df=pd.read_csv("C:/Users/LENOVO/Desktop/cv/dataset_phishing.csv")


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df=df.drop(labels="url",axis=1)
df.head()


# In[9]:


object_features=[col for col in df.columns if df[col].dtype=="O"]
print(object_features)


# In[10]:


df['status'].value_counts()


# In[11]:


with plt.style.context(style="bmh"):
    fig=df['status'].value_counts().plot.bar(figsize=(6,5),
                                             fontsize=15,
                                             title='Analysing status feature using bar-chart',
                                            xlabel='class labels',
                                            ylabel='number of records')
    plt.show()


# In[15]:


with plt.style.context(style="fivethirtyeight"):
    plt.pie(x=dict(df['status'].value_counts()).values(),
           labels=dict(df['status'].value_counts()).keys(),
           autopct="%.2f%%",
           colors=['red','violet'],
           startangle=90,
           explode=[0,0.05])
    centre_circle=plt.Circle((0,0),0.70,fc='white')
    fig=plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(label="Analysing status feature using donut-chart")
    plt.show()


# In[16]:


df.head()


# In[17]:


class_labels=df['status'].unique().tolist()
class_labels.sort()
print(class_labels)


# In[18]:


class_dict={}
for idx,label in enumerate(class_labels):
    class_dict[label]=idx
print(class_dict)


# In[19]:


df['status']=df['status'].map(class_dict)
df.head()


# In[20]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1:]


# In[21]:


X.head()


# In[22]:


y.head()


# In[23]:


scaler=MinMaxScaler()
scaler.fit(X.values)
X_scaled=scaler.transform(X.values)
print(X_scaled)


# In[26]:


import pickle
with open(file="C:/Users/LENOVO/Desktop/cv/dataset_phishing.csv",mode="wb") as file:
    pickle.dump(obj=scaler,file=file)


# In[27]:


new_X=pd.DataFrame(data=X_scaled,columns=X.columns)
new_X.head()


# In[28]:


X_train,X_test,y_train,y_test=train_test_split(new_X,y,test_size=0.2,random_state=42,shuffle=True,stratify=y)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[29]:


X_train.head()


# In[30]:


X_test.head()


# In[31]:


y_train.head()


# In[32]:


y_test.head()


# In[33]:


train_input_tensor=torch.from_numpy(X_train.values).float()
train_label_tensor=torch.from_numpy(y_train['status'].values).float()
val_input_tensor=torch.from_numpy(X_test.values).float()
val_label_tensor=torch.from_numpy(y_test['status'].values).float()


# In[34]:


train_input_tensor


# In[35]:


train_label_tensor=train_label_tensor.unsqueeze(1)
train_label_tensor


# In[36]:


val_input_tensor


# In[37]:


val_label_tensor=val_label_tensor.unsqueeze(1)
val_label_tensor


# In[38]:


train_dataset=TensorDataset(train_input_tensor,train_label_tensor)
val_dataset=TensorDataset(val_input_tensor,val_label_tensor)


# In[39]:


train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(dataset=val_dataset,batch_size=32,shuffle=True)


# In[40]:


print(f"number of batches in train_loader: {len(train_loader)}")
print(f"number of records in train_loader: {len(train_loader.dataset)}")
print(f"number of batches in val_loader: {len(val_loader)}")
print(f"number of records in val_loader: {len(val_loader.dataset)}")


# In[41]:


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[42]:


class MLP(nn.Module):
    def __init__(self,dropout=0.4):
        super(MLP,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(in_features=87,out_features=300), # in_feature value is 87. because this dataset has 87 independent features
            nn.ReLU(),
            nn.BatchNorm1d(num_features=300),
            nn.Dropout(p=dropout),
            
            nn.Linear(in_features=300,out_features=100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100),
            
            nn.Linear(in_features=100,out_features=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.network(x)
        return x


# In[43]:


model=MLP(dropout=0.4)
print(model)


# In[44]:


optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)
criterion=nn.BCELoss()


# In[47]:


def train_loop(model,train_loader,val_loader,device,optimizer,criterion,batch_size,epochs):
    model=model.to(device)
    train_batch_size=len(train_loader)
    val_batch_size=len(val_loader)
    
    history={"train_accuracy":[],"train_loss":[],"val_accuracy":[],"val_loss":[]}
    
    for epoch in range(epochs):
        model.train() # training mode
        
        train_accuracy=0
        train_loss=0
        val_accuracy=0
        val_loss=0
        
        for X,y in train_loader:
            X=X.to(device)
            y=y.to(device)
            
            # forward propagation
            outputs=model(X)
            pred=torch.round(outputs)
            loss=criterion(outputs,y)
            
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_train_loss=loss.item()
            cur_train_accuracy=(pred==y).sum().item()/batch_size
            
            train_accuracy+=cur_train_accuracy
            train_loss+=cur_train_loss
        model.eval()
        with torch.no_grad():
            for X,y in val_loader:
                X=X.to(device)
                y=y.to(device)
                outputs=model(X)
                pred=torch.round(outputs)
                
                loss=criterion(outputs,y)
                
                cur_val_loss=loss.item()
                cur_val_accuracy=(pred==y).sum().item()/batch_size
                
                val_accuracy+=cur_val_accuracy
                val_loss+=cur_val_loss
        train_accuracy=train_accuracy/train_batch_size
        train_loss=train_loss/train_batch_size
        val_accuracy=val_accuracy/val_batch_size
        val_loss=val_loss/val_batch_size
        
        print(f"[{epoch+1:>3d}/{epochs:>3d}], train_accuracy:{train_accuracy:>5f}, train_loss:{train_loss:>5f}, val_accuracy:{val_accuracy:>5f}, val_loss:{val_loss:>5f}")
        history['train_accuracy'].append(train_accuracy)
        history['train_loss'].append(train_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_loss'].append(val_loss)
    PATH="C:/Users/LENOVO/Desktop/cv/dataset_phishing.csv"
    torch.save(model.state_dict(),PATH)
    return history
            


# In[48]:


history=train_loop(model,train_loader,val_loader,device,optimizer,criterion,batch_size=32,epochs=100)


# In[53]:


with plt.style.context(style="fivethirtyeight"):
    plt.figure(figsize=(18,8))
    plt.plot(history['train_accuracy'],label="train accuracy")
    plt.plot(history['val_accuracy'],label="val accuracy")
    plt.title(label="Accuracy plots")
    plt.xlabel(xlabel='epochs')
    plt.ylabel(ylabel='accuracy')
    plt.show()
    
    plt.figure(figsize=(18,8))
    plt.plot(history['train_loss'],label="train loss")
    plt.plot(history['val_loss'],label="val loss")
    plt.title(label="loss plots")
    plt.xlabel(xlabel='epochs')
    plt.ylabel(ylabel='loss')
    plt.show()


# In[ ]:




