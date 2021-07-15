#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Team BugML
# Akarsh Jain (cs1200318@iitd.ac.in) (2020CS10318)
# Kushagra Rode (2020CS10354)
# Atharv Dabli (2020CS10328)

# all the code tried for implementing and its variations have been left of as comments for the reader so as to enable
# them to review the problem solving strategy used and also as a mark of original work


#about the file - task 1 of the project has been completed with 95% accuracy (approx)  
#implementation details given as comments
#Task 2 : unable to complete successfully - details of the attempt are as follows and also given at last as comments
# tried to implement three CNN's each for prefix,postfix and infix (these are classified by earlier CNN)
# did not work and the network seemed to be taking out the average of the labels(values) and outputing this for each image
# tried varying different hyper-parameters for 2-3 days wide range of values but to no use 
# this error seemed to be due to the less informative labels 
# Hence then tried a K means algorithm to classify *,/,+,- which also did not work (tried to divide in 4 clusters)

#importing various libraries
import numpy as np
import os
import random
import cv2
from os import listdir
from os.path import isfile,join
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#----------------showing an example image------------
from numpy import asarray
from matplotlib import image
image=image.imread("./SoML-50/data/2.jpg")
plt.imshow(image)
plt.show()
#---------------------------------------------------
#---------------reading the csv file for first task----------
y_label=pd.read_csv("./SoML-50/annotations.csv")

notation={"Label":{"infix":0,"prefix":1,"postfix":2}}

y_label=y_label.replace(notation)
y_label=y_label.to_numpy()
y_label=y_label[0:50000,1].reshape(50000,1)
y_label=y_label.astype(float)
#------------------------------------------------------
# - ---------- non useful comment for reader --------------
# # y_label=y_label.astype(int)
# # y_label=y_label.reshape(1,1000)
# # rows = np.arange(y_label.size, dtype=int)
# # one_hot = np.zeros((1000,3),dtype=int)
# # one_hot[rows, y_label] = 1
# # y_label=one_hot
# ---------------------------------------------------

def load_images(folder):         #loads 100 images with corresponding labels at a time
    train_data=np.zeros((100,128,384))
    target_label=np.zeros((100,1))
    for i in range(100):
        filename=random.choice(os.listdir(folder))
        while filename=='.DS_Store':
            filename=random.choice(os.listdir(folder))
        target_label[i]=y_label[int(filename[0:-4])-1]
        img =Image.open(os.path.join(folder,filename))
        if img is not None:
            data=asarray(img)
            train_data[i,:,:]=data
    train_data=torch.from_numpy(train_data)
    target_label=torch.from_numpy(target_label)
    train_data.float()
    target_label.float()
    return train_data,target_label

# ---------------different implementation but had to be discarded due to circumstances --------
#-------------------------------------------------------------------------------------------
# def load_images(folder):
#     train_data=np.zeros((10,128,384))
#     test_data=np.zeros((10,128,384))
#     i=0
#     for filename in os.listdir(folder):
#         if filename=='.DS_Store':
#             continue
#         if i>=20:
#             break
#         #img=cv2.imread(os.path.join(folder,filename),-1)
#         img =Image.open(os.path.join(folder,filename))
#         if img is not None:
#             #_,thresh=cv2.threshgold
#             #cv2.imshow('image',img)
#             data=asarray(img)
#             if i<10:
#                 train_data[i,:,:]=data
# #             if 10000<=i<20000:
# #                 train_data[1,i,:,:]=data
# #             if 20000<=i<30000:
# #                 train_data[2,i,:,:]=data
# #             if 30000<=i<40000:
# #                 train_data[3,i,:,:]=data
#             else:
#                 test_data[i-10,:,:]=data
#             i+=1
#     return train_data,test_data
#x_train,x_test=load_images("./SoML-50/data/")
# x_train=torch.from_numpy(x_train)
# y_train=torch.from_numpy(y_train)
#x_test=torch.from_numpy(x_test)
# x_train=x_train.float()
# y_train=y_train.float()
#x_test=x_test.float()
#print(x_test.size())
#-------------------------------------------------------------------
#-------------------------------------------------------------------


x_train,y_train=load_images("./SoML-50/data/")
print(x_train.size())
print(y_train.size())


# In[3]:


#----------------------non-useful------------------
# y_label=y_label.to_numpy()
# y_train=y_label[0:1000,1].reshape(1000,1)
# y_test=y_label[1000:2000,1].reshape(1000,1)
# y_train=y_train.astype(float)
# y_test=y_test.astype(float)
# y_train=torch.from_numpy(y_train)
# y_test=torch.from_numpy(y_test)
# y_train=y_train.float()
# y_test=y_test.float()
#----------------------------------------------------


# In[3]:


#--------------------- different strategy to load image data -------------
#from torchvision import datasets, transforms
#dataset = datasets.ImageFolder("./SoML-50/data/", transform=transforms)
#x=np.load("./SoML-50/data/1.jpg",allow_pickle=True)
#-----------------------------------------------------------


import cv2


# In[20]:


#---------------CNN has been defined-------------------

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,13))
        self.conv2 = nn.Conv2d(6, 16, (5,13))
        self.conv3 = nn.Conv2d(16, 20, (5,13))
        self.fc1 = nn.Linear(5*15*20, 480)  #15*20 from image dimension
        self.fc2 = nn.Linear(480, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10,3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (5,5))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=1) #note the softmaxx activation
        return x.float() 

net = Net()
print(net)


# In[21]:


#x_test=torch.reshape(x_test.float(),(20,1,128,384)) 

#--------------------checking whether the net is working properly ------------
x_train=torch.reshape(x_train.float(),(100,1,128,384))
x_train/=255

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

output=net(x_train.float())
print(output.size())


# In[22]:


target=y_train.type(torch.LongTensor)
target=torch.reshape(target,(100,))
print(target.size())
#-----------------------------------------------------------
# --------------loss caluculation cross entrpy loss---------------
criterion = nn.CrossEntropyLoss()

print(output.size())
loss = criterion(output, target)
print(loss)
#--------------------------------------------------------------------


# In[23]:


#------------------unique way to reduce bias in the network which was not working properly without this------------
#the network was not working as it was stuck in a local optima at around loss=1.082..
#this is to reduce bias in the problem optimally while reducing the learning rates gradually
#also if for 9 iterations the loss doesn't change then we must change the input to the other batch becuase the code is stuck in local optima
#though the bias is also not reduced so much as to lead to the problem of high variance hence as soon as loss<0.9 break statement is included
# due to naiveness of our method this does face some problems in some cases of intitial traning sets and hence in such cases model must be loaded again

import torch.optim as optim

lr=2 #learning rate
cheating=0 #parameter to check whether it is stuck in a local optima 
while loss>0.95:
    lr=2
    cheating=0
    x_train,y_train=load_images("./SoML-50/data/") #change of input if stuck in local minima
    x_train=torch.reshape(x_train.float(),(100,1,128,384))
    x_train/=255
    target=y_train.type(torch.LongTensor)
    target=torch.reshape(target,(100,))
    prev=0   #loss in previous iteration to track if the loss is changing or not
    while lr>0.07:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.1)         #stochastic gradient descent 
        if cheating>9:
            break             #local optima
        for aux in range(16):
            # going through the same batch of 100 images 16 times
            
            optimizer.zero_grad()
            output=net(x_train.float())
            target=y_train.type(torch.LongTensor)
            target=torch.reshape(target,(100,))
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            print(loss)
            if loss==prev:
                cheating+=1
            if loss!=prev:
                cheating=0
            prev=loss
            if loss>4 or loss<0.95:
                break
        if loss>4 or loss<0.8:
            break
        lr/=2            #reduces learning rate gradually
        print(loss)
        
# ----------- not useful----------------
# for aux in range(30):
#     optimizer.zero_grad()
#     output=net(x_train.float())
#     target=y_train.type(torch.LongTensor)
#     target=torch.reshape(target,(100,))
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(output, target)
#     loss.backward()
#     optimizer.step()
#     print(loss)
#-------------------------------

output=net(x_train.float())
target=y_train.type(torch.LongTensor)
target=torch.reshape(target,(100,))

criterion = nn.CrossEntropyLoss()

loss = criterion(output, target)
print(loss)
print(target,output)

#------------------------------------------------


# In[24]:


losses=[]
alpha=0.0008 #learning rate
iter=30  #number of times to go through the same batch
for epochs in range(200): #200 sets of 100 images
    if loss>4:
        print("sed lyf !!")        #output has exploded due to large learning rate
        break
    if loss<0.1:
        print("hpy lyf !!")        #never seen such low loss
        break
    x_train,y_train=load_images("./SoML-50/data/")
    x_train=torch.reshape(x_train.float(),(100,1,128,384))
    x_train/=255
    target=y_train.type(torch.LongTensor)
    target=torch.reshape(target,(100,))
#-----------------non-useful --------------    
#     output=net(x_train.float())
#     target=y_train.float()
    
#     criterion = nn.MSELoss()
# #    criterion = nn.CrossEntropyLoss()
#     loss = criterion(output, target)
#     losses.append(loss)
#------------------------------------------
    if epochs%6 and epochs<=30==0:
        alpha/=2
    optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.5)
    if epochs%10==0:
        print(loss)
    if epochs==20:
        iter=10
    if epochs==50:
        iter=2
    
    for aux in range(iter):        #going through the same batch of 100 images
        optimizer.zero_grad()
        output=net(x_train.float())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
            
    losses.append(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# plt.plot(losses)
# plt.show


# In[25]:


plt.plot(losses)  #shows variation of loss with iterations
plt.show

x_train,y_train=load_images("./SoML-50/data/")
x_train=torch.reshape(x_train.float(),(100,1,128,384))
x_train/=255
    
output=net(x_train.float())
target=y_train.float()
target=target.reshape(100,)
print(output)     #exmaple output and target
print(target)


# In[26]:


preds = torch.argmax(output, axis=-1)    #prediction made by model
print(preds)
print(torch.abs(target-preds))
print(torch.sum(torch.abs(target-preds)))


# In[71]:


torch.save(net,'model_save1')  #saving the model


# In[41]:


# model for 2nd task
class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,13))
        self.conv2 = nn.Conv2d(6, 16, (5,13))
        self.conv3 = nn.Conv2d(16, 20, (5,13))
        self.fc1 = nn.Linear(5*15*20, 480)  #15*20 from image dimension
        self.fc2 = nn.Linear(480, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10,1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (5,5))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))    #note the sigmoid activations and removal of softmax
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x.float()
    

net_in=Net1()
net_post=Net1()
net_pre=Net1()

y_label=pd.read_csv("./SoML-50/annotations.csv")

y_label=y_label.to_numpy()
y_label=y_label[0:50000,2].reshape(50000,1)
y_label=y_label.astype(float)


# In[42]:


#-----------new loading function------------
#uses the parameters of first net to classify the notations and then inserts the images in the corresponding arrays
def load_images_2(folder):
    train_data_in=[] #for infix images
    train_data_post=[] #for postfix images
    train_data_pre=[] #for prefix images
    target_label_in=[]
    target_label_post=[]
    target_label_pre=[]
    for i in range(300):
        filename=random.choice(os.listdir(folder))
        while filename=='.DS_Store':
            filename=random.choice(os.listdir(folder))
        img =Image.open(os.path.join(folder,filename))
#         target_label[i]=y_label[int(filename[0:-4])-1]
        
        data=np.asarray(img)
        data=torch.from_numpy(data)
        data=torch.reshape(data,(1,1,128,384))
        data=data.float()
        data/=255
        data=data.type(torch.LongTensor)
        output=net(data.float())
        out=torch.argmax(output,axis=-1)
        out=out.numpy()
        data=np.asarray(img)
        data=np.ndarray.tolist(data)
        if out[0]==0:
            train_data_in.append(data)
            target_label_in.append(y_label[int(filename[0:-4])-1])
        if out[0]==1:
            train_data_pre.append(data)
            target_label_pre.append(y_label[int(filename[0:-4])-1])
        if out[0]==2:
            train_data_post.append(data)
            target_label_post.append(y_label[int(filename[0:-4])-1])
    train_data_in=torch.tensor(train_data_in)
    train_data_post=torch.tensor(train_data_post)
    train_data_pre=torch.tensor(train_data_pre)
    target_label_in=torch.tensor(target_label_in)
    target_label_post=torch.tensor(target_label_post)
    target_label_pre=torch.tensor(target_label_pre)
#     train_data.float()
#     target_label.float()
    return train_data_in,train_data_pre,train_data_post,target_label_in,target_label_pre,target_label_post


# In[43]:


train_data_in,train_data_pre,train_data_post,target_label_in,target_label_pre,target_label_post=load_images_2("./SoML-50/data/")

train_data_in=torch.reshape(train_data_in.float(),(-1,1,128,384))
train_data_pre=torch.reshape(train_data_pre.float(),(-1,1,128,384))
train_data_post=torch.reshape(train_data_post.float(),(-1,1,128,384))
train_data_in/=255
train_data_pre/=255
train_data_post/=255

output_1=net_in(train_data_in.float())
output_2=net_pre(train_data_pre.float())
output_3=net_post(train_data_post.float())
print(output_1.size())
print(output_2.size())
print(output_3.size())


# In[44]:


print(target_label_in.size())
print(output_1,target_label_in)


# In[45]:



criterion = nn.MSELoss() #note change of loss function

loss_1 = criterion(output_1,target_label_in)
loss_2 = criterion(output_2,target_label_pre)
loss_3 = criterion(output_3,target_label_post)
print(loss_1,loss_2,loss_3)


# In[48]:


#clearly to our "sed" realisation trying different learning rates and other hyper-parameters is doing no good to this model 
#which is clearly trying to output the average of all the value targets which is not wanted
#hence we will try now a K means clustering algorithm to seperate out *,/,-,+ 

losses1=[]
losses2=[]
losses3=[]

alpha=0.001
iter=10

for epochs in range(100):
#     if loss>4:
#         print("sed lyf !!")
#         break
#     if loss<0.1:
#         print("hpy lyf !!")
#         break
    train_data_in,train_data_pre,train_data_post,target_label_in,target_label_pre,target_label_post=load_images_2("./SoML-50/data/")
    train_data_in=torch.reshape(train_data_in.float(),(-1,1,128,384))
    train_data_pre=torch.reshape(train_data_pre.float(),(-1,1,128,384))
    train_data_post=torch.reshape(train_data_post.float(),(-1,1,128,384))
    train_data_in/=255
    train_data_pre/=255
    train_data_post/=255

    output_1=net_in(train_data_in.float())
    output_2=net_pre(train_data_pre.float())
    output_3=net_post(train_data_post.float())
    
#     output=net(x_train.float())
#     target=y_train.float()
    
#     criterion = nn.MSELoss()
# #    criterion = nn.CrossEntropyLoss()
#     loss = criterion(output, target)
#     losses.append(loss)
    if epochs%6 and epochs<=30==0:
        lr/=2
        alpha*=2
    optimizer_1 = optim.SGD(net_in.parameters(), lr=5*1e-4, momentum=0.9)
    optimizer_2 = optim.SGD(net_pre.parameters(), lr=2*1e-5, momentum=0.9)
    optimizer_3 = optim.SGD(net_post.parameters(), lr=4*1e-5, momentum=0.9)
    if epochs%10==0:
        print(loss_1,end=",")
        print(loss_2,end=",")
        print(loss_3)
    if epochs==20:
        iter=5
    if epochs==50:
        iter=2
    
    for aux in range(iter):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        
        output_1=net_in(train_data_in.float())
        output_2=net_pre(train_data_pre.float())
        output_3=net_post(train_data_post.float())
        
        criterion = nn.MSELoss()
        
        loss_1 = criterion(output_1,target_label_in.float())
        loss_2 = criterion(output_2,target_label_pre.float())
        loss_3 = criterion(output_3,target_label_post.float())
        
        loss_1.backward()
        loss_2.backward()
        loss_3.backward()
        
        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        
    print(loss_1,end=",")
    print(loss_2,end=",")
    print(loss_3)
            
    losses1.append(loss_1)
    losses2.append(loss_2)
    losses3.append(loss_3)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# plt.plot(losses)
# plt.show


# In[51]:


plt.plot(losses1)
plt.show
plt.plot(losses2)
plt.show
plt.plot(losses3)
plt.show

print(output_3,target_label_in)


# In[50]:


# ----------- used for the failed K means algo ---------------------
#--------------------------------------------------------------------


# def load_images_3(folder):
#     train_data_in=[]
#     for i in range(300):
#         filename=random.choice(os.listdir(folder))
#         while filename=='.DS_Store':
#             filename=random.choice(os.listdir(folder))
#         img =Image.open(os.path.join(folder,filename))
# #         target_label[i]=y_label[int(filename[0:-4])-1]
        
#         data=np.asarray(img)
#         data=torch.from_numpy(data)
#         data=torch.reshape(data,(1,1,128,384))
#         data=data.float()
#         data/=255
#         data=data.type(torch.LongTensor)
#         output=net(data.float())
#         out=torch.argmax(output,axis=-1)
#         out=out.numpy()
#         if out[0]==0:
#             img =Image.open(os.path.join(folder,filename))
#             img=img.crop((128,0,256,128))
#             data=np.asarray(img)
#             data=np.ndarray.tolist(data)
#             train_data_in.append(data)
#     return train_data_in
# #     train_data.float()
# #     target_label.float()
# train_data=load_images_3("./SoML-50/data/")
# train_data=np.array(train_data)
# np.shape(train_data)
# train_data=train_data/255
# train_data=train_data.reshape(len(train_data-1),-1)


# In[119]:


#--------------------- non-useful implemention of k means algorithm -------------
#-----------------------------------------------------------------------------
# from sklearn.cluster import MiniBatchKMeans
# total_clusters = 4

# kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# for x in range(20):
#     train_data=load_images_3("./SoML-50/data/")
#     train_data=np.array(train_data)
#     np.shape(train_data)
#     train_data=train_data/255
#     train_data=train_data.reshape(len(train_data-1),-1)
#     for y in range(10):
#          kmeans.fit(train_data)
            
# print(kmeans.labels)

# train_data_in=[]
# img =Image.open("./SoML-50/data/13.jpg")
# img=img.crop((128,0,256,128))
# img.show()
# data=np.asarray(img)
# data=np.ndarray.tolist(data)
# train_data_in.append(data)
# train_data_in=np.array(train_data_in)
# np.shape(train_data_in)
# train_data_in=train_data_in/255
# train_data_in=train_data_in.reshape(len(train_data_in-1),-1)
# predicted_cluster = kmeans.predict(train_data_in)
# print(predicted_cluster)

#--------------------------------------------------
#-------------------------------------------------


# In[127]:





# 
