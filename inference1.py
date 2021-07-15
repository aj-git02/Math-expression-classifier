import pickle
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

#get_ipython().run_line_magic('matplotlib', 'inline')
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

net=(torch.load('model_save1'))
net.eval()


folder=input("PATH")
results=[]

def load_images(folder):
    i=0
    for filename in os.listdir(folder):
        i+=1
        if i==30:
            break
        if filename=='.DS_Store':
            continue
        img =Image.open(os.path.join(folder,filename))
        if img is not None:
            data=np.asarray(img)
            data=torch.from_numpy(data)
            data=torch.reshape(data,(1,1,128,384))
            data=data.float()
            data/=255
            data=data.type(torch.LongTensor)
            output=net(data.float())
            out=torch.argmax(output,axis=-1)
            out=out.numpy()
            results.append([filename,out[0]])

load_images(folder)
#./SoML-50/data/




# print(results)
import csv

def data_write_csv(file_name, datas): 
    with open(file_name, 'w') as file_csv:
        fieldnames = ['Image Name', 'Label']
        writer = csv.DictWriter(file_csv,fieldnames=fieldnames)
        writer.writeheader()
        for data in datas:
            writer.writerow({'Image Name':data[0],'Label':data[1]})
        
data_write_csv("BugML_1.csv",results)







