#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from VGG import *
from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


# In[ ]:


classes =  ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


# In[10]:


vgg = vgg16(pretrained=False, progress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device)
vgg.load_state_dict(torch.load('model.pth', map_location=device))
vgg.eval()


# In[13]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
correct_total = 0
count_total = 0
batch_cnt = 0
with torch.no_grad():
    for data in testloader:
        batch_cnt = batch_cnt + 1
        images, labels = data
        images = images
        labels = labels
        #print(labels)
        outputs = vgg(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(100):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        if batch_cnt == 10 :
            break


#for i in range(10):
    #count_total += class_total[i]
    #correct_total += class_correct[i]
    #print('Accuracy of %5s : %2d %%' % (
        #classes[i], 100 * class_correct[i] / class_total[i]))

print('Accuracy : %2d %%' % (
        100 * correct_total / count_total))

