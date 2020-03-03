#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


# In[ ]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
resize = transforms.Resize((224, 224))
my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])
#my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor()])


# In[3]:


train_data = dataset.MNIST(root='data/', train=True, transform=my_transform, download=True)
test_data  = dataset.MNIST(root='data/', train=False, transform=my_transform, download=True)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
#print('number of training data: ', len(train_data))
#print('number of test data: ', len(test_data))


# In[ ]:


#image, label = train_data[1]

#print('Image')
#print('========================================')
#print('shape of this image\t:', image.shape)
#print('7\'th row of this image\t:', image[1][6])

#print('Label')
#print('========================================')
#print('label: ', label)


# In[ ]:


#plt.imshow(image[1])
#plt.imshow(image.squeeze().numpy(), cmap='gray')
#plt.title('%i' % label)
#plt.show()


# In[ ]:


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.beforeResidual = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.shortcut = nn.Conv2d(64, 512, kernel_size=2, stride=20, padding=5)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #print(x.shape)
        x = self.beforeResidual(x)
        residual = x
        residual = self.shortcut(residual)
        # 64 channel, 112*112 size
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)  + torch.flatten(residual, 1)# x_1 dimension 맞추고 여기에 더해주기
        #print(x.shape)
        # 512 channel, 7*7 size
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 64
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)

vgg = vgg16(pretrained=False, progress=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = vgg.to(device)


# In[7]:


#from torchsummary import summary
#summary(vgg, (3, 224, 224))


# In[ ]:


classes =  ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(vgg.parameters(),lr=0.001)


# In[9]:


if __name__=="__main__":
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            #print("count: " )
            #print(i)
            # zero the parameter gradients
            optimizer.zero_grad()

            #print(inputs.shape)  
            # forward + backward + optimize
            #print(vgg(inputs).shape)
            outputs = vgg(inputs)
            #print(outputs.shape)
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if(loss.item() > 1000):
                print(loss.item())
                for param in vgg.parameters():
                    print(param.data)
            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    print('Finished Training')


# In[10]:


if __name__=="__main__":
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    correct_total = 0
    count_total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = vgg(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        count_total += class_total[i]
        correct_total += class_correct[i]
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('Overall Accuracy : %2d %%' % (
            100 * correct_total / count_total))


# In[ ]:


#torch.save(vgg.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/prography-6th-deep-jonghakim/model.pth')


# In[13]:


#from google.colab import auth
#auth.authenticate_user()

#from google.colab import drive
#drive.mount('/content/gdrive')


# In[16]:


#!cat '/content/gdrive/Colab Notebooks/prography-6th-deep-jonghakim/VGG16-codes.ipynb'

