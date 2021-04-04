import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from custom_model import Basicblock, Aresblock,Aresnet_2,Aresnet_3,Aresnet_4,Aresnet_5,Aresblock1_1
from custom_involution import Involution,Involution_bin

class pre_conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(pre_conv, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        return x

class custom_react(nn.Module):    
    def __init__(self, stage_channel = (32,64,128,128,256,256,512,512,512,512,512,512,1024,1024),num_class = 100):
        
        # Ares_5
        #stage_channel = (16,64,256,256,256,256,1024,1024,1024,1024,1024,1024,4096,4096)

        stage_channel = (32,64,64,128,128,256,256)
        stage_channel = [2*i for i in stage_channel]

        super(custom_react,self).__init__()
        self.feature = nn.ModuleList()
        self.num_class = num_class
        for i,out_channel in enumerate(stage_channel):
            if i==0:
                self.feature.append(pre_conv(3,out_channel,1))
            #elif(stage_channel[i-1] != out_channel and out_channel !=64):
            #    self.feature.append(Aresblock(stage_channel[i - 1], out_channel, 2))
            elif(stage_channel[i-1] != out_channel):
                self.feature.append(Aresblock1_1(stage_channel[i - 1], out_channel, stride=2))
            else:
                self.feature.append(Aresblock1_1(stage_channel[i - 1], out_channel, stride=1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channel[-1],num_class)
        self.soft = nn.Softmax(dim=1)



    def forward(self,x):
        for block in self.feature:
            x = block(x)

        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #x = self.soft(x)

        return x
