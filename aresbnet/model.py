'''AresBNet in PyTorch.
This AresBNet block is motivated from XNOR-Net and applied to ResNet below.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

######################################################################################
class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations. 
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)]  = grad_input[input.ge(1)] * 0.01 # avoid vanishing gradient
        grad_input[input.le(-1)] = grad_input[input.le(-1)] * 0.01 # avoid vanishing gradient
        return grad_input
#
#class ShuffleBlock(nn.Module):
#    def __init__(self, groups):
#        super(ShuffleBlock, self).__init__()
#        self.groups = groups
#
#    def forward(self, x):
#        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
#        N,C,H,W = x.size()
#        g = self.groups
#        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)
#

def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class BasicBlock_AresB(nn.Module): 
    expansion = 2 
    def __init__(self, in_planes, planes, stride=1, suffle=False):
        super(BasicBlock_AresB, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.suffle = suffle
        #self.shuffle1 = ShuffleBlock(groups=2)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.shuffle2 = ShuffleBlock(groups=2)
        self.conv2 = nn.Conv2d(2*planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
        self.maxpool2d= nn.MaxPool2d(3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2*planes)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(planes)
        
    def forward(self, x):
        if self.suffle:
          x = channel_shuffle(x)
        xa, xb = torch.chunk(x, 2, dim=1)
        x1 = BinActive.apply(x)
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x2 = self.bn1(x1)
        if self.stride != 1 :
          x3a = self.maxpool2d(x)
          x3b = x3a
        else:
          x3a = xa
          x3b = xb
        x3 = torch.cat((x2+x3a, x3b),1) 
        x3 = self.bn2(x3)
        x3 = channel_shuffle(x3)
        #print(x3.shape)
        x4 = BinActive.apply(x3)
        x4 = self.conv2(x4)
        x4 = self.relu2(x4)
        x4 = self.bn3(x4)
        x5a, x5b = torch.chunk(x3, 2, dim=1)
        x5 = torch.cat((x4+x5a, x5b),1) 
        out = x5
        return out 

class AresBNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(AresBNet, self).__init__()
        self.in_planes = 64 * 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.maxpool2d= nn.MaxPool2d(3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, suffle=False)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, suffle=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, suffle=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, suffle=True)
        self.bn2 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))            
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, suffle):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if suffle:
          dosuffle=True
        else:  
          dosuffle=False
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dosuffle))
            self.in_planes = planes * block.expansion
            dosuffle = True
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = torch.cat(2*[out], 1)
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out) 
        out = torch.chunk(out, 2, dim=1)[0] 
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def AresBNet10():
    return AresBNet(BasicBlock_AresB, [1,1,1,1])

def AresBNet12():
    print("hello")
    return AresBNet(BasicBlock_AresB, [2,1,1,1])

def AresBNet18():
    return AresBNet(BasicBlock_AresB, [2,2,2,2])

def AresBNet34(): 
    return AresBNet(BasicBlock_AresB, [3,4,6,3])
