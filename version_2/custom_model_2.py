import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3(in_channel,out_channel,stride = 1):
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False
                     )

def conv1(in_channel,out_channel,stride = 1):
    return nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False
                     )
# standard convolution

def channel_shuffle(x, groups=2):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)


    #######################
    # scatter 그룹에 단일 채널로 뿌린다 ex channel 64 group 4 라면
    # 4,16 이제 16*4 로 바뀌어 16개의 그룹에 각 그룹당 값들이 들어간다.
    # #######################
    # batchsize, num_channels, height, width = x.size()
    # channels_per_group = num_channels // groups
    #
    # # reshape
    # x = x.view(batchsize, groups,groups,
    #            channels_per_group//groups, height, width)
    #
    # x = torch.transpose(x, 1, 2).contiguous()
    #
    # # flatten
    # x = x.view(batchsize, -1, height, width)

    ##############
    # scatter 그룹에 단일 채널로 뿌린다 ex channel 64 group 4 라면
    # 4,4,16 이제 4,4 16 으로 바뀌어 4개의 그룹에 각 값들(그룹에서 4로 나눠진 channel의 집합)들이 들어간다.
    return x


# class custom_shuffle(nn.Module):
#     def __init__(self,groups = 2):
#         super(custom_shuffle, self).__init__()


def custom_shuffle(x,groups=4):
    batchsize, num_channels, height, width = x.size()
    cluster_idx = F.avg_pool2d(x,kernel_size = (height//2,width//2),stride = (height//2,width//2)).flatten(2)
    cluster_idx = F.softmax(cluster_idx,dim=-1)
    cluster_idx = torch.transpose(cluster_idx,1,2)
    cluster_idx = cluster_idx.view(batchsize,groups,num_channels,1,1)
    x = x.unsqueeze(dim=1)
    out = cluster_idx*x
    out = out.view(batchsize,groups,groups,num_channels//groups,height,width)
    out = torch.transpose(out, 1, 2).contiguous()
    out = out.view(batchsize, -1, height, width)
    return out



class GroupedConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1,groups =2):
        super(GroupedConv, self).__init__()
        assert in_channel % groups == 0
        assert out_channel % groups == 0
        self.groups = groups
        self.group_in_channel = in_channel//groups
        self.group_out_channel = out_channel//groups
        self.shape = (out_channel, in_channel, kernel_size, kernel_size)
        self.conv_list = nn.ModuleList([BinaryConv(self.group_in_channel,self.group_out_channel,kernel_size,stride,padding)
                                         for i in range(groups)])
    def forward(self,x):
        #x = channel_shuffle(x,groups=self.groups)

        out = torch.cat([conv(x[:,i*self.group_in_channel:(i+1)*self.group_in_channel,:,:])
                         for i,conv in enumerate(self.conv_list)],dim=1)
        return out
class BinaryConv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride = 1,padding = 1):
        super(BinaryConv,self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_channel,in_channel,kernel_size,kernel_size)
        num_weight = in_channel * out_channel * kernel_size * kernel_size
        self.weight = nn.Parameter((torch.rand((num_weight,1))*0.001).cuda(),requires_grad=True)


    def forward(self,x):
        real_weight = self.weight.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weight), dim=3, keepdim=True), dim=2, keepdim=True),
                                   dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weight = scaling_factor * torch.sign(real_weight)
        real_weight = torch.clamp(real_weight,-1.,1.)
        binary_filter = binary_weight.detach() - real_weight.detach() + real_weight
        return F.conv2d(x, binary_filter, stride=self.stride, padding=self.padding)

class Learnablebias(nn.Module):
    def __init__(self,out_channel):
        super(Learnablebias,self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_channel,1,1),requires_grad=True)

    def forward(self,x):
        return x + self.bias.expand_as(x)

class Binaryactivation(nn.Module):
    def __init__(self):
        super(Binaryactivation,self).__init__()

    def forward(self,x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1 - mask1.type(torch.float32))
        out = out * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out = out * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out.detach() + out


        return out

class Basicblock(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1):
        super(Basicblock,self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self.move1_1 = Learnablebias(in_channel)
        self.bin3x3 = BinaryConv(in_channel,in_channel,stride = stride)
        self.bn1 = norm_layer(in_channel)

        self.move1_2 = Learnablebias(in_channel)
        self.prelu1 = nn.PReLU(in_channel)
        self.move1_3 = Learnablebias(in_channel)


        self.move2_1 = Learnablebias(in_channel)

        if in_channel == out_channel:
            self.bin_res = BinaryConv(in_channel,out_channel,kernel_size=1,padding=0)
            self.bn2 = norm_layer(out_channel)
        else:
            self.bin_res_1 = BinaryConv(in_channel,in_channel,kernel_size=1,padding=0)
            self.bin_res_2 = BinaryConv(in_channel,in_channel,kernel_size=1,padding=0)
            self.bn2_1 = norm_layer(in_channel)
            self.bn2_2 = norm_layer(in_channel)

            self.pooling = nn.AvgPool2d(2,2)

        self.move2_2 = Learnablebias(out_channel)
        self.prelu2 = nn.PReLU(out_channel)
        self.move2_3 = Learnablebias(out_channel)

        self.bin_act = Binaryactivation()

    def forward(self,x):
        x_1 = self.move1_1(x)

        x_1 = self.bin_act(x_1)
        x_1 = self.bin3x3(x_1)
        x_1 = self.bn1(x_1)

        if self.stride == 2:
            x = self.pooling(x)

        x = x + x_1

        x = self.move1_2(x)
        x = self.prelu1(x)
        x = self.move1_3(x)

        x_2 = self.move2_1(x)
        x_2 = self.bin_act(x_2)

        if self.in_channel == self.out_channel:
            x_2 = self.bin_res(x_2)
            x_2 = self.bn2(x_2)
            x_2 = x_2 + x

        else:
            assert self.out_channel == self.in_channel *2
            x_2_1 = self.bin_res_1(x_2)
            x_2_2 = self.bin_res_2(x_2)
            x_2_1 = self.bn2_1(x_2_1)
            x_2_2 = self.bn2_2(x_2_2)

            x_2_1 = x_2_1 + x
            x_2_2 = x_2_2 + x

            x_2 = torch.cat([x_2_1,x_2_2],dim=1)

        x_2 = self.move2_2(x_2)
        x_2 = self.prelu2(x_2)
        x_2 = self.move2_3(x_2)

        return x_2

class Aresblock(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1,groups = 2):
        super(Aresblock,self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self.move1_1 = Learnablebias(self.in_channel)
        self.bn1 = norm_layer(self.out_channel//2)


        self.Grouped3x3 = GroupedConv(in_channel, out_channel // 2, stride=self.stride, groups=groups)


        self.pooling = nn.AvgPool2d(2, 2)

        # stage 1

        self.move2_1 = Learnablebias(out_channel)
        self.prelu2 = nn.PReLU(out_channel)
        self.move2_2 = Learnablebias(out_channel)

        # stage2

        self.move3_1 = Learnablebias(out_channel)
        self.bn3 = norm_layer(out_channel//2)
        self.Grouped1x1 = GroupedConv(out_channel, out_channel // 2, stride=1, kernel_size=1,padding=0,groups=groups)

        # stage3

        self.move4_1 = Learnablebias(out_channel)
        self.prelu4 = nn.PReLU(out_channel)
        self.move4_2 = Learnablebias(out_channel)

        self.bin_act = Binaryactivation()

    def forward(self,x):
        x = channel_shuffle(x)

        x_1 = self.move1_1(x)
        x_1 = self.bin_act(x_1)
        x_1 = self.Grouped3x3(x_1)
        x_1 = self.bn1(x_1)

        if self.out_channel == 2*self.in_channel:
            x_1_res = self.pooling(x)
            x_1 = x_1 + x_1_res
        else:

            x_1_res = x[:,self.out_channel//2:,:,:]
            x_1 = x_1 + x[:,:self.out_channel//2,:,:]

        x_2 = torch.cat([x_1,x_1_res],dim=1)

        x_2 = self.move2_1(x_2)
        x_2 = self.prelu2(x_2)
        x_2 = self.move2_2(x_2)

        x_2 = self.move3_1(x_2)
        x_3 = self.bin_act(x_2)

        x_3 = self.Grouped1x1(x_3)
        x_3 = self.bn3(x_3)

        x_3_res = x_2[:,self.out_channel//2:,:,:]
        x_3 = x_3 + x_2[:,:self.out_channel//2,:,:]
        x_4 = torch.cat([x_3,x_3_res],dim=1)

        x_5 = self.move4_1(x_4)
        x_5 = self.prelu4(x_5)
        x_5 = self.move4_2(x_5)



        return x_5


class Aresblock1_1(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1,groups = 2):
        super(Aresblock1_1,self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self.move1_1 = Learnablebias(self.in_channel)
        self.bn1 = norm_layer(self.out_channel//2)


        self.Grouped3x3 = GroupedConv(in_channel, out_channel // 2, stride=self.stride, groups=groups)


        self.pooling = nn.AvgPool2d(2, 2)

        # stage 1

        self.move2_1 = Learnablebias(out_channel)
        self.prelu2 = nn.PReLU(out_channel)
        self.move2_2 = Learnablebias(out_channel)

        # stage2

        self.move3_1 = Learnablebias(out_channel)
        self.bn3 = norm_layer(out_channel//2)
        self.Grouped1x1 = GroupedConv(out_channel, out_channel // 2, stride=1, kernel_size=3,padding=1,groups=groups)

        # stage3

        self.move4_1 = Learnablebias(out_channel)
        self.prelu4 = nn.PReLU(out_channel)
        self.move4_2 = Learnablebias(out_channel)

        self.bin_act = Binaryactivation()

        self.res_layer = BinaryConv(in_channel, out_channel, stride=stride, kernel_size=1,padding=0) \
            if in_channel != out_channel or stride !=1\
            else nn.Identity()

    def forward(self,x):
        x_res =x
        x = channel_shuffle(x)

        x_1 = self.move1_1(x)
        x_1 = self.bin_act(x_1)
        x_1 = self.Grouped3x3(x_1)
        x_1 = self.bn1(x_1)

        if self.out_channel == 2*self.in_channel:
            x_1_res = self.pooling(x)
            x_1 = x_1 + x_1_res
        else:

            x_1_res = x[:,self.out_channel//2:,:,:]
            x_1 = x_1 + x[:,:self.out_channel//2,:,:]

        x_2 = torch.cat([x_1,x_1_res],dim=1)

        x_2 = self.move2_1(x_2)
        x_2 = self.prelu2(x_2)
        x_2 = self.move2_2(x_2)

        x_2 = self.move3_1(x_2)
        x_3 = self.bin_act(x_2)

        x_3 = self.Grouped1x1(x_3)
        x_3 = self.bn3(x_3)

        x_3_res = x_2[:,self.out_channel//2:,:,:]
        x_3 = x_3 + x_2[:,:self.out_channel//2,:,:]
        x_4 = torch.cat([x_3,x_3_res],dim=1)

        x_5 = self.move4_1(x_4)
        x_5 = self.prelu4(x_5)
        x_5 = self.move4_2(x_5)

        x_5 = x_5 + self.res_layer(x_res)

        return x_5



class Aresblock1_2(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1,groups = 2):
        super(Aresblock1_2,self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self.move1_1 = Learnablebias(self.in_channel)
        self.bn1 = norm_layer(self.out_channel//2)


        self.Grouped3x3 = GroupedConv(in_channel, out_channel // 2, stride=self.stride, groups=groups)


        self.pooling = nn.AvgPool2d(2, 2)

        # stage 1

        self.move2_1 = Learnablebias(out_channel)
        self.prelu2 = nn.PReLU(out_channel)
        self.move2_2 = Learnablebias(out_channel)

        # stage2

        self.move3_1 = Learnablebias(out_channel)
        self.bn3 = norm_layer(out_channel//2)
        self.Grouped1x1 = GroupedConv(out_channel, out_channel // 2, stride=1, kernel_size=3,padding=1,groups=groups)

        # stage3

        self.move4_1 = Learnablebias(out_channel)
        self.prelu4 = nn.PReLU(out_channel)
        self.move4_2 = Learnablebias(out_channel)

        self.bin_act = Binaryactivation()


        self.res_layer = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.res_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self,x):
        x_res =x
        x = channel_shuffle(x)

        x_1 = self.move1_1(x)
        x_1 = self.bin_act(x_1)
        x_1 = self.Grouped3x3(x_1)
        x_1 = self.bn1(x_1)

        if self.out_channel == 2*self.in_channel:
            x_1_res = self.pooling(x)
            x_1 = x_1 + x_1_res
        else:

            x_1_res = x[:,self.out_channel//2:,:,:]
            x_1 = x_1 + x[:,:self.out_channel//2,:,:]

        x_2 = torch.cat([x_1,x_1_res],dim=1)

        x_2 = self.move2_1(x_2)
        x_2 = self.prelu2(x_2)
        x_2 = self.move2_2(x_2)

        x_2 = self.move3_1(x_2)
        x_3 = self.bin_act(x_2)

        x_3 = self.Grouped1x1(x_3)
        x_3 = self.bn3(x_3)

        x_3_res = x_2[:,self.out_channel//2:,:,:]
        x_3 = x_3 + x_2[:,:self.out_channel//2,:,:]
        x_4 = torch.cat([x_3,x_3_res],dim=1)

        x_5 = self.move4_1(x_4)
        x_5 = self.prelu4(x_5)
        x_5 = self.move4_2(x_5)

        x_5 = x_5 + self.res_layer(x_res)

        return x_5


class Aresblock1_3(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1,groups = 2):
        super(Aresblock1_3,self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self.move1_1 = Learnablebias(self.in_channel)
        self.bn1 = norm_layer(self.out_channel//2)


        self.Grouped3x3 = GroupedConv(in_channel, out_channel // 2, stride=self.stride, groups=groups)


        self.pooling = nn.AvgPool2d(2, 2)

        # stage 1

        self.move2_1 = Learnablebias(out_channel)
        self.prelu2 = nn.PReLU(out_channel)
        self.move2_2 = Learnablebias(out_channel)

        # stage2

        self.move3_1 = Learnablebias(out_channel)
        self.bn3 = norm_layer(out_channel//2)
        self.Grouped1x1 = GroupedConv(out_channel, out_channel // 2, stride=1, kernel_size=1,padding=0,groups=groups)

        # stage3

        self.move4_1 = Learnablebias(out_channel)
        self.prelu4 = nn.PReLU(out_channel)
        self.move4_2 = Learnablebias(out_channel)

        self.bin_act = Binaryactivation()


        self.res_layer = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.res_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self,x):
        x_res =x
        x = channel_shuffle(x)

        x_1 = self.move1_1(x)
        x_1 = self.bin_act(x_1)
        x_1 = self.Grouped3x3(x_1)
        x_1 = self.bn1(x_1)

        if self.out_channel == 2*self.in_channel:
            x_1_res = self.pooling(x)
            x_1 = x_1 + x_1_res
        else:

            x_1_res = x[:,self.out_channel//2:,:,:]
            x_1 = x_1 + x[:,:self.out_channel//2,:,:]

        x_2 = torch.cat([x_1,x_1_res],dim=1)

        x_2 = self.move2_1(x_2)
        x_2 = self.prelu2(x_2)
        x_2 = self.move2_2(x_2)

        # x_2 = self.move3_1(x_2)
        # x_3 = self.bin_act(x_2)
        #
        # x_3 = self.Grouped1x1(x_3)
        # x_3 = self.bn3(x_3)
        #
        # x_3_res = x_2[:,self.out_channel//2:,:,:]
        # x_3 = x_3 + x_2[:,:self.out_channel//2,:,:]
        # x_4 = torch.cat([x_3,x_3_res],dim=1)

        x_5 = self.move4_1(x_2)
        x_5 = self.prelu4(x_5)
        x_5 = self.move4_2(x_5)

        x_5 = x_5 + self.res_layer(x_res)

        return x_5


class Aresblock1_4(nn.Module):
    def __init__(self,in_channel,out_channel,stride = 1,groups = 2):
        super(Aresblock1_4,self).__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel

        norm_layer = nn.BatchNorm2d

        self.bn0 = norm_layer(in_channel)


        self.move1_1 = Learnablebias(self.in_channel)
        self.bn1 = norm_layer(self.out_channel//2)


        self.Grouped3x3 = GroupedConv(in_channel, out_channel // 2, stride=self.stride, groups=groups)

        self.prelu1 = nn.PReLU(out_channel//2)
        self.pooling = nn.AvgPool2d(2, 2)

        # stage 1

        self.move2_1 = Learnablebias(out_channel)
        self.prelu2 = nn.PReLU(out_channel)
        self.move2_2 = Learnablebias(out_channel)

        # stage2

        self.move3_1 = Learnablebias(out_channel)
        self.bn3 = norm_layer(out_channel//2)
        self.Grouped1x1 = GroupedConv(out_channel, out_channel // 2, stride=1, kernel_size=1,padding=0,groups=groups)
        self.prelu3 = nn.PReLU(out_channel//2)
        # stage3

        self.move4_1 = Learnablebias(out_channel)
        self.prelu4 = nn.PReLU(out_channel)
        self.move4_2 = Learnablebias(out_channel)

        self.bin_act = Binaryactivation()


        self.res_layer = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.res_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.prelu5 = nn.PReLU(out_channel)

    def forward(self,x):
        x_res =x
        x = self.bn0(x)
        x = channel_shuffle(x)

        x_1 = self.move1_1(x)
        x_1 = self.bin_act(x_1)
        x_1 = self.Grouped3x3(x_1)
        x_1 = self.prelu1(x_1)
        x_1 = self.bn1(x_1)

        if self.out_channel == 2*self.in_channel:
            x_1_res = self.pooling(x)
            x_1 = x_1 + x_1_res
        else:

            x_1_res = x[:,self.out_channel//2:,:,:]
            x_1 = x_1 + x[:,:self.out_channel//2,:,:]

        x_2 = torch.cat([x_1,x_1_res],dim=1)

        x_2 = self.move2_1(x_2)
        x_2 = self.prelu2(x_2)
        x_2 = self.move2_2(x_2)

        x_2 = self.move3_1(x_2)
        x_3 = self.bin_act(x_2)

        x_3 = self.Grouped1x1(x_3)
        x_3 = self.prelu3(x_3)
        x_3 = self.bn3(x_3)

        x_3_res = x_2[:,self.out_channel//2:,:,:]
        x_3 = x_3 + x_2[:,:self.out_channel//2,:,:]
        x_4 = torch.cat([x_3,x_3_res],dim=1)

        x_5 = self.move4_1(x_4)
        x_5 = self.prelu4(x_5)
        x_5 = self.move4_2(x_5)

        x_5 = x_5 + self.res_layer(x_res)

        x_5 = self.prelu5(x_5)
        return x_5
