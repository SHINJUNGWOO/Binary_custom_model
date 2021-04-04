import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Involution(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride = 1,padding = 1,dilation =1,reduce_rate = 2,group=1):
        super(Involution,self).__init__()
        self.in_channel = in_channel
        self.out_channel =out_channel
        self.kernel_size= kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.reduce_rate = reduce_rate
        self.group = group


        self.pre_conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.pre_norm = nn.BatchNorm2d(out_channel)

        self.unfold = nn.Unfold(kernel_size,dilation,padding=padding,stride=stride)

        self.reduce = nn.Conv2d(out_channel,out_channel//reduce_rate,kernel_size=1,padding=0)
        self.span = nn.Conv2d(out_channel//reduce_rate, kernel_size**2*group, kernel_size=1,padding=0)

        self.pooling = nn.AvgPool2d(stride,stride)

        self.mid_norm = nn.BatchNorm2d(out_channel)

        self.post_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.post_norm = nn.BatchNorm2d(out_channel)

        self.downsample =nn.Sequential(nn.AvgPool2d(stride,stride),nn.Conv2d(in_channel, out_channel, kernel_size=1)) if stride == 2 else nn.Identity()


    def forward(self,x):
        x_input = x
        batchsize, num_channels, height, width = x.size()
        x = self.pre_conv(x)
        x = self.pre_norm(x)
        x = F.relu(x)

        if self.stride  != 1:
            height = height//2
            width = width // 2

        kernel = x

        x = self.unfold(x)
        x = x.view(batchsize,self.group,self.out_channel//self.group,self.kernel_size*self.kernel_size,height,width)
        
        kernel = self.pooling(kernel)
        kernel = self.span(self.reduce(kernel)).view(batchsize,self.group,1,self.kernel_size**2,height,width)
        x = (x * kernel).sum(dim=3)
        x = x.view(batchsize,-1,height,width)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.post_conv(x)
        x = self.post_norm(x)
        #x = F.relu(x)
        
        x = x + self.downsample(x_input)

        return x

class Involution_bin(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, reduce_rate=2, group=1):
        super(Involution_bin, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.reduce_rate = reduce_rate
        self.group = group

        self.pre_conv = BinaryConv(in_channel, out_channel, kernel_size=1,padding=0)
        self.pre_norm = nn.BatchNorm2d(out_channel)

        self.pre_bias = Learnablebias(in_channel)
        self.pre_act = nn.PReLU(out_channel)

        self.unfold = nn.Unfold(kernel_size, dilation, padding=padding, stride=stride)

        self.reduce = BinaryConv(out_channel, out_channel // reduce_rate, kernel_size=1, padding=0)
        self.span = BinaryConv(out_channel // reduce_rate, kernel_size ** 2 * group, kernel_size=1, padding=0)

        self.pooling = nn.AvgPool2d(stride, stride)

        self.mid_norm = nn.BatchNorm2d(out_channel)

        self.mid_bias = Learnablebias(out_channel)
        self.mid_act = nn.PReLU(out_channel)

        self.post_conv = BinaryConv(out_channel, out_channel, kernel_size=1,padding=0)
        self.post_norm = nn.BatchNorm2d(out_channel)


        self.post_bias = Learnablebias(out_channel)
        self.post_act = nn.PReLU(out_channel)

        self.downsample = nn.Sequential(nn.AvgPool2d(stride, stride), BinaryConv(in_channel, out_channel,
                                                                                kernel_size=1,padding=0)) if stride == 2 else nn.Identity()

        self.bin_act = Binaryactivation()

    def forward(self, x):
        x_input = x
        batchsize, num_channels, height, width = x.size()

        x = self.bin_act(x)
        x = self.pre_bias(x)

        x = self.pre_conv(x)
        x = self.pre_norm(x)

        x = self.pre_act(x)

        if self.stride != 1:
            height = height // 2
            width = width // 2


        x = self.mid_bias(x)

        kernel = x


        x = self.unfold(x)
        x = x.view(batchsize, self.group, self.out_channel // self.group, self.kernel_size * self.kernel_size, height,
                   width)

        kernel = self.pooling(kernel)
        kernel = self.span(self.reduce(kernel)).view(batchsize, self.group, 1, self.kernel_size ** 2, height, width)
        x = (x * kernel).sum(dim=3)
        x = x.view(batchsize, -1, height, width)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_act(x)

        x = self.post_bias(x)


        x = self.post_conv(x)
        x = self.post_norm(x)
        # x = F.relu(x)

        x = x + self.downsample(x_input)

        return x

# class Involution(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel_size=3,stride = 1,padding = 1,dilation =1,reduce_rate = 2,group=1):
#         super(Involution,self).__init__()
#         self.in_channel = in_channel
#         self.out_channel =out_channel
#         self.kernel_size= kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.reduce_rate = reduce_rate
#         self.group = group
#
#         self.pre_conv = BinaryConv(in_channel,out_channel,kernel_size=1,padding=0)
#
#
#         self.pre_norm = nn.BatchNorm2d(out_channel)
#
#         self.pre_bias = Learnablebias(in_channel)
#         self.pre_act = nn.PReLU(out_channel)
#
#         self.unfold = nn.Unfold(kernel_size,dilation,padding=padding,stride=stride)
#
#         self.reduce = BinaryConv(in_channel,out_channel//reduce_rate,kernel_size=1,padding=0)
#         self.span = BinaryConv(out_channel//reduce_rate, kernel_size**2*group, kernel_size=1,padding=0)
#         self.pooling = nn .AvgPool2d(stride,stride)
#
#         self.mid_norm = nn.BatchNorm2d(out_channel)
#
#         self.mid_bias = Learnablebias(out_channel)
#         self.mid_act = nn.PReLU(out_channel)
#
#         self.post_conv = BinaryConv(out_channel, out_channel, kernel_size=1,padding=0)
#         self.post_norm = nn.BatchNorm2d(out_channel)
#
#         self.post_bias = Learnablebias(out_channel)
#         self.post_act = nn.PReLU(out_channel)
#
#         self.bin_act = Binaryactivation()
#
#         if self.in_channel != self.out_channel:
#             self.short_cut = BinaryConv(in_channel,out_channel,kernel_size=1,stride=stride,padding=0)
#
#
#     def forward(self,x):
#         batchsize, num_channels, height, width = x.size()
#         x_input = x
#         x = self.bin_act(x)
#         x = self.pre_bias(x)
#
#         x = self.pre_conv(x)
#         x = self.pre_norm(x)
#         x = self.pre_act(x)
#
#         # channel == out_channel;
#
#
#         x = self.bin_act(x)
#         x = self.mid_bias(x)
#         x = self.unfold(x)
#         if self.stride == 2:
#             height = height//2
#             width = width//2
#         x = x.view(batchsize,self.group,self.out_channel//self.group,self.kernel_size*self.kernel_size,height,width)
#
#         kernel = self.pooling(x_input)
#         kernel = self.reduce(kernel)
#         kernel = self.span(kernel)
#         kernel = kernel.view(batchsize,self.group,1,self.kernel_size*self.kernel_size,height,width)
#         x = (x * kernel).sum(dim=3)
#         x = x.view(batchsize,-1,height,width)
#         x = self.mid_norm(x)
#         x = self.mid_act(x)
#
#
#
#         x = self.bin_act(x)
#         x = self.post_bias(x)
#         x = self.post_conv(x)
#         x = self.post_norm(x)
#         x = self.post_act(x)
#
#
#         if self.in_channel != self.out_channel:
#             x = x + self.short_cut(x_input)
#         else:
#             x = x + x_input
#         return x
#
