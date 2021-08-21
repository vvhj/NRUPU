import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': '/root/work/YoCo_acrpv2/imagenet/model_path/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': '/root/work/YoCo_acrpv2/imagenet/model_path/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,bias=False,padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation,padding_mode=padding_mode)


def conv1x1(in_planes, out_planes, stride=1,bias=False,padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,padding_mode=padding_mode)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ReRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)#,padding_mode='circular')
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)#,padding_mode='circular')
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #print("ok")
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class ReRBasicBlockv2_new(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBasicBlockv2_new, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)#,padding_mode='circular')
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)#,padding_mode='circular')
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.downsample1 = conv1x1(inplanes, planes , stride,bias=True)
        self.downsample2 = conv1x1(planes, planes,bias=True)
        self.downsample1.weight.data.fill_(1)
        self.downsample1.bias.data.fill_(0)
        self.downsample2.weight.data.fill_(1)
        self.downsample2.bias.data.fill_(0)
        #初始化新参数
        #A1 = self.conv1.weight.data.clone()
        #o1,i1,w1,h1  = A1.shape
        
        #A2 = self.conv2.weight.data.clone()
        #o2,i2,w2,h2  = A2.shape
        
        #A3 = torch.zeros(o1,i1,3,3).to(A1.device)
                #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o1):
        #    A3[i, i % i1, 1, 1] = 1
        #self.downsample1.weight.data = A3
        #A4 = torch.zeros(o2,i2,3,3).to(A2.device)
        #    #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o2):
        #    A4[i, i % i2, 1, 1] = 1
        #self.downsample2.weight.data = A4
        self.stride = stride
    #def init_weight(self):
        
        #if self.downsample is not None:


    def forward(self, x):
        #print("ok")
        identity0 = x

        out = self.conv1(x)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = self.downsample1(x)
            identity += identity0
        out = self.bn1(out)
        out += identity
        
        
        out = self.relu(out)
        identity0 = out
        identity = out
        out = self.conv2(out)
        identity = self.downsample2(identity)
        identity += identity0
        out = self.bn2(out)
        
        out += identity
        
        out = self.relu(out)

        return out
class ReRBasicBlockv2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBasicBlockv2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)#,padding_mode='circular')
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)#,padding_mode='circular')
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.downsample1 = conv1x1(inplanes, planes , stride)
        self.downsample2 = conv1x1(planes, planes)
        self.downsample1.weight.data.fill_(1)
        self.downsample2.weight.data.fill_(1)
        #初始化新参数
        #A1 = self.conv1.weight.data.clone()
        #o1,i1,w1,h1  = A1.shape
        
        #A2 = self.conv2.weight.data.clone()
        #o2,i2,w2,h2  = A2.shape
        
        #A3 = torch.zeros(o1,i1,3,3).to(A1.device)
                #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o1):
        #    A3[i, i % i1, 1, 1] = 1
        #self.downsample1.weight.data = A3
        #A4 = torch.zeros(o2,i2,3,3).to(A2.device)
        #    #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o2):
        #    A4[i, i % i2, 1, 1] = 1
        #self.downsample2.weight.data = A4
        self.stride = stride
    #def init_weight(self):
        
        #if self.downsample is not None:


    def forward(self, x):
        #print("ok")
        #identity = x

        out = self.conv1(x)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = self.downsample1(x)
        out = self.bn1(out)
        out += identity
        
        out = self.relu(out)
        identity = out
        out = self.conv2(out)
        identity = self.downsample2(identity)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class ReRBasicBlock_base(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBasicBlock_base, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)#,padding_mode='circular')
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)#,padding_mode='circular')
        self.bn2 = norm_layer(planes)
        #self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #print("ok")
        #identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        #if self.downsample is not None:
            #identity = self.downsample(x)
        #out += identity
        out = self.relu(out)
        #identity = out
        out = self.conv2(out)
        out = self.bn2(out)
        #out += identity
        out = self.relu(out)

        return out

class ReRBasicBlock_ACPR_V1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBasicBlock_ACPR_V1, self).__init__()
        if purncfg == None:
            purncfg = [planes]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride)
        self.bn1 = norm_layer(purncfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock_ACPR_V1(nn.Module):
    expansion = 1

    def __init__(self, inplanes,planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_ACPR_V1, self).__init__()
        if purncfg == None:
            purncfg = [planes,planes]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride)
        self.bn1 = norm_layer(purncfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ReRBasicBlockv2_APRC(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBasicBlockv2_APRC, self).__init__()
        if purncfg == None:
            purncfg = [planes,planes]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride)#,padding_mode='circular')
        self.bn1 = norm_layer(purncfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], purncfg[1])#,padding_mode='circular')
        self.bn2 = norm_layer(purncfg[1])
        self.downsample = downsample
        self.downsample1 = conv1x1(inplanes, purncfg[0] , stride)
        self.downsample2 = conv1x1(purncfg[0], purncfg[1])
        #self.downsample1.weight.data.fill_(1)
        #self.downsample2.weight.data.fill_(1)
        #初始化新参数
        #A1 = self.conv1.weight.data.clone()
        #o1,i1,w1,h1  = A1.shape
        
        #A2 = self.conv2.weight.data.clone()
        #o2,i2,w2,h2  = A2.shape
        
        #A3 = torch.zeros(o1,i1,3,3).to(A1.device)
                #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o1):
        #    A3[i, i % i1, 1, 1] = 1
        #self.downsample1.weight.data = A3
        #A4 = torch.zeros(o2,i2,3,3).to(A2.device)
        #    #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o2):
        #    A4[i, i % i2, 1, 1] = 1
        #self.downsample2.weight.data = A4
        self.stride = stride
    #def init_weight(self):
        
        #if self.downsample is not None:


    def forward(self, x):
        #print("ok")
        #identity = x

        out = self.conv1(x)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = self.downsample1(x)
        out = self.bn1(out)
        out += identity
        
        out = self.relu(out)
        identity = out
        out = self.conv2(out)
        identity = self.downsample2(identity)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class ReRBasicBlockv2_APRC_merge_bn(nn.Module):#####################################
    expansion = 1

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(ReRBasicBlockv2_APRC_merge_bn, self).__init__()
        if purncfg == None:
            purncfg = [planes,planes]
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride,bias=True)#,padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], purncfg[1],bias=True)#,padding_mode='circular')
        #self.downsample = downsample
        #self.downsample1 = conv1x1(inplanes, purncfg[0] , stride)
        #self.downsample2 = conv1x1(purncfg[0], purncfg[1])
        #self.downsample1.weight.data.fill_(1)
        #self.downsample2.weight.data.fill_(1)
        #初始化新参数
        #A1 = self.conv1.weight.data.clone()
        #o1,i1,w1,h1  = A1.shape
        
        #A2 = self.conv2.weight.data.clone()
        #o2,i2,w2,h2  = A2.shape
        
        #A3 = torch.zeros(o1,i1,3,3).to(A1.device)
                #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o1):
        #    A3[i, i % i1, 1, 1] = 1
        #self.downsample1.weight.data = A3
        #A4 = torch.zeros(o2,i2,3,3).to(A2.device)
        #    #A3 = torch.nn.functional.pad(A3, [1,1,1,1])
        #for i in range(o2):
        #    A4[i, i % i2, 1, 1] = 1
        #self.downsample2.weight.data = A4
        self.stride = stride
    #def init_weight(self):
        
        #if self.downsample is not None:


    def forward(self, x):
        #print("ok")
        #identity = x

        out = self.conv1(x)
        #out = self.relu(out)
        
        out = self.relu(out)
        out = self.conv2(out)

        out = self.relu(out)

        return out

class BasicBlock_ACPR(nn.Module):
    expansion = 1

    def __init__(self, inplanes,planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_ACPR, self).__init__()
        if purncfg == None:
            purncfg = [planes,planes]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride)
        self.bn1 = norm_layer(purncfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], purncfg[1])
        self.bn2 = norm_layer(purncfg[1])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class BasicBlock_ACPR_merge_bn_v1(nn.Module):
    expansion = 1

    def __init__(self, inplanes,planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock_ACPR_merge_bn, self).__init__()
        if purncfg == None:
            purncfg = [planes,planes]
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], purncfg[1],bias=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock_ACPR_merge_bn(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock_ACPR_merge_bn, self).__init__()
        if purncfg == None:
            purncfg = [planes]
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, purncfg[0], stride,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(purncfg[0], planes,bias=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ReRBasicBlock_merge_bn(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(ReRBasicBlock_merge_bn, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,bias=True)#,padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,bias=True)#,padding_mode='circular')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        identity = out
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        

        return out
class ReRBasicBlock_merge_bnstep2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(ReRBasicBlock_merge_bnstep2, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,bias=True)#,padding_mode='circular')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        #self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.relu(out)
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ReRBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReRBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck_APRC_V1(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck_APRC_V1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        if purncfg == None:
            purncfg = [width,width]
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, purncfg[0])
        self.bn1 = norm_layer(purncfg[0])
        self.conv2 = conv3x3(purncfg[0], purncfg[1], stride, groups, dilation)
        self.bn2 = norm_layer(purncfg[1])
        self.conv3 = conv1x1(purncfg[1], planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck_APRC(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck_APRC, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        if purncfg == None:
            purncfg = [width,width,width]
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, purncfg[0])
        self.bn1 = norm_layer(purncfg[0])
        self.conv2 = conv3x3(purncfg[0], purncfg[1], stride, groups, dilation)
        self.bn2 = norm_layer(purncfg[1])
        self.conv3 = conv1x1(purncfg[1], purncfg[2])
        self.bn3 = norm_layer(purncfg[2])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck_APRC_merge_bn(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, purncfg=None, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck_APRC_merge_bn, self).__init__()
        
        width = int(planes * (base_width / 64.)) * groups
        if purncfg == None:
            purncfg = [width,width,width]
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, purncfg[0],bias=True)
        
        self.conv2 = conv3x3(purncfg[0], purncfg[1], stride, groups, dilation,bias=True)
       
        self.conv3 = conv1x1(purncfg[1], purncfg[2],bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.relu(out)

        out = self.conv3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet_merge_bn(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(ResNet_merge_bn, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        #self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet_ACPR_V1(nn.Module):

    def __init__(self, block, layers, bncfg=None,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_ACPR_V1, self).__init__()
        self.bncfg = bncfg
        self.bnindex = 0
        name = block.__name__
        if name == "Bottleneck_APRC_V1":
            self.bnstep = 2
        elif name == "BasicBlock_ACPR_V1":
            self.bnstep = 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if self.bncfg!=None:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep] ,stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.bnindex += self.bnstep
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep], groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
                self.bnindex += self.bnstep
        else:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, None ,stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, None ,groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))


        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
class ResNet_ACPR_old(nn.Module):
    def __init__(self, block, layers, bncfg=None,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_ACPR_old, self).__init__()
        self.bncfg = bncfg
        self.bnindex = 0
        self.tinplane = 0
        name = block.__name__
        if name == "Bottleneck_APRC":
            self.bnstep = 3
        elif name == "BasicBlock_ACPR" or "ReRBasicBlockv2_APRC":
            self.bnstep = 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.tinplane = self.inplanes
        self.dilation = 1
        self.tlink_bn = self.inplanes
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.bncfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if self.bncfg!=None:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 :#or self.inplanes != planes * block.expansion: #or self.tinplane != self.bncfg[self.bnindex+self.bnstep-1]:
                downsample = nn.Sequential(
                    conv1x1(self.tinplane, self.bncfg[self.bnindex+self.bnstep-1], stride),
                    norm_layer(self.bncfg[self.bnindex+self.bnstep-1]),
                )

            layers = []
            layers.append(block(self.tinplane, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep] ,stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.tinplane = self.bncfg[self.bnindex+self.bnstep-1]
            self.bnindex += self.bnstep
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.tinplane, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep], groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
                self.tinplane = self.bncfg[self.bnindex+self.bnstep-1]
                self.bnindex += self.bnstep
        else:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, None ,stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, None ,groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))


        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
class ResNet_ACPR(nn.Module):
    def __init__(self, block, layers, bncfg=None,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_ACPR, self).__init__()
        self.bncfg = bncfg
        self.bnindex = 0
        self.tinplane = 0
        name = block.__name__
        if name == "Bottleneck_APRC":
            self.bnstep = 3
        elif name == "BasicBlock_ACPR" or "ReRBasicBlockv2_APRC":
            self.bnstep = 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.tinplane = self.inplanes
        self.dilation = 1
        self.tlink_bn = self.inplanes
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.bncfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if self.bncfg!=None:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 :#or self.inplanes != planes * block.expansion: #or self.tinplane != self.bncfg[self.bnindex+self.bnstep-1]:
                downsample = nn.Sequential(
                    conv1x1(self.tinplane, self.bncfg[self.bnindex+self.bnstep-2], stride),
                    norm_layer(self.bncfg[self.bnindex+self.bnstep-2]),
                )

            layers = []
            layers.append(block(self.tinplane, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep] ,stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.tinplane = self.bncfg[self.bnindex+self.bnstep-1]
            self.bnindex += self.bnstep
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.tinplane, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep], groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))
                self.tinplane = self.bncfg[self.bnindex+self.bnstep-1]
                self.bnindex += self.bnstep
        else:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, None ,stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, None ,groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))


        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet_ACPR_merge_bn_V1(nn.Module):
    def __init__(self, block, layers, bncfg=None,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 ):
        super(ResNet_ACPR_merge_bn_V1, self).__init__()
        self.bncfg = bncfg
        self.bnindex = 0
        self.tinplane = 0
        name = block.__name__
        if "Bottleneck" in name:
            self.bnstep = 3
        elif "BasicBlock" in name:
            self.bnstep = 2

        #self._norm_layer = norm_layer

        self.inplanes = 64
        self.tinplane = self.inplanes
        self.dilation = 1
        self.tlink_bn = self.inplanes
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        #self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.bncfg[-1], num_classes)

        #for m in self.modules():
            #if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
             #   nn.init.constant_(m.weight, 1)
              #  nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
            #for m in self.modules():
                #if isinstance(m, Bottleneck):
                    #nn.init.constant_(m.bn3.weight, 0)
                #elif isinstance(m, BasicBlock):
                    #nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if self.bncfg!=None:
            #norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.tinplane != self.bncfg[self.bnindex+self.bnstep-1]:
                downsample = nn.Sequential(
                    conv1x1(self.tinplane, self.bncfg[self.bnindex+self.bnstep-1], stride,bias=True),
                    #norm_layer(self.bncfg[self.bnindex+self.bnstep-1]),
                )

            layers = []
            layers.append(block(self.tinplane, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep] ,stride, downsample, self.groups,
                                self.base_width, previous_dilation))
            self.tinplane = self.bncfg[self.bnindex+self.bnstep-1]
            self.bnindex += self.bnstep
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.tinplane, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep], groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation))
                self.tinplane = self.bncfg[self.bnindex+self.bnstep-1]
                self.bnindex += self.bnstep
        else:
            #norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),bias=True
                    #norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, None ,stride, downsample, self.groups,
                                self.base_width, previous_dilation))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, None ,groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation))


        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class ResNet_ACPR_merge_bn(nn.Module):
    def __init__(self, block, layers, bncfg=None,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 ):
        super(ResNet_ACPR_merge_bn, self).__init__()
        self.bncfg = bncfg
        self.bnindex = 0
        self.tinplane = 0
        name = block.__name__
        if name == "Bottleneck_APRC_merge_bn":
            self.bnstep = 2
        elif name == "BasicBlock_ACPR_merge_bn":
            self.bnstep = 1

        #self._norm_layer = norm_layer

        self.inplanes = 64
        self.tinplane = self.inplanes
        self.dilation = 1
        self.tlink_bn = self.inplanes
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        #self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #for m in self.modules():
            #if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
             #   nn.init.constant_(m.weight, 1)
              #  nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #if zero_init_residual:
            #for m in self.modules():
                #if isinstance(m, Bottleneck):
                    #nn.init.constant_(m.bn3.weight, 0)
                #elif isinstance(m, BasicBlock):
                    #nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if self.bncfg!=None:
            downsample = None
            previous_dilation = self.dilation
            
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride,bias=True),
                )

            layers = []
            layers.append(block(self.inplanes, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep] ,stride, downsample, self.groups,
                                self.base_width, previous_dilation))
            self.bnindex += self.bnstep
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes,self.bncfg[self.bnindex:self.bnindex+self.bnstep], groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation))
                self.bnindex += self.bnstep
        else:
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride,bias=True)
                )

            layers = []
            layers.append(block(self.inplanes, planes, None ,stride, downsample, self.groups,
                                self.base_width, previous_dilation))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, None ,groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation))


        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def _resnet_merge_bn(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_merge_bn(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def _resnet_acpr(block, layers, bncfg,**kwargs):
    model = ResNet_ACPR(block, layers, bncfg=bncfg,**kwargs)
    return model

def _resnet_acpr_merge_bn(block, layers, bncfg,**kwargs):
    model = ResNet_ACPR_merge_bn(block, layers, bncfg=bncfg,**kwargs)
    return model

def _resnet_acpr_merge_bn_v1(block, layers, bncfg,**kwargs):
    model = ResNet_ACPR_merge_bn_V1(block, layers, bncfg=bncfg,**kwargs)
    return model

def _resnet_acpr_v1(block, layers, bncfg,**kwargs):
    model = ResNet_ACPR_V1(block, layers, bncfg=bncfg,**kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def rerresnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('rerresnet18', ReRBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
def rerresnet18v2(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('rerresnet18v2', ReRBasicBlockv2, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
def rerresnet18_base(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('rerresnet18_base', ReRBasicBlock_base, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


#----------------------------------------------------------------------------------------

def resnet18_acpr(bncfg=None,**kwargs):
    r"""ResNet-18_acpr model

    Args:
        bncfg (path): if none return original resnet18
    """
    return _resnet_acpr_v1(BasicBlock_ACPR_V1, [2, 2, 2, 2],bncfg,
                   **kwargs)

def rerresnet18v2_acpr(bncfg=None,**kwargs):
    r"""ResNet-18_acpr model

    Args:
        bncfg (path): if none return original resnet18
    """
    return _resnet_acpr(ReRBasicBlockv2_APRC, [2, 2, 2, 2],bncfg,
                   **kwargs)

def resnet34_acpr(bncfg=None, **kwargs):
    r"""ResNet-34_acpr model

    Args:
        bncfg (path): if none return original resnet34
    """
    return _resnet_acpr(BasicBlock_ACPR, [3, 4, 6, 3], bncfg,
                   **kwargs)


def resnet50_acpr(bncfg=None, **kwargs):
    r"""ResNet-50_acpr model

    Args:
        bncfg (path): if none return original resnet50
    """
    return _resnet_acpr( Bottleneck_APRC, [3, 4, 6, 3], bncfg,
                   **kwargs)

def resnet50_acpr_v1(bncfg=None, **kwargs):
    r"""ResNet-50_acpr model

    Args:
        bncfg (path): if none return original resnet50
    """
    return _resnet_acpr_v1( Bottleneck_APRC_V1, [3, 4, 6, 3], bncfg,
                   **kwargs)

def resnet101_acpr(bncfg=None, **kwargs):
    r"""ResNet-101_acpr model

    Args:
        bncfg (path): if none return original resnet101
    """
    return _resnet_acpr(Bottleneck_APRC, [3, 4, 23, 3], bncfg,
                   **kwargs)


def resnet101_acpr_v1(bncfg=None, **kwargs):
    r"""ResNet-101_acpr model

    Args:
        bncfg (path): if none return original resnet101
    """
    return _resnet_acpr_v1(Bottleneck_APRC_V1, [3, 4, 23, 3], bncfg,
                   **kwargs)


def resnet152_acpr(bncfg=None, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_acpr(Bottleneck_APRC,  [3, 8, 36, 3], bncfg,
                   **kwargs)
def resnet18_acpr_merge_bn(bncfg=None,**kwargs):
    r"""ResNet-18_acpr model

    Args:
        bncfg (path): if none return original resnet18
    """
    return _resnet_acpr_merge_bn(BasicBlock_ACPR_merge_bn, [2, 2, 2, 2],bncfg,
                   **kwargs)
def rerresnet18_merge_bn_v2(bncfg=None,**kwargs):
    r"""ResNet-18_acpr model

    Args:
        bncfg (path): if none return original resnet18
    """
    return _resnet_acpr_merge_bn_v1(ReRBasicBlockv2_APRC_merge_bn, [2, 2, 2, 2],bncfg,
                   **kwargs)

def rerresnet18_merge_bn(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18_acpr model

    Args:
        bncfg (path): if none return original resnet18
    """
    return _resnet_merge_bn("rerresnet18_merge_bn",ReRBasicBlock_merge_bn, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
def rerresnet18_merge_bn_step2(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18_acpr model

    Args:
        bncfg (path): if none return original resnet18
    """
    return _resnet_merge_bn("rerresnet18_merge_bn_step2",ReRBasicBlock_merge_bnstep2, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34_acpr_merge_bn(bncfg=None, **kwargs):
    r"""ResNet-34_acpr model

    Args:
        bncfg (path): if none return original resnet34
    """
    return _resnet_acpr_merge_bn(BasicBlock_ACPR_merge_bn, [3, 4, 6, 3], bncfg,
                   **kwargs)


def resnet50_acpr_merge_bn(bncfg=None, **kwargs):
    r"""ResNet-50_acpr model

    Args:
        bncfg (path): if none return original resnet50
    """
    return _resnet_acpr_merge_bn( Bottleneck_APRC_merge_bn, [3, 4, 6, 3], bncfg,
                   **kwargs)

# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)


# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)
