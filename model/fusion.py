

import copy
import torch

def fuse_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def fuse_conv_bn_conv_bn(conv1, bn1,conv2,bn2):
    assert(not (conv1.training or bn1.training)), "Fusion only for eval!"
    fused_bn = copy.deepcopy(bn1)
    fused_conv = copy.deepcopy(conv1)

    fused_conv1 = copy.deepcopy(conv1)
    fused_conv2 = copy.deepcopy(conv2)
    fused_conv1.weight, fused_conv1.bias = \
        fuse_conv_bn_weights(fused_conv1.weight, fused_conv1.bias,
                             bn1.running_mean, bn1.running_var, bn1.eps, bn1.weight, bn1.bias)
    fused_conv2.weight, fused_conv2.bias = \
        fuse_conv_bn_weights(fused_conv2.weight, fused_conv2.bias,
                             bn2.running_mean, bn2.running_var, bn2.eps, bn2.weight, bn2.bias)
    fused_conv.weight = (fused_conv2.weight.clone()+fused_conv1.weight.clone())/fused_conv1.weight.clone()*fused_conv.weight.clone()
    fused_bn.bias = fused_bn.bias.clone()+fused_conv2.bias.clone()
    return fused_conv,fused_bn

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)

def fuse_linear_bn_eval(linear, bn):
    assert(not (linear.training or bn.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_linear.weight, fused_linear.bias = fuse_linear_bn_weights(
        fused_linear.weight, fused_linear.bias,
        bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_linear

def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)

    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b

    return torch.nn.Parameter(fused_w), torch.nn.Parameter(fused_b)
