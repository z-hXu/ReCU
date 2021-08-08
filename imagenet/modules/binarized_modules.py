import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from utils.options import args

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1,2,3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1,2,3], keepdim=True) + 1e-5)
        else: 
            a0 = a
        
        #* binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input