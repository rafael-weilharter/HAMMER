import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from torch.autograd import Variable

from models.transformers.local_attention import AttentionLayer

#UNet with optional local AttentionLayers instead of convolution layers in the residualBlock
#with GN
class unet(nn.Module):
    def __init__(self, stages=5, num_chan=32, heads=1, first_stride=2):
        super(unet, self).__init__()
        self.stages = stages
        self.heads = heads #if heads=0, convolutions will be used instead of AttentionLayers

        assert num_chan % 4 == 0, f"num_chan ({num_chan}) must be divisible by 4!"

        # Encoder
        self.convbnrelu1_1 = conv2DGroupNormReLU(in_channels=3, k_size=3, n_filters=num_chan//2,
                                                 padding=1, stride=first_stride, bias=False, with_gn=num_chan//4)
        self.convbnrelu1_2 = conv2DGroupNormReLU(in_channels=num_chan//2, k_size=3, n_filters=num_chan//2,
                                                 padding=1, stride=1, bias=False, with_gn=num_chan//4)
        self.convbnrelu1_3 = conv2DGroupNormReLU(in_channels=num_chan//2, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False, with_gn=num_chan//4)

        self.convbnrelu1_4 = conv2DGroupNormReLU(in_channels=num_chan, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False, with_gn=num_chan//4)

        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock,num_chan,num_chan,1,stride=2,with_gn=num_chan//4)
        self.res_block5 = self._make_layer(residualBlock,num_chan,num_chan,1,stride=2,with_gn=num_chan//4)

        self.res_block6 = self._make_layer(residualBlock,num_chan,num_chan,1,stride=2,with_gn=num_chan//4)
        self.res_block7 = self._make_layer(residualBlock,num_chan,num_chan,1,stride=2,with_gn=num_chan//4)

        self.res_block8 = self._make_layer(residualBlock,num_chan,num_chan,1,stride=2,with_gn=num_chan//4)

        self.res_block9 = self._make_layer(residualBlock,num_chan,num_chan,1,stride=2,with_gn=num_chan//4)

        self.upconv8 = nn.Sequential(nn.Upsample(scale_factor=2),
                                    conv2DGroupNormReLU(in_channels=num_chan, k_size=3, n_filters=num_chan,
                                                padding=1, stride=1, bias=False))
        self.iconv7 = conv2DGroupNormReLU(in_channels=num_chan*2, k_size=3, n_filters=num_chan,
                                                padding=1, stride=1, bias=False)

        self.upconv7 = nn.Sequential(nn.Upsample(scale_factor=2),
                                    conv2DGroupNormReLU(in_channels=num_chan, k_size=3, n_filters=num_chan,
                                                padding=1, stride=1, bias=False))
        self.iconv6 = conv2DGroupNormReLU(in_channels=num_chan*2, k_size=3, n_filters=num_chan,
                                                padding=1, stride=1, bias=False)

        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DGroupNormReLU(in_channels=num_chan, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False))
        self.iconv5 = conv2DGroupNormReLU(in_channels=num_chan*2, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DGroupNormReLU(in_channels=num_chan, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False))
        self.iconv4 = conv2DGroupNormReLU(in_channels=num_chan*2, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv2DGroupNormReLU(in_channels=num_chan, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False))
        self.iconv3 = conv2DGroupNormReLU(in_channels=num_chan*2, k_size=3, n_filters=num_chan,
                                                 padding=1, stride=1, bias=False)

        # self.proj6 = conv2DGroupNormReLU(in_channels=128,k_size=1,n_filters=num_chan, padding=0,stride=1,bias=False)
        # self.proj5 = conv2DGroupNormReLU(in_channels=128,k_size=1,n_filters=num_chan, padding=0,stride=1,bias=False)
        # self.proj4 = conv2DGroupNormReLU(in_channels=128,k_size=1,n_filters=num_chan, padding=0,stride=1,bias=False)
        # self.proj3 = conv2DGroupNormReLU(in_channels=num_chan*2, k_size=1,n_filters=num_chan, padding=0,stride=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight, 0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#                m.running_mean.data.fill_(0)
#                m.running_var.data.fill_(1)
       

    def _make_layer(self, block, planes_in, planes, blocks, stride=1, with_gn=8):
        downsample = None
        if stride != 1 or planes_in != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(planes_in, planes * block.expansion,
                                                 kernel_size=3, padding=1, stride=stride, bias=False),
                                       nn.GroupNorm(with_gn, planes * block.expansion),)
        layers = []
        layers.append(block(planes_in, planes, stride, downsample, self.heads))
        # planes_in = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes_in, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)
        conv1 = self.convbnrelu1_4(conv1)

        ## H/2, W/2 -> H/4, W/4
        # pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # conv2 = self.res_block2(conv1)

        # H/4, W/4 -> H/16, W/16
        conv3 = self.res_block3(conv1)
        conv4 = self.res_block5(conv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)

        conv7 = self.res_block8(conv6)

        conv8 = self.res_block9(conv7)

        concat7 = torch.cat((conv7,self.upconv8(conv8)),dim=1)
        conv7 = self.iconv7(concat7)

        concat6 = torch.cat((conv6,self.upconv7(conv7)),dim=1)
        conv6 = self.iconv6(concat6)

        concat5 = torch.cat((conv5,self.upconv6(conv6)),dim=1)
        conv5 = self.iconv5(concat5) 

        concat4 = torch.cat((conv4,self.upconv5(conv5)),dim=1)
        conv4 = self.iconv4(concat4) 

        concat3 = torch.cat((conv3,self.upconv4(conv4)),dim=1)
        conv3 = self.iconv3(concat3)

        # concat2 = torch.cat((conv2,self.upconv3(conv3)),dim=1)
        # conv2 = self.iconv2(concat2)

        # proj6 = self.proj6(conv6)
        # proj5 = self.proj5(conv5)
        # proj4 = self.proj4(conv4)
        # proj3 = self.proj3(conv3)

        outputs = {}
        # outputs["stage0"] = conv2
        outputs["stage1"] = conv3
        outputs["stage2"] = conv4
        outputs["stage3"] = conv5
        outputs["stage4"] = conv6

        if(self.stages == 5):
            outputs["stage5"] = conv7

        return outputs
        # return proj6,proj5,proj4,proj3


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.GroupNorm(n_filters//4, int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DGroupNormReLU(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_gn=0):
        super(conv2DGroupNormReLU, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_gn > 1:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.GroupNorm(with_gn, int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None, heads=1, dilation=1):
        super(residualBlock, self).__init__()

        self.heads = heads

        if dilation > 1:
            padding = dilation
        else:
            padding = 1
        self.convbnrelu1 = conv2DGroupNormReLU(in_channels, n_filters, 3,  stride, padding, bias=False,dilation=dilation)
        
        if self.heads > 0:
            self.attention = AttentionLayer(n_filters, n_filters, 3, 1, 1, groups=heads, bias=False)
            # self.norm_layer = nn.GroupNorm(8, n_filters)

            # Layer Scale: https://arxiv.org/pdf/2103.17239.pdf
            self.diag_weights = torch.FloatTensor(n_filters)
            self.diag_weights = self.diag_weights.fill_(0.1)
            self.diag_matrix = nn.Parameter(torch.diag(self.diag_weights), requires_grad=True)
        else:
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)

        self.downsample = downsample
        # self.stride = stride
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        if self.heads > 0:
            out = self.attention(out)
            out = torch.einsum('b c h w, c c -> b c h w', out, self.diag_matrix)
            # out = self.norm_layer(out)
        else:
            out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        return out