from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import _init_paths

from fairseq.models.transformer import MultiheadAttention

import pdb

class transformer(nn.Module):
    def __init__(self, triplet_size, input_size=128):
        super(transformer, self).__init__()
        self.key_depth = 64*4
        self.value_depth = 64*4
        self.output_depth = 2048
        self.num_heads = 1
        self.pos_size = 4
        self.out_channel = 4

        self.bn_conv = nn.BatchNorm3d(num_features=30,
                                 track_running_stats=True)

        self.conv1_3d = nn.Conv3d(in_channels=30,
                                  out_channels=10,
                                  kernel_size=(3,3,3),
                                  stride=(4,2,2),
                                  padding=(1,1,1),
                                  dilation=(1,1,1),
                                  bias=True)
        self.conv2_3d = nn.Conv3d(in_channels=10,
                                  out_channels=self.out_channel,
                                  kernel_size=(3,3,3),
                                  stride=(2,1,1),
                                  padding=(1,1,1),
                                  dilation=(1,1,1),
                                  bias=True)

        self.transformer_model = MultiheadAttention(self.key_depth, self.value_depth,
                                                    self.output_depth, self.num_heads,
                                                    dropout=0.1)

        #self.bn = nn.BatchNorm3d(num_features=self.out_channel,
        #                        track_running_stats=True)

        self.bias = self.generate_bias()
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc1 = Linear(in_features=self.pos_size*self.pos_size*self.output_depth, out_features=2048)
        self.dropout2 = nn.Dropout(p=0.1)
        self.cls_fc = Linear(in_features=2048, out_features=triplet_size, bias=False)




    def generate_bias(self):
        emb_dim = self.pos_size**2
        emb_size = self.pos_size**2
        pos = torch.tensor([x for x in range(self.pos_size)])
        pos_emb = pos.repeat(self.pos_size, 1)**2 + pos.repeat(self.pos_size, 1).t()**2
        basic_emb = pos_emb.float().sqrt()
        product_basic = basic_emb.unsqueeze(0).repeat(emb_size, 1, 1)
        ratio = 10000**(2*torch.tensor([x for x in range(emb_size)]).float()/emb_dim)
        ratio_matrix = ratio.unsqueeze(1).unsqueeze(2).repeat(1, self.pos_size, self.pos_size)
        production = product_basic*ratio_matrix
        sin_ind = [2*x for x in range(emb_size//2)]
        sin_ind.append(self.pos_size**2-1)
        cos_ind = [2*x+1 for x in range(emb_size//2)]
        cos_emb = torch.cos(production)
        cos_emb[sin_ind] = 0
        sin_emb = torch.sin(production)
        sin_emb[cos_ind] = 0
        return (sin_emb+cos_emb).view(emb_dim, -1)



    def forward(self, inputs, save_flag=False):
        in_size = torch.tensor(inputs.size()).tolist()
        re_inputs = inputs.view(in_size[0], in_size[1], in_size[2], 7, 7)
        bn_inputs = self.bn_conv(re_inputs)
        conv_l1 = F.relu(self.conv1_3d(bn_inputs))
        conv_l2 = F.relu(self.conv2_3d(conv_l1))
        #inputs_bn = self.bn(conv_l2)
        re_insize = torch.tensor(conv_l2.size()).tolist()

        permute_input = conv_l2.view(re_insize[0], re_insize[1]*re_insize[2], re_insize[3]*re_insize[4]).permute(0, 2, 1)
        trans_out = self.transformer_model(permute_input, permute_input, self.bias.cuda())
        trans_out = F.relu(self.dropout1(trans_out.view(in_size[0], -1)))
        fc1_out = self.fc1(trans_out)
        cls_in = F.relu(self.dropout2(fc1_out))
        out = self.cls_fc(cls_in)
        if save_flag:
            return F.relu(fc1_out), out
        return out

class bilinear_model(nn.Module):
    def __init__(self, triplet_size):
        super(bilinear_model, self).__init__()
        self.fc_outsize = triplet_size

        self.bn_conv = nn.BatchNorm3d(num_features=30,
                                 track_running_stats=True)
        self.conv1 = nn.Conv2d(in_channels=512,
                               out_channels=256,
                               kernel_size=(3,3),
                               stride=(2,2),
                               padding=(1,1),
                               dilation=(1,1),
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=256,
                               out_channels=128,
                               kernel_size=(3,3),
                               stride=(2,2),
                               padding=(1,1),
                               dilation=(1,1),
                               bias=True)

        self.conv_fc = nn.Linear(128*4, 64)
        self.dropout = nn.Dropout(p=0.5)


        self.fc_insize = 64*30*30
        self.cls_dropout = nn.Dropout(p=0.3)
        self.cls_fc = nn.Linear(self.fc_insize, self.fc_outsize)

        self.obj_dropout = nn.Dropout(p=0.3)
        self.obj_fc = nn.Linear(self.fc_insize, 400)

        self.rela_dropout = nn.Dropout(p=0.3)
        self.rela_fc = nn.Linear(self.fc_insize, 90)

    def forward(self, inputs, save_flag=False):
        in_size = torch.tensor(inputs.size()).tolist()
        re_inputs = inputs.view(in_size[0], in_size[1], in_size[2], 7, 7)
        bn_inputs = self.bn_conv(re_inputs)
        re_inputs = bn_inputs.view(in_size[0]*in_size[1], in_size[2], 7, 7)
        conv1_out = F.relu(self.conv1(re_inputs))
        conv2_out = F.relu(self.conv2(conv1_out))

        refc_in = conv2_out.view(in_size[0], in_size[1], -1)
        in_1 = self.dropout(self.conv_fc(refc_in))
        in_2 = in_1
        bin_op = torch.matmul(in_1.transpose(1,2).unsqueeze(3), in_2.transpose(1,2).unsqueeze(2))
        tri_in = bin_op.view(in_size[0], -1)
        tri_out = self.cls_dropout(self.cls_fc(tri_in))
        rela_out = self.obj_dropout(self.rela_fc(tri_in))
        obj_out = self.rela_dropout(self.obj_fc(tri_in))

        if save_flag:
            return tri_in, cls_out
        return tri_out, obj_out, rela_out









class conv3d_model(nn.Module):
    def __init__(self, triple_size):
        super(conv3d_model, self).__init__()
        self.fc_outsize = triple_size
        '''
        self.avg_pool = nn.AvgPool3d(kernel_size=(3,3,3),
                                     stride=(2,2,2),
                                     padding=(0,0,0) )
        '''
        self.bn = nn.BatchNorm3d(num_features=30,
                                 track_running_stats=True)

        self.conv1_3d = nn.Conv3d(in_channels=30,
                                  out_channels=10,
                                  #kernel_size=(1,1,1),
                                  kernel_size=(3,3,3),
                                  stride=(4,2,2),
                                  #padding=(0,0,0),
                                  padding=(1,1,1),
                                  dilation=(1,1,1),
                                  bias=True)

        '''
        self.conv2_3d = nn.Conv3d(in_channels=20,
                                  out_channels=10,
                                  kernel_size=(3,3,3),
                                  stride=(2,1,1),
                                  padding=(1,1,1),
                                  dilation=(1,1,1),
                                  bias=True)
        '''
        self.conv3_3d = nn.Conv3d(in_channels=10,
                                  #out_channels=8,
                                  out_channels=4,
                                  #kernel_size=(1,1,1),
                                  kernel_size=(3,3,3),
                                  #stride=(4,2,2),
                                  stride=(2,1,1),
                                  #padding=(0,0,0),
                                  padding=(1,1,1),
                                  dilation=(1,1,1),
                                  bias=True)
        self.conv_fc_tri = nn.Conv3d(in_channels=4096,
                                 out_channels=2048,
                                 kernel_size=(1,1,1),
                                 stride=(1,1,1),
                                 padding=(0,0,0),
                                 dilation=(1,1,1),
                                 bias=True)

        self.conv_fc_cls = nn.Conv3d(in_channels=4096,
                                 out_channels=2048,
                                 kernel_size=(1,1,1),
                                 stride=(1,1,1),
                                 padding=(0,0,0),
                                 dilation=(1,1,1),
                                 bias=True)

        self.conv_fc_rela = nn.Conv3d(in_channels=4096,
                                 out_channels=2048,
                                 kernel_size=(1,1,1),
                                 stride=(1,1,1),
                                 padding=(0,0,0),
                                 dilation=(1,1,1),
                                 bias=True)
        self.fc_insize = 4096#2048 #5120#5120#1200*4#640
        self.fc1 = nn.Linear(self.fc_insize, 2048)
        self.dropout = nn.Dropout(p=0.3)
        self.retrieval_fc = nn.Linear(2048, self.fc_outsize, bias=False)
        self.obj_fc1 = nn.Linear(self.fc_insize, 2048)
        self.obj_dropout = nn.Dropout(p=0.3)
        self.obj_fc2 = nn.Linear(2048, 400)

        self.rela_fc1 = nn.Linear(self.fc_insize, 2048)
        self.rela_dropout = nn.Dropout(p=0.3)
        self.rela_fc2 = nn.Linear(2048, 90)

    def trans_conv(self, in_layer, out_channel, kernel_size, strides):
        layer_shape = in_layer.shape()
        trans_conv = nn.ConvTranspose2d(in_channels=layer_shape[0],
                                        out_channels=out_channel,
                                        kernel_size=kernel_size,
                                        stride=strides)
        return trans_conv

    def forward(self, inputs, save_flag=False):
        in_size = torch.tensor(inputs.size()).tolist()
        re_inputs = inputs.view(in_size[0], in_size[1], in_size[2], 7, 7)
        bs = in_size[0]
        #l1 = self.avg_pool(inputs)
        inputs_bn = self.bn(re_inputs)
        l1 = F.relu(self.conv1_3d(inputs_bn))
        #l2 = F.relu(self.conv2_3d(l1))
        l3 = F.relu(self.conv3_3d(l1))
        #l1 = self.pool_3d(inputs)
        #l2 = F.relu(self.conv2_3d(l1))

        #feat_to_fc = l3.view(bs, -1)
        feat_to_fc = l3.view(bs, 4096, 1, 1, 1)
        #fc_l1 = F.relu(self.dropout(self.fc1(feat_to_fc)))
        fc_l1 = F.relu(self.conv_fc_tri(feat_to_fc)).view(bs, -1)
        retrieval_out = self.retrieval_fc(fc_l1)

        #obj_l1 = F.relu(self.obj_dropout(self.obj_fc1(feat_to_fc)))
        obj_l1 = F.relu(self.conv_fc_cls(feat_to_fc)).view(bs, -1)
        obj_out = self.obj_fc2(obj_l1)

        #rela_l1 = F.relu(self.rela_dropout(self.rela_fc1(feat_to_fc)))
        rela_l1 = F.relu(self.conv_fc_rela(feat_to_fc)).view(bs, -1)
        rela_out = self.rela_fc2(rela_l1)

        if save_flag:
            return feat_to_fc, retrieval_out
        return retrieval_out, obj_out, rela_out


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m



if __name__ == '__main__':
    tmp_feature = torch.rand(4, 30, 512, 7, 7)
    trans_mod = cls_model(128)
    tmp = trans_mod(tmp_feature)
    pdb.set_trace()

