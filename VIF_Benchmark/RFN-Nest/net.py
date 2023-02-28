import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-10


def var(x, dim=0):
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)


class MultConst(nn.Module):
    def forward(self, input):
        return 255*input


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        # out_channels_def = 16
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


class FusionBlock_res(torch.nn.Module):
    def __init__(self, channels, index):
        super(FusionBlock_res, self).__init__()
        ws = [3, 3, 3, 3]
        self.conv_fusion = ConvLayer(2*channels, channels, ws[index], 1)

        self.conv_ir = ConvLayer(channels, channels, ws[index], 1)
        self.conv_vi = ConvLayer(channels, channels, ws[index], 1)

        block = []
        block += [ConvLayer(2*channels, channels, 1, 1),
                  ConvLayer(channels, channels, ws[index], 1),
                  ConvLayer(channels, channels, ws[index], 1)]
        self.bottelblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        # print('conv')
        f_cat = torch.cat([x_ir, x_vi], 1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi) # 原来的代码有问题，写成了conv_ir，现在重新训练
        out = torch.cat([out_ir, out_vi], 1)
        out = self.bottelblock(out)
        out = f_init + out
        return out


# Fusion network, 4 groups of features
class Fusion_network(nn.Module):
    def __init__(self, nC, fs_type):
        super(Fusion_network, self).__init__()
        self.fs_type = fs_type

        self.fusion_block1 = FusionBlock_res(nC[0], 0)
        self.fusion_block2 = FusionBlock_res(nC[1], 1)
        self.fusion_block3 = FusionBlock_res(nC[2], 2)
        self.fusion_block4 = FusionBlock_res(nC[3], 3)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp


class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp


class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp


class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# fuison strategy based on nuclear-norm (channel attention form NestFuse)
class Fusion_Nuclear(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        # calculate channel attention
        global_p1 = nuclear_pooling(en_ir)
        global_p2 = nuclear_pooling(en_vi)

        # get weight map
        global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
        global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

        global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
        global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

        tensor_f = global_p_w1 * en_ir + global_p_w2 * en_vi
        return tensor_f


# sum of S V for each chanel
def nuclear_pooling(tensor):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, fs_type):
        super(Fusion_strategy, self).__init__()
        self.fs_type = fs_type
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_spa = Fusion_SPA()
        self.fusion_nuc = Fusion_Nuclear()

    def forward(self, en_ir, en_vi):
        if self.fs_type is 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type is 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type is 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type is 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type is 'nuclear':
            fusion_operation = self.fusion_nuc

        f1_0 = fusion_operation(en_ir[0], en_vi[0])
        f2_0 = fusion_operation(en_ir[1], en_vi[1])
        f3_0 = fusion_operation(en_ir[2], en_vi[2])
        f4_0 = fusion_operation(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]


# NestFuse network - light, no desnse
class NestFuse_light2_nodense(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(NestFuse_light2_nodense, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        # # no short connection
        # self.DB1_2 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        # self.DB2_2 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        # self.DB1_3 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)

        # short connection
        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_2 = block(nb_filter[1] * 2+ nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1)

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            # self.conv4 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def decoder_train(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))
        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]


