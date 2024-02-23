import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

momentum = 0.001


# def mish(x):
#     """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
#     return x * torch.tanh(F.softplus(x))
#
#
# class PSBatchNorm2d(nn.BatchNorm2d):
#     """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""
#
#     def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
#         super().__init__(num_features, eps, momentum, affine, track_running_stats)
#         self.alpha = alpha
#
#     def forward(self, x):
#         return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False, use_7x7_kernel=False,with_softmax=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        #self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
        #                       padding=1, bias=True)
        if use_7x7_kernel:
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2,
                                   padding=3, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                                   padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        
        
        self.with_softmax = with_softmax

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        output = self.fc(out)

        if ood_test:
            return output, out
        else:
            if self.is_remix:
                rot_output = self.rot_classifier(out)
                return output, rot_output
            else:
                if self.with_softmax:
                    return nn.functional.softmax(output,dim=1)
                return output


class build_WideResNet:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False,use_7x7_kernel=False,with_softmax=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix
        self.use_7x7_kernel = use_7x7_kernel
        self.with_softmax = with_softmax

    def build(self, num_classes):
        return WideResNet(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
            use_7x7_kernel=self.use_7x7_kernel,
            with_softmax=self.with_softmax
        )


def WRESNET(net_from_name,num_classes,
             leaky_slope=0.1, bn_momentum=0.001, dropRate=0.0,
             **kwargs):
    net_from_name = net_from_name.lower()
    if net_from_name == "wideresnet_52_2":
        return build_WideResNet(first_stride=1, depth=52, widen_factor=2,
                                    leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                    use_embed=False, is_remix=False, **kwargs).build(num_classes=num_classes)
    if net_from_name == "wideresnet_28_10":
        return build_WideResNet(first_stride=1, depth=28, widen_factor=10,
                                    leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                    use_embed=False, is_remix=False, **kwargs).build(num_classes=num_classes)
    if net_from_name == "wideresnet_28_2":
        return build_WideResNet(first_stride=1, depth=28, widen_factor=2,
                                    leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                    use_embed=False, is_remix=False, **kwargs).build(num_classes=num_classes)
    if net_from_name == "wideresnet_28_1":
        return build_WideResNet(first_stride=1, depth=28, widen_factor=1,
                                leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                use_embed=False, is_remix=False, **kwargs).build(num_classes=num_classes)
    if net_from_name.startswith("wideresnet_16_2"):
        use_7x7_kernel = False
        first_stride = 1
        if net_from_name.endswith("7x7"):
            use_7x7_kernel = True
            first_stride = 2
        return build_WideResNet(first_stride=first_stride, depth=16, widen_factor=2, use_7x7_kernel=use_7x7_kernel,
                                    leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                    use_embed=False, is_remix=False, **kwargs).build(num_classes=num_classes)
    if net_from_name.startswith("wideresnet_16_5"):
        use_7x7_kernel = False
        first_stride = 1
        if net_from_name.endswith("7x7"):
            use_7x7_kernel = True
            first_stride = 2
        return build_WideResNet(first_stride=first_stride, depth=16, widen_factor=5,use_7x7_kernel=use_7x7_kernel,
                                    leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                    use_embed=False, is_remix=False, **kwargs).build(num_classes=num_classes)
    if net_from_name.startswith("wideresnet_28_5"):
        use_7x7_kernel = False
        first_stride = 1
        if net_from_name.endswith("7x7"):
            use_7x7_kernel = True
            first_stride = 2
        return build_WideResNet(first_stride=first_stride, depth=28, widen_factor=5,
                                    leaky_slope = leaky_slope, bn_momentum=bn_momentum,dropRate=dropRate,
                                    use_embed=False, is_remix=False, use_7x7_kernel= use_7x7_kernel, **kwargs).build(num_classes=num_classes)
    return None

if __name__ == '__main__':
    name = "wideresnet_16_5_7x7"
    wrn = WRESNET(name,num_classes=2)
    # if 1:
    #     saved_path = r"D:\dev\projects\static-libs\torch\torchssl\codes\saved_models_weibosi_tujian\fixmatch_weibosi_tujian\model_best.pth"
    #     checkpoint = torch.load(saved_path, map_location=lambda storage, loc: storage.cuda(0))
    #     wrn.load_state_dict(checkpoint['model'], strict=True)
    #     torch.save(wrn.state_dict(),"x.pth")
    # else:
    #     weight = torch.load("x.pth")
    #     wrn.load_state_dict(weight, strict=True)
    input = torch.randn(1, 3, 224, 224)
    output = wrn(input)

    #print(wrn)
    print("wrn {} parameters: {:.3f}M ".format(name,sum(p.numel() for p in wrn.parameters()) / (1024*1024)))
    print(input.shape, output.shape)
