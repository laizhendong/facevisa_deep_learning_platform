import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
#from symbol.frn_layer import FRNLayer2d
import os
__all__ = ['RESNET']

model_urls = {
    #'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    "resnet18": os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",'pretrained_models','resnet18-5c106cde.pth'),
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 use_global_context = False,
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

        if use_global_context:
            self.container = nn.Conv2d((64+128+256+512) * block.expansion,512,kernel_size=1,stride=1)
        else:
            self.container = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        
        # self.conv128 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=(1,1)), nn.BatchNorm2d(128)) #
        #
        # self.conv256 = nn.Sequential(nn.Conv2d(448, 256, kernel_size=(1,1)), nn.BatchNorm2d(256)) # #nn.Conv2d(448, 256, kernel_size=(1,1))


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

    # def _forward_impl_fused(self, x):
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #
    #
    #     x = self.layer1(x)  # c=64
    #     temp_1 = nn.functional.avg_pool2d(x, kernel_size=(2,2), stride=2) #  ×8
    #
    #     x = self.layer2(x)  # c=128 ×8
    #     x = torch.cat((x, temp_1), axis=1)  # ×8  : c_in = 64 + 128 = 192  c_out
    #     x = self.conv128(x)
    #     temp_2 = nn.functional.avg_pool2d(x, kernel_size=(2,2), stride=2)   # × 16
    #
    #     x = self.layer3(x)  # ×16  c=256
    #     temp_3 = nn.functional.avg_pool2d(temp_1, kernel_size=(2,2), stride=2) #  ×8
    #     x = torch.cat((x, temp_2, temp_3), axis=1)  # ×16 : c = 256 + 128 + 64 =
    #     x = self.conv256(x)
    #
    #     x = self.layer4(x) # c=512
    #     x = self.avgpool(x)
    #
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #
    #     return x


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        if self.container is None:
            x = self.layer1(x)  # c=64
            x = self.layer2(x)  # c=128 ×8
            x = self.layer3(x)  # ×16  c=256
            x = self.layer4(x) # c=512
        else:
            x1 = self.layer1(x)  # c=64
            x2 = self.layer2(x1)  # c=128 ×8
            x3 = self.layer3(x2)  # ×16  c=256
            x4 = self.layer4(x3)  # c=512
            global_context = []
            for ks,f in zip([8,4,2,1],[x1,x2,x3,x4]):
                if ks > 1:
                    f = nn.AvgPool2d(kernel_size=ks,stride=ks)(f)
                ff = torch.pow(f,2)
                ff_mean = torch.mean(ff)
                f =  torch.div(f,ff_mean)
                global_context.append(f)
            x = self.container(torch.cat(global_context,dim=1))


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if isinstance(self.fc,torch.nn.Linear):
            x = self.fc(x)
        elif isinstance(self.fc,torch.nn.ModuleList):
            outs = [ fc(x) for fc in self.fc ]
            x = outs
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, load_strict=True, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls[arch])
        state_dict = torch.load(model_urls[arch],map_location=None)
        model.load_state_dict(state_dict,strict=load_strict)
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

#
# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)
#
#
# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
#                    pretrained, progress, **kwargs)
#

class RESNET(nn.Module):
    def __init__(self,backbone_name,input_channels,num_classes, pretrained_backbone, with_softmax=False):
        #pretrained_backbone: pretrained model to loaded
        super(RESNET, self).__init__()
        name = backbone_name.lower()
        self.with_softmax = with_softmax
        if name == "resnet18":
            self.backbone = resnet18(pretrained_backbone)#, norm_layer=FrozenBatchNorm2d
        elif name == "resnet18gc":
            self.backbone = resnet18(pretrained_backbone,load_strict=False, use_global_context=True)
        elif name == "resnet50":
            self.backbone = resnet50(pretrained_backbone)
        #elif name == "resnet50frn":
        #    self.backbone = resnet50(pretrained_backbone,load_strict = False, norm_layer=FRNLayer2d)
        elif name == "resnet101":
            self.backbone = resnet101(pretrained_backbone)

        in_features = self.backbone.fc.in_features
        outputs = []
        for _,num_class in enumerate(num_classes):
            outputs.append( nn.Linear(in_features, num_class) )
            torch.nn.init.xavier_uniform_(outputs[-1].weight)
            outputs[-1].bias.data.fill_(0.0)
        if len(outputs) == 1:
            self.backbone.fc = outputs[0]
        else:
            self.backbone.fc = torch.nn.ModuleList(outputs)
        if input_channels != 3:
            self.foot = nn.Conv2d(in_channels=input_channels,out_channels=3,kernel_size=1,stride=1,bias=False)
            torch.nn.init.xavier_uniform_(self.foot.weight)
        else:
            self.foot = None
        return
    def forward(self, x):
        if self.foot is None:
            if self.with_softmax:
                y = self.backbone(x)
                if isinstance(y, list):
                    return [nn.functional.softmax(o,  dim=1) for o in y]
                else:
                    return nn.functional.softmax(y,  dim=1)
            return self.backbone(x)

        if self.with_softmax:
            y = self.backbone(self.foot(x))
            if isinstance(y,list):
                return [nn.functional.softmax(o,  dim=1) for o in y]
            else:
                return  nn.functional.softmax(y,  dim=1)
        return self.backbone(self.foot(x))


if __name__ == "__main__":

    net = resnet18(pretrained=False,num_classes=2,use_global_context=True)

    input = torch.randn(1,3, 320//2,320//2)

    output = net(input)
    print(net)
    print("resnet18 parameters: {:.3f}M ".format(sum(p.numel() for p in net.parameters()) / (1024*1024)))