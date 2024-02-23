#!/usr/bin/env python
# -*- coding:utf-8 -*-


__author__ = "Facevisa_lilai"

import re
import struct
import torch
import pandas as pd

from torch import nn


# from torch.nn import functional as F
# import torchvision
# from torchsummaryX import summary


def summary_model(model: torch.nn.Module, related_layer_types=(nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
    def parse_param_bn(module):
        layer_param = []
        layer_param.append(module.num_features)
        layer_param.append(module.eps)
        return layer_param

    def parse_param_fc(module):
        layer_param = []
        layer_param.append(module.in_features)
        layer_param.append(module.out_features)

        if type(module.bias) == nn.parameter.Parameter:
            layer_param.append(1)
        else:
            layer_param.append(0)
        return layer_param

    def parse_param_conv(module):
        layer_param = []
        layer_param.append(module.in_channels)
        layer_param.append(module.out_channels)

        if type(module.kernel_size) == int:
            layer_param.append(module.kernel_size)
            layer_param.append(module.kernel_size)
        elif type(module.kernel_size) == tuple:
            layer_param.append(module.kernel_size[0])
            layer_param.append(module.kernel_size[1])

        if type(module.stride) == int:
            layer_param.append(module.stride)
            layer_param.append(module.stride)
        elif type(module.stride) == tuple:
            layer_param.append(module.stride[0])
            layer_param.append(module.stride[1])

        if type(module.padding) == int:
            layer_param.append(module.padding)
            layer_param.append(module.padding)
        elif type(module.padding) == tuple:
            layer_param.append(module.padding[0])
            layer_param.append(module.padding[1])

        if type(module.dilation) == int:
            layer_param.append(module.dilation)
            layer_param.append(module.dilation)
        elif type(module.dilation) == tuple:
            layer_param.append(module.dilation[0])
            layer_param.append(module.dilation[1])

        layer_param.append(module.groups)
        if module.bias is None:
            layer_param.append(0)
        else:
            layer_param.append(1)

        return layer_param

    info = []
    info_param = {}
    for name, module in model.named_modules():
        if type(module) == nn.Conv2d:
            info.append({"name": name, "module": module})
            info_param[name + ".weight"] = parse_param_conv(module)
        elif type(module) == nn.BatchNorm2d:
            info.append({"name": name, "module": module})
            info_param[name + ".weight"] = parse_param_bn(module)
        elif type(module) == nn.Linear:
            info.append({"name": name, "module": module})
            info_param[name + ".weight"] = parse_param_fc(module)

    df = pd.DataFrame(info)
    df = df.reindex(columns=["name", "module"])
    # print(df.to_markdown())

    return info_param


def Facevisa_info_count(net_params):
    total_info_num = 0
    if net_params.get("hwc2chw"):
        total_info_num += 1
    if net_params.get("bgr2rgb"):
        total_info_num += 1
    if net_params.get("vid255"):
        total_info_num += 1
    if net_params.get("mean"):
        total_info_num += 1
    if net_params.get("std"):
        total_info_num += 1
    return total_info_num


def Facevisa_weight_Convert(model,net_params, wts_name, prefix=""):
    #prefix: 模型变量的前缀将不保存到wts中

    # check operation
    total_info_num = Facevisa_info_count(net_params)

    # load
    device = torch.device('cpu')
    model.to(device).eval()

    # summary
    net_info = summary_model(model)

    # save
    f = open(wts_name, 'w')
    f.write("{}\n".format(len(model.state_dict().keys()) + len(net_info) + total_info_num))

    # fill preprocess info
    if net_params.get("hwc2chw"):
        f.write("{} {}".format("hwc2chw", 1))
        if net_params["hwc2chw"]:
            f.write(" ")
            f.write(struct.pack(">f", float(1.0)).hex())
        else:
            f.write(" ")
            f.write(struct.pack(">f", float(0.0)).hex())
        f.write("\n")

    if net_params.get("bgr2rgb"):
        f.write("{} {}".format("bgr2rgb", 1))
        if net_params["bgr2rgb"]:
            f.write(" ")
            f.write(struct.pack(">f", float(1.0)).hex())
        else:
            f.write(" ")
            f.write(struct.pack(">f", float(0.0)).hex())
        f.write("\n")

    if net_params.get("vid255"):
        f.write("{} {}".format("vid255", len(net_params["vid255"])))
        for vv in net_params["vid255"]:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    if net_params.get("mean"):
        f.write("{} {}".format("mean", len(net_params["mean"])))
        for vv in net_params["mean"]:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    if net_params.get("std"):
        f.write("{} {}".format("std", len(net_params["std"])))
        for vv in net_params["std"]:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    for k, v in model.state_dict().items():
        if prefix != "":
            knorm = re.sub(f"^{prefix}",'',k)
        else:
            knorm = k
        vr = v.reshape(-1).cpu().numpy()
        # record weight
        f.write("{} {}".format(knorm, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
        # record info
        if net_info.get(k):
            f.write("{}_info {}".format(knorm, len(net_info[k])))
            for idx in net_info[k]:
                f.write(" ")
                f.write(struct.pack(">f", float(idx)).hex())
            f.write("\n")


