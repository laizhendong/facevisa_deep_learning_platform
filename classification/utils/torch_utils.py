import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    if type(device) is int:
        device = str(device)

    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logging.info("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                        (s, i, x[i].name, x[i].total_memory / c))
    else:
        logging.info('Using CPU')

    logging.info('')  # skip a line
    return torch.device('cuda:0'.format(device) if cuda else 'cpu')


def get_num_classes(trainset_file):
    with open(trainset_file) as f:
        info = f.readlines()
    
    max_index = -1
    for i in info:
        class_index = int(i.strip().split(' ')[-1])
        if class_index > max_index:
            max_index = class_index

    return max_index+1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)