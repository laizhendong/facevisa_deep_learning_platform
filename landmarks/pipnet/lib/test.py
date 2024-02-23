import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import pickle
import importlib
import time

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from networks import *
from functions import *
from mobilenetv3 import mobilenetv3_large
import platform


if platform.platform().lower().split('-')[0] == "windows":    
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #!!!!!!!!!!!!!!!!!!!!!!!!
    
#pdb.set_trace()
if not len(sys.argv) == 4:
    print('Format:')
    print('python lib/test.py config_file test_labels test_images')
    exit(0)
experiment_name = sys.argv[1].split(os.path.sep)[-1][:-3]
data_name = sys.argv[1].split(os.path.sep)[-2]                                    #数据集名字
test_labels = sys.argv[2]                                                 #测试标签
test_images = sys.argv[3]                                                 #测试图片
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
package_name = os.path.abspath(__file__).split(os.path.sep)[-3]  
my_config = importlib.import_module(config_path, package=package_name)        #相对导入
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name                                     #增加config配置
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.mkdir(os.path.join('./logs', cfg.data_name))
log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if cfg.det_head == 'pip':
    ##  meanface_indices：每一个关键点与均值关键点距离最小的排序后距离从小到大排序的前10对应的关键点索引  （{0：【10个】}）
    ##  reverse_index1, reverse_index2     （600）
    ##  max_len-----num_nb

    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
    
if cfg.det_head == 'pip':
    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=cfg.pretrained)
        net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=cfg.pretrained)
        net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v2':
        mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
        net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    elif cfg.backbone == 'mobilenet_v3':
        mbnet = mobilenetv3_large()
        if cfg.pretrained:
            mbnet.load_state_dict(torch.load('lib/mobilenetv3-large-1cd25616.pth'))
        net = Pip_mbnetv3(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    else:
        print('No such backbone!')
        exit(0)
else:
    print('No such head:', cfg.det_head)
    exit(0)


if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

net = net.to(device)

#weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
weight_file = os.path.join(save_dir, cfg.test_weight)
state_dict = torch.load(weight_file)
net.load_state_dict(state_dict)
 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size[1], cfg.input_size[0])), transforms.ToTensor(), normalize])

norm_indices = None
if cfg.data_name == 'data_300W':
    norm_indices = [36, 45]
elif cfg.data_name == 'DTY_TOP':
    pass
elif cfg.data_name == 'DTY_BOTTOM_BOARDER':
    pass
elif cfg.data_name == 'KPDTYSidebottomborderbrightBobbinOuterSilkOuter':
    pass
else:
    print('No such data!')
    exit(0)

labels = get_label(cfg.data_name, test_labels)                               #将字符的形式转成float型

nmes_std = []
nmes_merge = []
mses_merge_big = []
mses_merge_crop=[]
norm = None
time_all = 0
for label in labels:                                                          #遍历每一张图
    image_name = label[0]
    print('image_name:',image_name)
    lms_gt = label[1]                                                         #关键点标签
    if cfg.data_name == 'data_300W' :
        norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])
    elif cfg.data_name == 'DTY_TOP':
        norm = 1
    elif cfg.data_name == 'DTY_BOTTOM_BOARDER':
        norm = 1
    else:
        norm = 1
    image_path = os.path.join('data', cfg.data_name, test_images, image_name)
    print("image_path:",image_path)
    name_crop = image_name.split('_')  
    
    pic_name = image_name[6:]
    print("pic_name:",pic_name)
    #ori_path = os.path.join('data', cfg.data_name, name_crop[0] ,name_crop[1]  ,name_crop[2],image_name[9:])
    ori_path = os.path.join('data', cfg.data_name, name_crop[0], pic_name)
    print('ori_path:',ori_path)
   
    image = cv2.imread(image_path)
    ori = cv2.imread(ori_path)
    
    image = cv2.resize(image, (cfg.input_size[0], cfg.input_size[1]))
    inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
    inputs = preprocess(inputs).unsqueeze(0)                            

    inputs = inputs.to(device)
    t1 = time.time()
    if cfg.det_head == 'pip':
        ###  x ,y 相对最大得分格子左上角的偏移量 lms_pred_x, lms_pred_y  
        ###  x,y 邻近关键点相对于最大得分格子左上角偏移量  lms_pred_nb_x, lms_pred_nb_y 
        ###  outputs_cls----输出格子得分图索引 
        ###  max_cls---格子的最大得分分值
        
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
    else:
        print('No such head!')
    
   
    if cfg.det_head == 'pip':
        # merge neighbor predictions
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()                      
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len) 
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)       
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)       
        
        #tmp_x = torch.mean(lms_pred_x, dim=1).view(-1,1)
        #tmp_y = torch.mean(lms_pred_y,dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()                          
    t2 = time.time()
    time_all += (t2-t1)

    if cfg.det_head == 'pip':
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
    
    nme_std = compute_nme(lms_pred_merge, lms_gt, norm)
    nmes_std.append(nme_std)

    if cfg.det_head == 'pip':
        nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)                    #lms_gt就是个格子中的偏移量（归一化后的）
        mse_merge_big = MSE_eval_points_big(lms_pred_merge, lms_gt,norm,ori, pic_name,cfg.crop_size)
        mse_merge_crop = MSE_eval_points_crop(lms_pred_merge, lms_gt,norm,image, pic_name)
        nmes_merge.append(nme_merge)
        mses_merge_big.append(mse_merge_big)
        mses_merge_crop.append(mse_merge_crop)

print('Total inference time:', time_all)
print('Image num:', len(labels))
print('Average inference time:', time_all/len(labels))

if cfg.det_head == 'pip':
    print('nme: {}'.format(np.mean(nmes_merge)))
    logging.info('nme: {}'.format(np.mean(nmes_merge)))  
    
    print('mse_big: {}'.format(np.mean(mses_merge_big)))
    logging.info('mse_big: {}'.format(np.mean(mses_merge_big)))
    
    print('mse_crop: {}'.format(np.mean(mses_merge_crop)))
    logging.info('mse_crop: {}'.format(np.mean(mses_merge_crop)))

    fr, auc = compute_fr_and_auc(nmes_merge)
    #print('fr : {}'.format(fr))
    #logging.info('fr : {}'.format(fr))
    print('auc: {}'.format(auc))
    logging.info('auc: {}'.format(auc))
