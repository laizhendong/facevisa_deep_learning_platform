import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
import time
import platform

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *
from mobilenetv3 import mobilenetv3_large
import pdb



if platform.platform().lower().split('-')[0] == "windows":    
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' #!!!!!!!!!!!!!!!!!!!!!!!!

def main():
    if not len(sys.argv) == 3:
        print('Format:')
        print('python lib/demo.py config_file image_file')
        exit(0)
    experiment_name = sys.argv[1].split(os.path.sep)[-1][:-3]                         #配置文件名字
    data_name = sys.argv[1].split(os.path.sep)[-2]                                    #数据集名字
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
    image_file = sys.argv[2]                                                  #测试图片路径名
    package_name = os.path.abspath(__file__).split(os.path.sep)[-3]    
    my_config = importlib.import_module(config_path, package=package_name)
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)


    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

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

    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)

    weight_file = os.path.join(save_dir, cfg.test_weight)
    print('weight_file:',weight_file)
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize((cfg.input_size[1], cfg.input_size[0])), transforms.ToTensor(), normalize])


    #写pts
    def write_pts(path, pts, ver='version:1'):
        lines = [ver]
        lines.append('n_points:{}'.format(len(pts)))
        lines.append("{")
        for (x,y) in pts:
            lines.append('{} {}'.format(x,y))
        lines.append("}")
        with open(path,'w') as f:
            f.write('\n'.join(lines))
        return

    def demo_image(image_name,pic_name, net, preprocess, cfg, device):

        det_box_scale = 1.1        #1.2
        net.eval()
        image = cv2.imread(image_name)
        image_height, image_width, _ = image.shape
        
        if True:
            det_xmin = cfg.crop_size[0]                         
            det_ymin = cfg.crop_size[1]
            det_xmax = cfg.crop_size[2]
            det_ymax = cfg.crop_size[3]
            
            #找到x,y 中的最大值和最小值，定位出检测目标物框的大小
            det_width = det_xmax - det_xmin
            det_height = det_ymax - det_ymin
            scale = 1.1 
            det_xmin -= int((scale-1)/2*det_width)
            det_ymin -= int((scale-1)/2*det_height)                          #左上角的点进行微调，因为矩形框的宽和高对应放大1.1倍，x向左扩0.1*w/2
            det_width *= scale                                               #宽高放大1.1倍
            det_height *= scale
            det_width = int(det_width)
            det_height = int(det_height)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_width = min(det_width, image_width-det_xmin-1)               #越界判断
            det_height = min(det_height, image_height-det_ymin-1)
            
            det_xmax=det_xmin + det_width
            det_ymax=det_ymin + det_height
            
            #print('det_xmin:',det_xmin,'det_ymin:',det_ymin,'det_xmax:',det_xmax,'det_ymax:',det_ymax)
            cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax  , det_ymax), (0, 0, 255), 2)     
            
            #pic=cv2.imread('./data/DTY_BOTTOM_BOARDER/images_test/B_ori_38_defect_7_38_DTY_B25601AS17_20220309210016_original_WW_Q100_A@1646830816413.jpg')
            
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop, (cfg.input_size[0], cfg.input_size[1]))
            
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
        
        else:
            #官方给定的（可以参考）
            det_xmin = cfg.crop_size[0] 
            det_ymin = cfg.crop_size[1]
            det_width = cfg.crop_size[2]-cfg.crop_size[0]
            det_height = cfg.crop_size[3]-cfg.crop_size[1]
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (det_box_scale-1)/2)
            # remove a part of top area for alignment, see paper for details
            det_ymin += int(det_height * (det_box_scale-1)/2)
            det_xmax += int(det_width * (det_box_scale-1)/2)
            det_ymax += int(det_height * (det_box_scale-1)/2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width-1)
            det_ymax = min(det_ymax, image_height-1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            
            #print('det_xmin:',det_xmin,'det_ymin:',det_ymin,'det_xmax:',det_xmax,'det_ymax:',det_ymax)
            
            cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
            
            #pic=cv2.imread('./data/DTY_BOTTOM_BOARDER/images_test/B_ori_38_defect_7_38_DTY_B25601AS17_20220309210016_original_WW_Q100_A@1646830816413.jpg')
            
            det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
            det_crop = cv2.resize(det_crop,(cfg.input_size[0], cfg.input_size[1]))
            inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
        
        
        last_time = time.time()
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
        fr_time = time.time()-last_time
        print('time of each pic is',fr_time)
        
        #输出结果处理
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        #tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        #tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        #tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        #tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        
        tmp_x = torch.mean(lms_pred_x, dim=1).view(-1,1)
        tmp_y = torch.mean(lms_pred_y,dim=1).view(-1,1)
        
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        #print('lms_pred_merge:',lms_pred_merge)
        
        point_list=[]
        for i in range(cfg.num_lms):
            point=[]
            x_pred = lms_pred_merge[i*2] * det_width + det_xmin 
            y_pred = lms_pred_merge[i*2+1] * det_height+ det_ymin
            cv2.circle(image, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 2)
            point.append(int(x_pred))
            point.append(int(y_pred))
            point_list.append(point)
            #print ('Point_x:',point[0],'Point_y:',point[1])
        
        name_crop_tmp = pic_name.split('.')
        com_name_tmp = name_crop_tmp[0].split('/')[-1]
        ptsname = com_name_tmp +'.pts'
        print('image_name:',pic_name)
        
        
        pts_dir='images/forward/pts/'
        if not os.path.exists(pts_dir):
            os.makedirs(pts_dir)
        
        mark_dir='images/forward/mark/'
        if not os.path.exists(mark_dir):
            os.makedirs(mark_dir)
        
        #保存pts
        write_pts(pts_dir+ ptsname, point_list)
        
        #保存mark
        cv2.imwrite(mark_dir+ pic_name, image)

        return fr_time    
    
    def _main_run():
        for root, dirs, files in os.walk(image_file):      
            print('root:', root)                    #获取文件所属目录
            count=0
            time_sum=0
            for id , file in enumerate(files):
                #pdb.set_trace()
                #获取文件路径
                name_crop = file.split('.')
                com_name= name_crop[0]
                last_name=name_crop[1]
                pic_name=''
                if last_name !='pts':
                    if last_name =='jpg':
                        image_name = os.path.join(root, com_name+'.jpg')
                        pic_name=com_name+'.jpg'
                    elif last_name =='png':
                        image_name = os.path.join(root, com_name + '.png')
                        pic_name=com_name+'.png'
                    elif last_name =='bmp':
                        image_name = os.path.join(root, com_name + '.bmp')
                        pic_name=com_name+'.bmp'
                    else:
                        image_name=os.path.join(root, com_name + '.jpeg')
                        pic_name=com_name+'.jpeg'
                    count+=1

                    print('idx is:', count)   

                    infr_time =demo_image(image_name, pic_name, net, preprocess, cfg, device)
                    time_sum += infr_time
                    
            print("inference time is :",time_sum*1.0/count)
    _main_run()

if __name__ == "__main__":
    main()