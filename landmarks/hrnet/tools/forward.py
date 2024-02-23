#----duanting---------
#----2022.04.18-------

#----inference--------
#step1：准备前向数据--
#step2：进行前向------
#step3: 保存前向结果--


####################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xml.etree import ElementTree as ET



from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
import argparse
import os 

#需要改
import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform
from utils.utils import get_model_summary

import pdb


#读xml函数
def read(xml_path,voc_only = False,poly2rect=True):

    bbox_list = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
 
    sizes = root.findall('size')[0]
    width = int(float(sizes.findall('width')[0].text))
    height = int(float(sizes.findall('height')[0].text))
    depth = int(float(sizes.findall('depth')[0].text))
    for child in root.findall('object'):
        names = child.findall('name')
        if len(names) == 0 or names[0].text == None:
            name = ''
        else:
            name = names[0].text.strip()
        level = int(float((child.findall('level')[0].text)))
        find_rect = False
        if not voc_only:
            for shape in child.findall('shape'):
                if shape.attrib["type"] == "rect":
                    find_rect = True
                    points = shape.findall('points')[0]
                    xmin = int(points.findall('x')[0].text)
                    ymin = int(points.findall('y')[0].text)
                    xmax = int(points.findall('x')[1].text)
                    ymax = int(points.findall('y')[1].text)
                    info = {"type": "rect",
                            "points": [(xmin,ymin), (xmax,ymax)],
                            "color": "white",
                            "name": name,
                            "level": level}
                    bbox_list.append(info)
                elif shape.attrib['type'] == 'bezier':
                    points = shape.findall('points')[0]
                    x0 = int(points.findall('x')[0].text)
                    y0 = int(points.findall('y')[0].text)
                    x1 = int(points.findall('x')[1].text)
                    y1 = int(points.findall('y')[1].text)
                    x2 = int(points.findall('x')[2].text)
                    y2 = int(points.findall('y')[2].text)
                    control_pts = [(x0,y0),(x1,y1),(x2,y2)]

                    info = {"type":"bezier",
                            "points":control_pts,
                            "color":shape.attrib['color'],
                            "name":name,
                            "level":level}
                    bbox_list.append(info)
                elif shape.attrib['type'] == 'pen':
                    points = shape.findall('points')[0]
                    xlist, ylist = points.findall('x'), points.findall('y')
                    points_list = []
                    for x, y in zip(xlist, ylist):
                        x,y = int(x.text), int(y.text)
                        points_list.append((x,y))
                    info = {"type":"pen",
                            "points":points_list,
                            "color":shape.attrib['color'],
                            "name":name,
                            "level":level}
                    bbox_list.append(info)
                elif shape.attrib['type'] == 'polygon':
                    points = shape.findall('points')[0]
                    xlist, ylist = points.findall('x'), points.findall('y')
                    X,Y = [], []
                    for x, y in zip(xlist, ylist):
                        x, y = int(x.text), int(y.text)
                        X.append(x)
                        Y.append(y)
                    x0,x1 = min(X), max(X)
                    y0,y1 = min(Y), max(Y)
                    info = {"type": "rect",
                            "points": [(x0,y0), (x1, y1)],
                            "color": "white",
                            "name": name,
                            "level": level}
                    bbox_list.append(info)
                else:
                    raise Exception("unk type:{}".format(shape.attrib['type']))
                    

        if not find_rect and voc_only:
            for bndbox in child.findall('bndbox'):
                xmin = int(bndbox.findall('xmin')[0].text)
                ymin = int(bndbox.findall('ymin')[0].text)
                xmax = int(bndbox.findall('xmax')[0].text)
                ymax = int(bndbox.findall('ymax')[0].text)
                info = {"type": "rect",
                        "points": [(xmin,ymin), (xmax, ymax)],
                        "color": "white",
                        "name": name,
                        "level": level}
                bbox_list.append(info)
    return {"bboxes":bbox_list,"image_info":(width,height,depth)}

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
    
def get_person_detection_boxes(xml_read):
    #拿到框的信息
    bboxes=xml_read['bboxes']
    #[(x1,y1),(x2,y2)]
    points=bboxes[0]['points']    
    box=[]
    x1=points[0][0]
    y1=points[0][1]
    x2=points[1][0]
    y2=points[1][1]
    
    box.append(x1)
    box.append(y1)
    box.append(x2)
    box.append(y2)
    
    return points
    

def box_to_center_scale(box, model_image_width, model_image_height,aspect_ratio):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    
    
    center = np.zeros((2), dtype=np.float32)
    
    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = aspect_ratio
    pixel_std = 200

    #pdb.set_trace()
    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


#数据制作并前向推理
def get_pose_estimation_prediction(pose_model, image, center, scale,crop_dir,pic_name):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
        
        
    data_image_affine = model_input.copy()
    cv2.imwrite(crop_dir + pic_name, data_image_affine)
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        start = time.time()
        output = pose_model(model_input)
        end = time.time()
        print('inference time is: %.3f ms'%((end-start)*1000))
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds

#保存生成的pts
def save_batch_final_joints_image_pts(img_path,input, final_joints_pred, outputdir):
    #input ---- [H W 3]
    #final_joints_pred ---- N C  2  经过仿射变换之后的逆矩阵之后的点
    filename=img_path
    ptsname=''   
    #pdb.set_trace()
    for i in range(final_joints_pred.shape[0]):   
        point_list=[]
    
        name_crop = filename.split('.')
        com_name= name_crop[0].split('/')[-1]
        ptsname=com_name+'.pts'
        imagename = filename.split('/')[-1]   
        
        
        #遍历每一个关键点
        for idx in range(final_joints_pred.shape[1]):
            point = final_joints_pred[i,idx,:]
            point_list.append(point.tolist())
                              
    
        ptspath=os.path.join(outputdir, ptsname)
        imgpath=os.path.join(outputdir, imagename)
        
        write_pts(ptspath, point_list)
        cv2.imwrite(imgpath, input)


def parse_args():
    parser = argparse.ArgumentParser(description='inference keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/mpii/hrnet/inference-config.yaml')
    parser.add_argument('--src_dir',type=str,default='')
    parser.add_argument('--showFps',default=True, action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args



def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    
    #导入模型
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    
    #加载模型参数
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)      
    else:
        print('expected model defined in config at TEST.MODEL_FILE')
        
    
    #设置显卡号，启动GPU进行多卡前向
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")
    
    
    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    information = get_model_summary(pose_model, dump_input)
    print(information)
    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS).cuda()
    pose_model.eval()   
    

    
#step1：准备前向数据--
    #读入图片和对应的XML，经过前向生成预测的关键点pts
      
    for root, dirs, files in os.walk(args.src_dir):      
        print('root:', root)    #获取文件所属目录
        count=0
        for id , file in enumerate(files):
            #pdb.set_trace()
            #获取文件路径
            
            name_crop = file.split('.')
            com_name= name_crop[0]
            last_name=name_crop[1]
            
            pic_name=''
            if last_name !='xml':
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
                xml_name = os.path.join(root, com_name+'.xml')
            
                print('idx is:', count)   
                image_bgr = cv2.imread(image_name)
                xml_read = read(xml_name)

                last_time = time.time()

                image = image_bgr[:, :, [2, 1, 0]]   


                pred_boxes = get_person_detection_boxes(xml_read)
                
                #根据框的信息转成中心点和scale
                center, scale = box_to_center_scale(pred_boxes, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1], cfg.TEST.ASPECT_RATIO)    

                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
            
#step2：进行前向------
                
                #返回到原图上的坐标
              
                if not os.path.exists( cfg.OUTPUT_DIR +'/inference/input_crop' ):  
                    os.makedirs(cfg.OUTPUT_DIR +'/inference/input_crop' )
                    
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale,cfg.OUTPUT_DIR +'/inference/input_crop/',pic_name)   

                if args.showFps:
                    fps = 1/(time.time()-last_time)
                    print('fps of each pic is',fps)
                
                
#step3: 保存前向结果--
                
                if not os.path.exists( cfg.OUTPUT_DIR +'/inference/inference_rlt/' ):  
                    os.makedirs(cfg.OUTPUT_DIR +'/inference/inference_rlt/' )
                
                save_batch_final_joints_image_pts(image_name, image_bgr, pose_preds, cfg.OUTPUT_DIR +'/inference/inference_rlt/')
    

if __name__ == '__main__':
    main()

























