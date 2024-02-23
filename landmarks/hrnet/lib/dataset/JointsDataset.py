# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
from PIL import Image
import cv2,copy
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints 
from utils.transforms import ColorJitter 
import pdb 
from utils.vis import save_batch_image_with_joints

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []
        
        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        #子类重写了该变量，在父类进行了申明，因此是可以调用的
        self.db = []                                                             

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):

        #for循环触发了dataloader
        #都是对一张图片操作，对应生成的
        db_rec = copy.deepcopy(self.db[idx])
        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        #pdb.set_trace()
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
            
        orijoints = copy.deepcopy(db_rec['joints_3d'])
        joints = copy.deepcopy(db_rec['joints_3d'])
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0
        if True:
            
            if self.is_train:
     
                if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                   
                    c_half_body, s_half_body = self.half_body_transform(
                        joints, joints_vis
                    )

                    if c_half_body is not None and s_half_body is not None:
                        c, s = c_half_body, s_half_body

                sf = self.scale_factor                                                       
                rf = self.rotation_factor                                                  
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)                    
                r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0           
               
               
                if self.flip and random.random() <= 0.5:
                    data_numpy = data_numpy[:, ::-1, :]                                       #将图片进行翻转
                    joints, joints_vis = fliplr_joints(
                        joints, joints_vis, data_numpy.shape[1], self.flip_pairs)             #换成自己的数据集是此处的数据增强需要关掉
                    c[0] = data_numpy.shape[1] - c[0] - 1
                    
                #if random.random() <= 0.5:
                if False:
                    color=ColorJitter(0.5,0.5,0.5)(Image.fromarray(np.uint8(data_numpy)))     #输入数据个是为PIL
                    data_numpy=np.asarray(color)                                              #将PIL转numpy
                   
           

            trans= get_affine_transform(c, s, r, self.image_size)                              #原图经过变换，得到仿射后的矩阵，此处做了一个图片的尺寸变换
            input = cv2.warpAffine(                                                            #验证阶段的图是不进行数据增强的，但是做了反射变换，回归原图需要逆回来
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)                                                        
          
            if self.transform:
                input = self.transform(input)
            
            #坐标点做与图片相同的变换    
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)    
            
                   
        
        else:        
            #换一种数据制作方式，不进行数据增强
            #step1:跟据scale和center把坐标框挖出来
            #step2:把原图上的关键点映射回挖出来的块
            #step3:再进行ressize

            box_w=s[0]*200/1.25
            box_h=s[1]*200/1.25


            x1 = int(np.ceil(c[0]-0.5 * box_w))
            y1 = int(np.ceil(c[1]-0.5 * box_h))
            x2 = int(np.ceil(c[0]+0.5*box_w))
            y2 = int(np.ceil(c[1]+0.5*box_h))

            image = data_numpy[y1:y2,x1:x2,:]
            #print(db_rec['image'])

            for i in range(self.num_joints):    
                joints[i,0] = joints[i,0]-x1
                joints[i,1] = joints[i,1]-y1
            
            
            input=cv2.resize(image,(int(self.image_size[0]), int(self.image_size[1])))
            
            #大图比上resize之后的图片
            offerx=image.shape[1]*1.0/int(self.image_size[0])    
            offery=image.shape[0]*1.0/int(self.image_size[1])
            
            #pdb.set_trace()
            for i in range(self.num_joints):    
                joints[i,0] = joints[i,0]/offerx
                joints[i,1] = joints[i,1]/offery
            
            
            if False:
                m=0
                for p in joints:
                     cv2.circle(input, (int(p[0]), int(p[1])), 2, (0, 255, 0),-1)
                     cv2.putText(input, str(m), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)
                     m+=1
                pdb.set_trace()
                cv2.imwrite(self.root+'rlt/pic/'+str(idx)+'pic.jpg', input)
        
        #if self.transform:
        #input_imag = self.transform(input)
        
        #根据小图上的关键点坐标和关键点可见与否生成gt热图和训练的损失加权    
        target, target_weight = self.generate_target(joints, joints_vis)    
        
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
       

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'orijoints': orijoints
        }
        
        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
       
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                #生成了一张高斯热图
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))     

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    #将生成的高斯热图的点移到含有关键点位置上的gt图上去，则生成gt的高斯热图
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]                                   

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)
        
        return target, target_weight
