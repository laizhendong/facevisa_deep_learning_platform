# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict
from PIL import Image
import numpy as np
from scipy.io import loadmat, savemat
import numpy as np
from dataset.JointsDataset_wflw import JointsDataset_wflw
import pandas as pd 
import pdb


logger = logging.getLogger(__name__)


class WFLWDataset(JointsDataset_wflw):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)                     #继承
        
        self.num_joints = 98
        self.filp_pairs=[[0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
             [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
             [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
             [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73],
             [55, 59], [56, 58],
             [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
             [88, 92], [89, 91], [95, 93], [96, 97]]

        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
        self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET
       
        self.db = self._get_db()
        
        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)
 
        
        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        
        
        # load annotations
        landmarks_frame = pd.read_csv(self.csv_file)
        #pdb.set_trace()
        gt_db = []
        for idx in range(len(landmarks_frame)):
            #获取图片路径
            image_path = os.path.join(self.root,
                landmarks_frame.iloc[idx, 0]
            )
            
            
            scale_tmp =  np.array(landmarks_frame.iloc[idx, 1],dtype=np.float)                        #拿到scale
            scale = np.append(scale_tmp,landmarks_frame.iloc[idx, 1])                                  #拿到scale
            scale *= 1.25
            
           
            center_tmp =np.array(landmarks_frame.iloc[idx, 2],dtype=np.float) 
            center = np.append(center_tmp,landmarks_frame.iloc[idx, 3])                             #读取中心点x #中心点y
            
            
            
            pts = landmarks_frame.iloc[idx, 4:].values                                              #取关键点的值，给reshape成[98,2]
            pts = pts.astype('float').reshape(-1, 2)
            
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float)
            joints_3d[:,0:2]= pts[:]
            #
            for idx_p in range(self.num_joints):
                if self.image_set != 'test':
                    joints_3d[idx_p, 0:2] = pts[idx_p] - 1
                    t_vis = 1
                    if joints_3d[idx_p][0] < 0 or joints_3d[idx_p][1] < 0:
                        t_vis = 0
                    joints_3d_vis[idx_p, 0] = t_vis
                    joints_3d_vis[idx_p, 1] = t_vis
                    joints_3d_vis[idx_p, 2] = 0
                    
            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'        
            gt_db.append(
                {
                    'image': os.path.join( self.root,image_dir,landmarks_frame.iloc[idx, 0]),
                    'center': center,
                    'scale': scale,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                })
        #pdb.set_trace()
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})
            #保存为preds.mat

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               'gt_{}.mat'.format(cfg.DATASET.TEST_SET))
        
        #导入gt_vaild.mat 
        gt_dict = loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        #取关键点的坐标
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        #计算pckh
        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
