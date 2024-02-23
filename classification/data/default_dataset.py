import sys,os
#sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.augments as augops
import cv2
import random
from collections import defaultdict


class BasicDataset(Dataset):
    def __init__(self,cfg,list_file, train_flag=True):
        self.items = []
        self.train_gray_image = cfg.DATA.TRAIN_GRAY_IMAGE
        self.total_each_label = {}
        if list_file:
            if list_file[0] == '.':
                list_file = os.path.join(os.path.dirname(__file__),'..',list_file)
                print(f"switch relative path to {list_file}")
            with open(list_file, 'r') as f:
                for line in f:
                    try:
                        items = line.strip().split(' ')
                        path = items[0]
                        labels = [int(l.strip()) for l in items[1:]]
                        if len(cfg.DATA.SRC_DIR) > 0:
                            path =os.path.join(cfg.DATA.SRC_DIR, path)
                    except Exception as e:
                        print(line)
                        print(e)
                        continue
                    for out_idx,label in enumerate(labels):
                        if out_idx not in self.total_each_label:
                            self.total_each_label[out_idx] = {label:1}
                        elif label not in self.total_each_label[out_idx]:
                            self.total_each_label[out_idx][label] = 1
                        else:
                            self.total_each_label[out_idx][label] += 1
                    self.items.append((path, labels))
                
        self.train_transforms_preprocess = [
            augops.Resize(cfg,random_method=True),
        ]
        self.train_transforms = [
            augops.RandomBlur(cfg),
            augops.RandomContrast(cfg),
            augops.RandomRotation(cfg),
            augops.ShuffleColor(cfg),
            augops.JPEGCompression(cfg),
            augops.RandomLight(cfg),
            augops.RandomHFlip(cfg),
            augops.RandomVFlip(cfg),            
        ]
        self.train_transforms_postprocess = [
            augops.Crop(cfg,random_crop=True),
            augops.ToTensor(cfg.DATA.MEAN, cfg.DATA.STD)           
        ]

        self.val_transforms = [
            augops.Resize(cfg,random_method=False),
            augops.Crop(cfg,random_crop=False),
            augops.ToTensor(cfg.DATA.MEAN, cfg.DATA.STD)
        ]
        self.train_flag = train_flag


    def __len__(self):
        return len(self.items)

    def get_index_wrt_label(self,out_idx = 0):
        data = defaultdict(list)
        for index,paths_labels in enumerate(self.items):
            data[paths_labels[-1][out_idx]].append(index)
        return data

    # def class_size(self):
    #     data = []
    #     for k in range(self.class_num()):
    #         data.append(self.total_each_label[k])
    #     return data

    def input_channels(self):
        pass

    def output_num(self):
        return len(self.total_each_label.keys())
    
    def class_num(self, out_idx=0):
        return len(self.total_each_label[out_idx].keys())


    def __getitem__(self, item):
        return

   

class ClassificationDataset(BasicDataset):
    def __init__(self, **kwargs):
        super(ClassificationDataset,self).__init__(**kwargs)
    def __getitem__(self, item):
        image, labels = self.items[item]
        #image_data = cv2.imread(image,1)
        image_data = cv2.imdecode(np.fromfile(image,dtype=np.uint8),1) #support chinese
        
        if self.train_gray_image:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)

        if self.train_flag:
            for tran in self.train_transforms_preprocess:
                image_data, _ = tran.forward((image_data, labels))
                
            tran_indices = [k for k in range(len(self.train_transforms))]
            random.shuffle(tran_indices)
            for k in tran_indices:
                tran = self.train_transforms[k]
                image_data, _ = tran.forward((image_data,labels))

            for tran in self.train_transforms_postprocess:
                image_data, _ = tran.forward((image_data, labels))
        else:
            for tran in self.val_transforms:
                image_data, _ = tran.forward((image_data, labels))


        image_data = np.transpose(image_data,(2,0,1))


        return {
            'image': torch.from_numpy(image_data).type(torch.FloatTensor),
            'label': torch.from_numpy(np.asarray(labels)).type(torch.FloatTensor)
        }

# class DualClassificationDataset(BasicDataset):
#     def __init__(self, merge_mode, **kwargs):
#         super(DualClassificationDataset, self).__init__(**kwargs)
#         self.merge_mode = merge_mode
#     def __getitem__(self, item):
#         image, label = self.items[item]
#         image_data = cv2.imread(image, 1)
#         dual_image_data = cv2.imread(os.path.splitext(image)[0] + ".png",0)
#         if self.train_gray_image:
#             image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
#             image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
#         for augment in self.augments:
#             image_data, mask_data = augment.forward((image_data, label))

#         H, W, C = image_data.shape
#         dual_image_data = cv2.resize(dual_image_data, (W,H), cv2.INTER_NEAREST)

#         image_data = np.transpose(image_data, (2, 0, 1))
#         dual_image_data = np.expand_dims(dual_image_data,axis=0).astype(np.float32)

#         if self.merge_mode == "concat":
#             image_data = np.concatenate((image_data, dual_image_data//255.0),axis=0)
#         else:
#             image_data = image_data + dual_image_data / 255.0

#         return {
#             'image': torch.from_numpy(image_data).type(torch.FloatTensor),
#             'label': torch.from_numpy(np.asarray([label])).type(torch.FloatTensor)
#         }


class CLASSICAL_CLASSIFIER_BASELINE(ClassificationDataset):
    def __init__(self,**kwargs):
        super(CLASSICAL_CLASSIFIER_BASELINE, self).__init__(**kwargs)
        return
    def input_channels(self):
        return 3
 

if __name__=="__main__":
    from config.default_config import get_default_config
    #os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    cfg = get_default_config()
    #cfg.DATA.AUGMENT.VFLIP = True
    #cfg.DATA.AUGMENT.HFLIP = True
    #cfg.DATA.AUGMENT.BLUR = 19
    cfg.DTAT.TRAIN_GRAY_IMAGE = True
    cfg.DATA.AUGMENT.RANDOMBRIGHTNESS = 0.5
    cfg.DATA.AUGMENT.ROTATION = 10
    cfg.DATA.AUGMENT.GAMMA = 0.3
    cfg.DATA.RESIZE = [256,256]
    cfg.DATA.CROP = [224,224]
    ds = CLASSICAL_CLASSIFIER_BASELINE(list_file=r"K:\_GDISK\dty\bottom_border\robbin_break\stage\_data\smallimages\all_test.txt", train_flag=True, cfg=cfg)
    #ds = BasicDatasetLMDB(cfg,"G:/train_data/dty/fc/lmdb/test/",True)
    for k,one in enumerate(ds):
        print(one['image'].shape)
        if k > 10:
            break