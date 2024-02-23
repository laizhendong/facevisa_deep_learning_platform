# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import numbers
import torchvision.transforms.functional as TF
import pdb


#亮度、对比度、饱和度数据增强
class Compose(object):
   def __init__(self, transforms):
      self.transforms = transforms

   def __call__(self, sample):
      for t in self.transforms:
         sample = t(sample)
      return sample

class Lambda(object):
   """Apply a user-defined lambda as a transform.

   Args:
       lambd (function): Lambda/function to be used for transform.
   """

   def __init__(self, lambd):
      assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
      self.lambd = lambd

   def __call__(self, img):
      return self.lambd(img)


class ColorJitter(object):
   """Randomly change the brightness, contrast and saturation of an image.

   Args:
       brightness (float or tuple of float (min, max)): How much to jitter brightness.
           brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
           or the given [min, max]. Should be non negative numbers.
       contrast (float or tuple of float (min, max)): How much to jitter contrast.
           contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
           or the given [min, max]. Should be non negative numbers.
       saturation (float or tuple of float (min, max)): How much to jitter saturation.
           saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
           or the given [min, max]. Should be non negative numbers.
       hue (float or tuple of float (min, max)): How much to jitter hue.
           hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
           Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
   """

   def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
      self.brightness = self._check_input(brightness, 'brightness')
      self.contrast = self._check_input(contrast, 'contrast')
      self.saturation = self._check_input(saturation, 'saturation')
      self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                   clip_first_on_zero=False)

   def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
      if isinstance(value, numbers.Number):
         if value < 0:
            raise ValueError("If {} is a single number, it must be non negative.".format(name))
         value = [center - value, center + value]
         if clip_first_on_zero:
            value[0] = max(value[0], 0)
      elif isinstance(value, (tuple, list)) and len(value) == 2:
         if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name, bound))
      else:
         raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

      # if value is 0 or (1., 1.) for brightness/contrast/saturation
      # or (0., 0.) for hue, do nothing
      if value[0] == value[1] == center:
         value = None
      return value

   @staticmethod
   def get_params(brightness, contrast, saturation, hue):

      transforms = []

      if brightness is not None:
         brightness_factor = random.uniform(brightness[0], brightness[1])
         transforms.append(Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))

      if contrast is not None:
         contrast_factor = random.uniform(contrast[0], contrast[1])
         transforms.append(Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))

      if saturation is not None:
         saturation_factor = random.uniform(saturation[0], saturation[1])
         transforms.append(Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))

      if hue is not None:
         hue_factor = random.uniform(hue[0], hue[1])
         transforms.append(Lambda(lambda img: TF.adjust_hue(img, hue_factor)))

      random.shuffle(transforms)
      transform = Compose(transforms)

      return transform

   def __call__(self, x):
      transform = self.get_params(self.brightness, self.contrast,
                                  self.saturation, self.hue)
      img = transform(x)
      return img






def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    #关于对称的关键点，左右坐标进行交换
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):

    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    #遍历每一个关键点,将关键点返回去
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])


    scale_tmp = scale * 200.0    #1.25box_w
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)                             #对点进行旋转【0，-0.5*src_w 】 角度为0等同于不做旋转
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)                                  #3个点
    dst = np.zeros((3, 2), dtype=np.float32)
    
    src[0, :] = center + scale_tmp * shift                                    #大图（猜测是框，目标物）的中心点
    src[1, :] = center + src_dir + scale_tmp * shift                          #x不变，y 往上减了，0.5*src_w 
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]                                    #小图的中心点
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir                #小图的y也往上减了0.5*小图的宽

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])                          #计算出2个点的高相差0.5src_w，第2个点y轴不变，x减0.5src_w
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))       #2个3*2的坐标得到，一个2*3的仿射变换矩阵
    
    return trans


def affine_transform(pt, t):
    #关键点坐标进行仿射变换
    new_pt = np.array([pt[0], pt[1], 1.]).T                                    #该转置没起作用
    new_pt = np.dot(t, new_pt)
    #new_pt = np.array([[pt[0], pt[1], 1.]]).T                                 
    #new_pt = np.dot(t, new_pt).T                                              #内积
    return new_pt[:2]


def get_3rd_point(a, b):
    #根据中心点和根据中心点偏移的点进行第三个点的构造
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    #根据旋转角度，进行点的旋转
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img
