#*******关键点pts转json格式**********
#***********duanting**************
#***********20220505**************
import cv2
import argparse
import numpy as np
import json
import os
from pts2json_config import get_default_config
import pdb 

def _box2cs(box,aspect_ratio):
    x, y, w, h = box[:]                             #拿到框的坐标
    return _xywh2cs(x, y, w, h,aspect_ratio)


def _xywh2cs( x, y, w, h,aspect_ratio):
    center = np.zeros((2), )
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5                        #拿到中心点的坐标
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],  # 宽高都除以200
       )
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def pts2json(image_name,pts_name,image,cfg):
    # image_name--输入图片名字
    # pts_name---输入图片的关键点标注
    # json_dict--返回一张图片的dict <'class',dict >
    # 将pts转成json文件思路:
        # step1:拿到图片名 image
        # step2:构造检测框的坐标--->据框的坐标转成center与scale
        # step3:将关键点转成joints
        # step4:根据关键点构造joint_vis
        # step5:生成json_dict,需要将array转成list


    # 解析pts
    name_crop = image_name.split('/')
    file_name = name_crop[-1]
    with open(pts_name) as file_obj:
        contents = file_obj.readlines()

    landmarks = []
    for i, line in enumerate(contents):
        TT = line.strip("\n")
        if i > 2 and i < (cfg.INFO.NUM_POINTS+3):
            # print TT
            if i % 1 == 0:
                TT_temp = TT.split(" ")
                x = float(TT_temp[0])
                y = float(TT_temp[1].strip("\r"))
                landmarks.append((x, y))

    img_w = image.shape[1]                                # HWC
    img_h = image.shape[0]
    x1 = cfg.INFO.LEFT_RANGE                              # bottom: 39    1                                 top:300    1
    y1 = cfg.INFO.TOP_RANTGE                             # bottom:799    1                                 top:150    1
    w = (img_w - cfg.INFO.RIGHT_RANGE - 1)- x1            # bottom:(img_w - 40 - 1)- x1    img_w-2          top:(img_w - 300 - 1)- x1    img_w-2
    h = (img_h - cfg.INFO.BOTTOM_RANGE - 1)- y1           # bottom:(img_h - 300 - 1)-y1    img_h-2          top:(img_h - 200 - 1)- y1    img_h-2

    box = [x1, y1, w, h]
    print('box is ',box)
    print('w:h is ', box[2]/box[3])
    center, scale = _box2cs(box,cfg.INFO.RATION)           #根据框的坐标，得到中心点坐标和scale缩放比   #需要根据长宽比进行调整        3,1   crop:2.65

    joints_3d = np.zeros((len(landmarks), 2),dtype=np.float32)
    joints_3d_vis = np.ones((len(landmarks)),dtype=np.float32)

    for idx, keypoint in enumerate(landmarks):
        joints_3d[idx, 0] = keypoint[0]  # 原关键点中的0，1是关键点的坐标，第三位是VIS（0，1，2）
        joints_3d[idx, 1] = keypoint[1]

        joints_3d_vis[idx] = 1

    json_dict = {"joints_vis": joints_3d_vis.tolist(), "joints": joints_3d.tolist(), "image": file_name,
                 "scale": list(scale), "center": list(center)}
    return json_dict,landmarks,center,box


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_file",type=str,help="yaml configuration file")
    ap.add_argument("--src_root",type=str, default='',help="yaml configuration file")
    ap.add_argument("--dst_root",type=str, default='',help="yaml configuration file")
    args = ap.parse_args()

    cfg = get_default_config()
    #print(args.yaml_file)

    cfg.merge_from_file(args.yaml_file)
    cfg.freeze()

    src_root = cfg.INFO.SRC_ROOT
    dst_root=cfg.INFO.DST_ROOT
    dst_json_name=cfg.INFO.JSON_NAME

    # 读取文件夹，遍历文件夹底下的图片和pts
    json_all = []
    for root, dirs, files in os.walk(src_root):      #dir--root底下的文件夹列表格式，files--文夹名
        #print('files:', files)    #获取文件所属目录
        for id,file in enumerate(files):
            #获取文件路径
            #pdb.set_trace()
            name_crop = file.split('.')
            #print('name_crop:',name_crop)

            com_name= name_crop[0]
            last_name=name_crop[1]
            mark_name=''
            if last_name != 'pts':
                if last_name =='jpg':
                    image_name = os.path.join(root, com_name+'.jpg')
                    mark_name= com_name+'.jpg'
                elif last_name =='png':
                    image_name = os.path.join(root, com_name + '.png')
                    mark_name = com_name + '.png'
                elif last_name =='bmp':
                    image_name = os.path.join(root, com_name + '.bmp')
                    mark_name = com_name + '.bmp'
                else:
                    image_name=os.path.join(root, com_name + '.jpeg')
                    mark_name = com_name + '.jpeg'

                pts_name = os.path.join(root,com_name+'.pts')
                # print('pts_name:',pts_name)

                image = cv2.imread(image_name, 1)  # 1 彩色，0 灰色
                if image is None:
                    print('*************图片是空图*************')
                    continue
                print('image_name',image_name)
                json_dict, landmarks, center, box = pts2json(file, pts_name, image,cfg)
                json_str = json.dumps(json_dict)
                json_all.append(json_dict)
                # print('json_str:',json_str)
                # print('json_all:',json_all)
                print('pic is ',str(id))
                if cfg.INFO.VIS_FLAG:
                    #  将关键点标在图片上
                    '''
                    cv2.circle(image, center_coordinates, radius, color, thickness)
                    '''
                    m = 0  # 标号初始为0
                    for point in landmarks:
                        # print(point[0],point[1])
                        cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0),
                                   -1)  # 颜色顺序：BGR (0, 255, 0)绿色,-1 实心圆
                        m += 1
                        cv2.putText(image, str(m), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    1)  # 每个关键点上标号
                        # plt.scatter(np.transpose(point)[0], np.transpose(point)[1])  # 散点图
                        # plt.show()

                    cv2.circle(image, (int(center[0]), int(center[1])), 2, (0, 255, 255),-1)  # 颜色顺序：BGR (0, 255, 0)绿色,-1 实心圆
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),(0, 255, 255), 2)  # 坐标点为整数
                    iamge = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))

                    if not os.path.exists( dst_root +'/mark' ):  
                        os.makedirs(dst_root +'/mark' )
                    cv2.imwrite(dst_root +'/mark/' + mark_name,iamge)
                    # cv2.imshow("pointImg", iamge)
                    # cv2.waitKey()

    #保存json文件图片
    json_file = os.path.join(dst_root ,dst_json_name)
    json_fp = open(json_file, 'w')

    # print('json_file:',dst_root)
    # 将字典形式转换成字符串形式
    json_str = json.dumps(json_all, indent=4)

    # print('json_str:', json_str)
    json_fp.write(json_str)
    json_fp.close()


