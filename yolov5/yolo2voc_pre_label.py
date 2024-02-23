import os
import glob
from PIL import Image
import yaml

voc_annotations = '/output'
yolo_txt = '/inference/output'
img_path = '/data'
 
curPath = os.path.dirname(os.path.realpath(__file__))
# 获取yaml文件路径
yamlPath = os.path.join(curPath, "data.yaml")
 
# open方法打开直接读出来
f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
# 用load方法转字典
label_dict = yaml.load(cfg)  

sets = label_dict['dataset']
#labels = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ] # label for datasets
labels = label_dict['names']
 
# 图像存储位置
src_img_dir = img_path 
# 图像的txt文件存放位置
 
 
src_txt_dir = yolo_txt
src_xml_dir = voc_annotations
 
imgjpg_Lists = glob.glob(src_img_dir + '/*.jpg')
imgpng_Lists = glob.glob(src_img_dir + '/*.png')
imgbmp_Lists = glob.glob(src_img_dir + '/*.bmp')

txt_Lists = glob.glob(src_txt_dir + '/*.txt')
 
img_basenames = []
for item in imgjpg_Lists:
    img_basenames.append(os.path.basename(item))

for item in imgpng_Lists:
    img_basenames.append(os.path.basename(item))

for item in imgbmp_Lists:
    img_basenames.append(os.path.basename(item))
 
img_names = []
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
 
for img in img_basenames:
    im = Image.open((src_img_dir + '/' + img))
    width, height = im.size
    img_name = os.path.splitext(img)[0]   
    # 打开txt文件
    isExists=os.path.exists(src_txt_dir + '/' + img_name + '.txt')
    if isExists:    
        gt = open(src_txt_dir + '/' + img_name + '.txt').read().splitlines()
        print(gt)
        if gt:
            # 将主干部分写入xml文件中
            xml_file = open((src_xml_dir + '/' + img_name + '.xml'), 'w')
            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + str(img) + '</filename>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(width) + '</width>\n')
            xml_file.write('        <height>' + str(height) + '</height>\n')
            xml_file.write('        <depth>3</depth>\n')
            xml_file.write('    </size>\n')
     
            # write the region of image on xml file
            for img_each_label in gt:
                spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
                print(f'spt:{spt}')
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + str(labels[int(spt[0])]) + '</name>\n')
                #add score
                xml_file.write('        <score>' + str(float(spt[1])) + '</score>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
     
                center_x = round(float(spt[2].strip()) * width)
                center_y = round(float(spt[3].strip()) * height)
                bbox_width = round(float(spt[4].strip()) * width)
                bbox_height = round(float(spt[5].strip()) * height)
                xmin = str(int(center_x - bbox_width / 2))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))
     
                xml_file.write('            <xmin>' + xmin + '</xmin>\n')
                xml_file.write('            <ymin>' + ymin + '</ymin>\n')
                xml_file.write('            <xmax>' + xmax + '</xmax>\n')
                xml_file.write('            <ymax>' + ymax + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('        <shape type="rect" color="red" thickness="3">\n')
                xml_file.write('            <points>\n')
                xml_file.write('                <x>' + xmin + '</x>\n')
                xml_file.write('                <y>' + ymax + '</y>\n')
                xml_file.write('                <x>' + xmin + '</x>\n')
                xml_file.write('                <y>' + ymin + '</y>\n')
                xml_file.write('                <x>' + xmax + '</x>\n')
                xml_file.write('                <y>' + ymin + '</y>\n')
                xml_file.write('                <x>' + xmax + '</x>\n')
                xml_file.write('                <y>' + ymax + '</y>\n')
                xml_file.write('            </points>\n')
                xml_file.write('        </shape>\n')
                xml_file.write('    </object>\n')
     
            xml_file.write('</annotation>')