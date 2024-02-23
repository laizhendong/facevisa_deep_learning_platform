import os
from unicodedata import name
import xml.etree.ElementTree as ET
import glob
 
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                L.append(os.path.join(root, file))
        return L 
 
def count_num(indir):
    # 提取xml文件列表
    os.chdir(indir)
    #annotations = os.listdir('.')    
    #annotations = glob.glob(str(annotations) + '*.xml')    
    
    annotations = file_name(indir)
    dict = {}  # 新建字典，用于存放各类标签名及其对应的数目
    #for i, file in enumerate(annotations):  # 遍历xml文件
    for i, file in enumerate(annotations):  # 遍历xml文件
 
        # actual parsing
        in_file = open(file, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
 
        # 遍历文件的所有标签
        for obj in root.iter('object'):
            name = obj.find('name').text
            if (name in dict.keys()):
                dict[name] += 1  # 如果标签不是第一次出现，则+1
            else:
                dict[name] = 1  # 如果标签是第一次出现，则将该标签名对应的value初始化为1
 
    KeyDict = sorted(dict)
   
    #创建data.yaml文件
    f1 = open("/code/yolov5/test_data.yaml","w")    
    f1.writelines("test: /test_datasets/yolo_data/stain/images/test/\n")    
    f1.writelines("dataset: ['test']\n")
    f1.writelines("nc: ")
    f1.writelines(str(len(KeyDict))+'\n')
    f1.writelines("names: ")    
    f1.writelines(str(KeyDict))
    f1.close()
 
    # 打印结果
    print("%d kind labels and %d labels in total" % (len(KeyDict), sum(dict.values())))
    print('labels:',KeyDict)
    print('\n')
    print("Label Name and it's number//各类标签的数量分别为:")
    for key in dict.keys():
        print(key + ': ' + str(dict[key]))
    print("\n总标签数目:{}个".format(sum(dict.values())))
    print('\t')
    print('检索完成！')
 
 
indir = '/test_datasets/voc_data/stain/imgxml'  # xml文件所在的目录
 
count_num(indir)  #统计各类标签数目
 