import os, cv2
import numpy as np
import sys


#根据关键点裁块
def process_300w(root_folder, folder_name, image_name, label_name, target_size):
    image_path = os.path.join(root_folder, folder_name, image_name)       #图片绝对路径
    label_path = os.path.join(root_folder, folder_name, label_name)       #标签绝对路径

    with open(label_path, 'r') as ff:                                     #读pts文件
        anno = ff.readlines()[3:-1]                                       #读取关键点
        anno = [x.strip().split() for x in anno]                          #[[x1,y1],[x2,y2]....],转成关键点成对模式
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]       # 将关键点转成int
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]                                     #所有x的坐标
        anno_y = [x[1] for x in anno]                                     #所有y的坐标
        bbox_xmin = min(anno_x)
        bbox_ymin = min(anno_y)
        bbox_xmax = max(anno_x)
        bbox_ymax = max(anno_y)                                           #找到x,y 中的最大值和最小值，定位出检测目标物框的大小
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        scale = 1.1 
        bbox_xmin -= int((scale-1)/2*bbox_width)
        bbox_ymin -= int((scale-1)/2*bbox_height)                         #左上角的点进行微调，因为矩形框的宽和高对应放大1.1倍，x向左扩0.1*w/2
        bbox_width *= scale                                               #宽高放大1.1倍
        bbox_height *= scale
        bbox_width = int(bbox_width)
        bbox_height = int(bbox_height)
        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_width = min(bbox_width, image_width-bbox_xmin-1)              #越界判断
        bbox_height = min(bbox_height, image_height-bbox_ymin-1)
        anno = [[(x-bbox_xmin)/bbox_width, (y-bbox_ymin)/bbox_height] for x,y in anno]         #求出每个关键点相对于检测框的x,y的偏移量，再分别对宽和高进行归一化（等同于在小块图中的坐标，进行了归一化）

        bbox_xmax = bbox_xmin + bbox_width                                                     #右下角坐标
        bbox_ymax = bbox_ymin + bbox_height
        image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]                        #进行图片裁剪 【H,W,C】                         
        image_crop = cv2.resize(image_crop, (target_size, target_size))                        #图片进行resize
        return image_crop, anno





def process_dty_bottom(root_folder, folder_name, image_name, label_name, target_size,crop_size):
    image_path = os.path.join(root_folder, folder_name, image_name)      #图片绝对路径
    label_path = os.path.join(root_folder, folder_name, label_name)      #标签绝对路径

    with open(label_path, 'r') as ff:                                     #读pts文件
        anno = ff.readlines()[3:-1]                                       #读取关键点
        anno = [x.strip().split() for x in anno]                          #[[x1,y1],[x2,y2]....],转成关键点成对模式
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]       # 将关键点转成int
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]                                     #所有x的坐标
        anno_y = [x[1] for x in anno]                                     #所有y的坐标
        bbox_xmin = crop_size[0]
        bbox_ymin = crop_size[1]
        bbox_xmax = crop_size[2]
        bbox_ymax = crop_size[3]                                          #找到x,y 中的最大值和最小值，定位出检测目标物框的大小
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        scale = 1.1 
        bbox_xmin -= int((scale-1)/2*bbox_width)
        bbox_ymin -= int((scale-1)/2*bbox_height)                         #左上角的点进行微调，因为矩形框的宽和高对应放大1.1倍，x向左扩0.1*w/2
        bbox_width *= scale                                               #宽高放大1.1倍
        bbox_height *= scale
        bbox_width = int(bbox_width)
        bbox_height = int(bbox_height)
        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_width = min(bbox_width, image_width-bbox_xmin-1)              #越界判断
        bbox_height = min(bbox_height, image_height-bbox_ymin-1)
        anno = [[(x-bbox_xmin)/bbox_width, (y-bbox_ymin)/bbox_height] for x,y in anno]         #求出每个关键点相对于检测框的x,y的偏移量，再分别对宽和高进行归一化（等同于在小块图中的坐标，进行了归一化）

        bbox_xmax = bbox_xmin + bbox_width                                                     #右下角坐标
        bbox_ymax = bbox_ymin + bbox_height
        image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]                        #进行图片裁剪 【H,W,C】                         
        print('target_size[0]:',target_size[0], 'target_size[1]:',target_size[1] )
        image_crop = cv2.resize(image_crop, (target_size[0], target_size[1]))                  #图片进行resize   (W,H)
        return image_crop, anno




def process_dty_top(root_folder, folder_name, image_name, label_name, target_size, crop_size):
    image_path = os.path.join(root_folder, folder_name, image_name)      #图片绝对路径
    label_path = os.path.join(root_folder, folder_name, label_name)      #标签绝对路径

    with open(label_path, 'r') as ff:                                    #读pts文件
        anno = ff.readlines()[3:-1]                                      #读取关键点
        anno = [x.strip().split() for x in anno]                         #[[x1,y1],[x2,y2]....],转成关键点成对模式
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]      # 将关键点转成int
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]                                    #所有x的坐标
        anno_y = [x[1] for x in anno]                                    #所有y的坐标
        bbox_xmin = crop_size[0]   #300
        bbox_ymin = crop_size[1]   #150
        bbox_xmax = crop_size[2]   #2291
        bbox_ymax = crop_size[3]   #1847
        #找到x,y 中的最大值和最小值，定位出检测目标物框的大小
        bbox_width = bbox_xmax - bbox_xmin
        bbox_height = bbox_ymax - bbox_ymin
        scale = 1.1 
        bbox_xmin -= int((scale-1)/2*bbox_width)
        bbox_ymin -= int((scale-1)/2*bbox_height)                         #左上角的点进行微调，因为矩形框的宽和高对应放大1.1倍，x向左扩0.1*w/2
        bbox_width *= scale                                               #宽高放大1.1倍
        bbox_height *= scale
        bbox_width = int(bbox_width)
        bbox_height = int(bbox_height)
        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_width = min(bbox_width, image_width-bbox_xmin-1)              #越界判断
        bbox_height = min(bbox_height, image_height-bbox_ymin-1)
        anno = [[(x-bbox_xmin)/bbox_width, (y-bbox_ymin)/bbox_height] for x,y in anno]         #求出每个关键点相对于检测框的x,y的偏移量，再分别对宽和高进行归一化（等同于在小块图中的坐标，进行了归一化）

        bbox_xmax = bbox_xmin + bbox_width                                                     #右下角坐标
        bbox_ymax = bbox_ymin + bbox_height
      
        image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]                        #进行图片裁剪 【H,W,C】                         
        image_crop = cv2.resize(image_crop, (target_size[0], target_size[1]))                        #图片进行resize
        return image_crop, anno


def gen_meanface(root_folder, data_name):
    with open(os.path.join(root_folder, data_name, 'train.txt'), 'r') as f:
        annos = f.readlines()
    annos = [x.strip().split()[1:] for x in annos]              #取关键点的坐标，放在一个列表里，是字符类型
    annos = [[float(x) for x in anno] for anno in annos]        #将关键点重新放成列表，以float型
    annos = np.array(annos)        #【3148,136】                68个关键点，3148张图
    meanface = np.mean(annos, axis=0)   #【136】                对0每一行求均值，剩下的就是列，压缩行，对各列求均值
    meanface = meanface.tolist()
    meanface = [str(x) for x in meanface]                       #转成字符串的形式
    
    with open(os.path.join(root_folder, data_name, 'meanface.txt'), 'w') as f:
        f.write(' '.join(meanface))


def gen_data(root_folder, data_name, target_size, crop_size):
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))
    
    #########################################################################################################
    if data_name == 'KPDTYSidebottomborderbrightBobbinOuterSilkOuter':
        folders_train = ['train']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_train)))    #文件目录名：../data/nor
            image_files = [x for x in all_files if '.pts' not in x]                               #所有的文件名列表
            label_files = [x for x in all_files if '.pts' in x]                                   #所有的标签名列表与文件名列表一一对应
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                #print(image_name)
                #image_crop, anno = process_300w(os.path.join(root_folder, 'DTY_BOTTOM_BOARDER'), folder_train, image_name, label_name, target_size)    #一张图片处理,输出：裁剪resize好的图片，小块坐标归一化的x,y列表
                image_crop, anno = process_dty_bottom(os.path.join(root_folder, 'KPDTYSidebottomborderbrightBobbinOuterSilkOuter'), folder_train, image_name, label_name, target_size,crop_size)    #一张图片处理,输出：裁剪resize好的图片，小块坐标归一化的x,y列表
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)      #保存图片      
                annos_train[image_crop_name] = anno                                                                 #将列表保存成对应字典形式，后面的图片依旧以key-value的形式保存
        
        with open(os.path.join(root_folder, data_name, 'train.txt'), 'w') as f:          #将字典中的内容写入txt文件中并保存，写入txt中的是以字符的形式写入
            for image_crop_name, anno in annos_train.items():                            #遍历每一个字典
                #print('image_crop_name:',image_crop_name)
                #print('anno:',anno)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        

        folders_test = ['train']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                #print(image_name)
                image_crop, anno = process_dty_bottom(os.path.join(root_folder, 'KPDTYSidebottomborderbrightBobbinOuterSilkOuter'), folder_test, image_name, label_name, target_size,crop_size)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        
        gen_meanface(root_folder, data_name)          #只针对生成的train.txt
        print('KPDTYSidebottomborderbrightBobbinOuterSilkOuter is ok!')
        exit(0)
    #########################################################################################################
    if data_name == 'DTY_TOP':
        folders_train = ['nor']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_train)))    #文件目录名：../data/nor
            image_files = [x for x in all_files if '.pts' not in x]                               #所有的文件名列表
            label_files = [x for x in all_files if '.pts' in x]                                   #所有的标签名列表与文件名列表一一对应
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                #print(image_name)
                #image_crop, anno = process_300w(os.path.join(root_folder, 'DTY_TOP'), folder_train, image_name, label_name, target_size)    #一张图片处理,输出：裁剪resize好的图片，小块坐标归一化的x,y列表
                image_crop, anno = process_dty_top(os.path.join(root_folder, 'DTY_TOP'), folder_train, image_name, label_name, target_size)    #一张图片处理,输出：裁剪resize好的图片，小块坐标归一化的x,y列表
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)      #保存图片      
                annos_train[image_crop_name] = anno                                                                 #将列表保存成对应字典形式，后面的图片依旧以key-value的形式保存
        
        with open(os.path.join(root_folder, data_name, 'train.txt'), 'w') as f:          #将字典中的内容写入txt文件中并保存，写入txt中的是以字符的形式写入
            for image_crop_name, anno in annos_train.items():                            #遍历每一个字典
                #print('image_crop_name:',image_crop_name)
                #print('anno:',anno)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        

        folders_test = ['siyang']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                #print(image_name)
                image_crop, anno = process_dty_top(os.path.join(root_folder, 'DTY_TOP'), folder_test, image_name, label_name, target_size)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        gen_meanface(root_folder, data_name)          #只针对生成的train.txt
        print('DTY_TOP is ok!')        
        exit(0)
    ################################################################################################################
    if data_name == 'data_300W':
        folders_train = ['afw', 'helen/trainset', 'lfpw/trainset']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_train)))    #文件目录名：../data/afw
            image_files = [x for x in all_files if '.pts' not in x]                               #所有的文件名列表
            label_files = [x for x in all_files if '.pts' in x]                                   #所有的标签名列表与文件名列表一一对应
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                #print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_train, image_name, label_name, target_size)    #一张图片处理,输出：裁剪resize好的图片，小块坐标归一化的x,y列表
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)      #保存图片      
                annos_train[image_crop_name] = anno                                                                 #将列表保存成对应字典形式，后面的图片依旧以key-value的形式保存
        
        with open(os.path.join(root_folder, data_name, 'train.txt'), 'w') as f:          #将字典中的内容写入txt文件中并保存，写入txt中的是以字符的形式写入
            for image_crop_name, anno in annos_train.items():                            #遍历每一个字典
                #print('image_crop_name:',image_crop_name)
                #print('anno:',anno)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        

        folders_test = ['helen/testset', 'lfpw/testset', 'ibug']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                #print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_test, image_name, label_name, target_size)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
        
        annos = None
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'r') as f:
            annos = f.readlines()
        with open(os.path.join(root_folder, data_name, 'test_common.txt'), 'w') as f:           #测试集样本不加ibug
            for anno in annos:
                if not 'ibug' in anno:
                    f.write(anno)
        with open(os.path.join(root_folder, data_name, 'test_challenge.txt'), 'w') as f:        #保存测试样本中的ibug
            for anno in annos:
                if 'ibug' in anno:
                    f.write(anno)

        gen_meanface(root_folder, data_name)          #只针对生成的train.txt 
        print('data_300W is ok!')
        exit(0)
    else:
        print('Wrong data!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input the data name.')
        print('1. data_300W')
        print('2. DTY_TOP')
        print('3. KPDTYSidebottomborderbrightBobbinOuterSilkOuter')
        exit(0)
    else:
        data_name = sys.argv[1]
        root_dir = os.path.join(os.path.dirname(__file__),"..","data")
        gen_data(root_dir, data_name, [256,256], [39,399,2551,2039])    #[W,H]   [x_min,y_min,x_max,y_max]   [39,799,2551,1947]


