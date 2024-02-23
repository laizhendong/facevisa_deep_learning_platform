import os  #通过os模块调用系统命令

file_path = "/data/train"  #文件路径
path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表

path_name = []#定义一个空列表

for i in path_list:
    if os.path.splitext(i)[1] == '.jpg' or os.path.splitext(i)[1] == '.png' or os.path.splitext(i)[1] == '.bmp':
        #path_name.append('Data/DTY/YW-Top/CLS/DTY_xx/cascade/data/youguang/20220509/train/1/' + i + ' 1') #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名
        path_name.append(i)
        #path_name.sort() #排序+

for i in range(0,len(path_name)):
#for file_name in path_name:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open("/datasets/voc_data/stain/ImageSets/train.txt", "a") as file:
        file_name = str(path_name[i])[0:-4]
        file.write(file_name + "\n")
        print(file_name)
    file.close()   
