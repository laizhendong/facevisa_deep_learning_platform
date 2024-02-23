import json
import cv2
import os

# 读取目标框的 JSON 标签文件
def read_labels(json_file):
    with open(json_file, 'r') as file:
        labels = json.load(file)
    return labels

# 对一张图像进行裁剪
def crop_image(image, bbox):
    x = int(bbox['top'])
    y = int(bbox['left'])
    width = int(bbox['width'])
    height = int(bbox['height'])
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image

# 对一批图像进行批量裁剪
def batch_crop_images(image_dir, label_dir, output_dir):
    # 获取图像文件列表
    image_files = os.listdir(image_dir)

    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print('无法读取图像文件:', image_path)
            continue

        # 构建标签文件路径
        json_file = os.path.join(label_dir, image_file + '.json')
        if not os.path.exists(json_file):
            print('找不到对应的标签文件:', json_file)
            continue

        # 读取标签文件
        labels = read_labels(json_file)

        # 对每个目标框进行裁剪并保存
        #for label_data in labels:
        for label_data in range(len(labels['instances'])):
            cropped_image = crop_image(image, label_data)

            # 构建输出文件路径
            output_file = os.path.join(output_dir, label_data['id'] + '_' + image_file)
            cv2.imwrite(output_file, cropped_image)

            print('已保存裁剪图像:', output_file)

# 设置输入和输出目录路径
image_dir = 'C:/Users/22746/Desktop/test/imgjson'
label_dir = 'C:/Users/22746/Desktop/test/imgjson'
output_dir = 'C:/Users/22746/Desktop/test/crop_img'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 对一批图像进行批量裁剪
batch_crop_images(image_dir, label_dir, output_dir)