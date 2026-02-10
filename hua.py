import os
import random
import shutil

# 设置数据集路径
dataset_dir = 'YOLODataset'  # 需要替换成你的数据集路径
images_dir = os.path.join(dataset_dir, 'images')  # 图像文件夹
labels_dir = os.path.join(dataset_dir, 'labels')  # 标注文件夹

# 设置训练、验证和测试集的划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 创建保存训练、验证、测试集的文件夹
train_images_dir = os.path.join(dataset_dir, 'train/images')
val_images_dir = os.path.join(dataset_dir, 'val/images')
test_images_dir = os.path.join(dataset_dir, 'test/images')
train_labels_dir = os.path.join(dataset_dir, 'train/labels')
val_labels_dir = os.path.join(dataset_dir, 'val/labels')
test_labels_dir = os.path.join(dataset_dir, 'test/labels')

# 创建文件夹
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 随机打乱图像文件
random.shuffle(image_files)

# 划分数据集
train_count = int(len(image_files) * train_ratio)
val_count = int(len(image_files) * val_ratio)

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# 移动图像和标签文件
def move_files(files, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
    for file in files:
        # 移动图像文件
        image_path = os.path.join(src_images_dir, file)
        shutil.copy(image_path, dst_images_dir)

        # 移动标签文件
        label_path = os.path.join(src_labels_dir, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        shutil.copy(label_path, dst_labels_dir)

move_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
move_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)
move_files(test_files, images_dir, labels_dir, test_images_dir, test_labels_dir)

# 生成 train.txt 和 val.txt 配置文件
def generate_file_list(file_list, output_file):
    with open(output_file, 'w') as f:
        for file in file_list:
            f.write(f"{os.path.join(dataset_dir, 'images', file)}\n")

generate_file_list(train_files, os.path.join(dataset_dir, 'train.txt'))
generate_file_list(val_files, os.path.join(dataset_dir, 'val.txt'))
generate_file_list(test_files, os.path.join(dataset_dir, 'test.txt'))

print("数据集划分完成！")
