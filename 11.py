# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("yolo11n.pt")
#
# # Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=10,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )
#
# # Evaluate model performance on the validation set
# metrics = model.val()
#
# # Perform object detection on an image
# results = model("E:/1chengxu/YOLO/11/ultralytics-main/data/NEU-DET/train/images/")
# results[0].show()
#
# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model


# from ultralytics import YOLO
#
# # 1. 加载模型
# model = YOLO("yolo11n.pt")  # 使用预训练模型或你自己的模型
#
# # 2. 训练模型
# train_results = model.train(
#     data="coco8.yaml",  # 数据集路径，coco8.yaml 配置文件应包含训练集、验证集路径等
#     epochs=10,  # 训练的轮数
#     imgsz=640,  # 图像大小
#     batch=16,  # 每批次的图像数，建议根据显存大小调整
#     device="cpu",  # 使用GPU时指定设备号，或使用"cpu"进行CPU训练
# )
#
# # 3. 评估模型性能（通常在验证集上评估）
# metrics = model.val()  # 获取评估结果，如mAP（mean Average Precision）
#
# # 4. 对整个验证集进行预测并展示结果
# # 这里假设 coco8.yaml 配置了验证集路径，模型训练过程中会自动对验证集进行评估
# val_results = model("E:/1chengxu/YOLO/11/ultralytics-main/data/NEU-DET")  # 提供验证集文件夹路径
# # 展示第一张图像的预测结果
# val_results[0].show()
#
# # 5. 导出模型为ONNX格式
# path = model.export(format="onnx")  # 导出ONNX格式模型
# print(f"模型已导出为ONNX格式，路径为：{path}")
# coding:utf-11
from ultralytics import YOLO
# 加载预训练模型
# model = YOLO("yolo11n.pt")
model = YOLO("yolo11-AKConv.yaml")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='coco128.yaml', epochs=4, batch=10 )  # 训练模型