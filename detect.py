# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve() # 获取当前脚本文件的绝对路径。
ROOT = FILE.parents[0]  # YOLOv5 root directory 通过获取父目录，确定YOLOv5的根目录。
if str(ROOT) not in sys.path: # 检查根目录是否已经在系统路径中。
    sys.path.append(str(ROOT))  # add ROOT to PATH 如果根目录不在系统路径中，则将根目录添加到系统路径中。
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative计算根目录相对于当前工作目录的相对路径。
from ultralytics.utils.plotting import Annotator, colors, save_one_box
# 从ultralytics.utils.plotting模块中导入了Annotator、colors和save_one_box等函数或类。这些函数可能用于绘制标注、处理颜色、保存检测框等可视化操作。
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.

    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    source = str(source) # 将source参数转换为字符串类型，以确保后续操作的一致性。
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    # 根据条件判断，确定是否保存推理图像。条件为不禁止保存（not nosave）且source不以".txt"结尾。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # ：判断source是文件还是URL。首先检查source的后缀是否在图片格式或视频格式中，以确定是否为文件。
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    # 判断source是否以特定协议开头，如"rtsp://", “rtmp://”, “http://”,
    # “https://”，以确定是否为URL。
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    # 判断source是否为数字（摄像头编号）、以".streams"结尾或是URL但不是文件。
    screenshot = source.lower().startswith("screen")
    # 判断是否为截图或屏幕截图：判断source是否以"screen"开头，以确定是否为屏幕截图。
    if is_url and is_file:
        source = check_file(source)  # download
        # 处理URL和文件的情况： 如果source同时是URL和文件，则调用check_file(source)函数进行下载处理。

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 保存目录路径：
    # save_dir是根据project和name参数构建的保存目录路径。如果exist_ok为True，则会递增命名以避免覆盖已存在的目录。
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 创建目录：
    # 根据条件判断，如果save_txt为True，则在save_dir下创建一个名为"labels"的子目录；否则直接在save_dir下创建目录。
    # 使用mkdir(parents=True, exist_ok=True)方法创建目录，确保父目录存在且避免因目录已存在而引发异常。
    device = select_device(device)
    # select_device(device)函数用于选择设备，根据device参数指定的值选择CUDA设备（GPU编号）或CPU设备。
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 初始化模型：使用DetectMultiBackend类初始化模型，传入模型权重路径weights、设备类型device、是否使用OpenCV
    # DNN进行推理dnn、数据集配置文件路径data以及是否使用FP16半精度推理half等参数。
    stride, names, pt = model.stride, model.names, model.pt
    # 获取模型信息： 从初始化的模型中获取模型的步长（stride）、类别名称列表（names）和模型的pt属性。
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 检查图像尺寸是否符合要求，根据模型的步长（stride）调整图像尺寸，以确保推理过程中输入图像的尺寸符合模型要求。
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # 调用model.warmup()方法对模型进行预热，传入图像尺寸参数。
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    # 初始化seen、windows和dt变量。seen用于记录已处理的数据量，windows用于存储窗口信息，dt是包含三个Profile对象的元组，用于记录时间性能信息。
    for path, im, im0s, vid_cap, s in dataset:
        # for循环中遍历数据集中的每个数据项，包括路径、图像数据、原始图像数据、视频捕获对象和其他信息。
        with dt[0]:
            im = torch.from_numpy(im).to(model.device) # 将图像数据转换为PyTorch张量(Tensor)，并移动到模型所在的设备上。
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 根据模型是否使用FP16半精度推理，将图像数据转换为半精度或全精度浮点数。 将像素值从0-255缩放到0.0-1.0之间。
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                # 如果图像数据维度为3维，则扩展一个维度以匹配模型的输入要求。
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
                # 如果模型需要XML格式输入并且图像数据批量大小大于1，则对图像数据进行分块处理。

        # Inference 模型推理过程中的可视化和预测结果处理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
# 在dt[2]时间性能记录块中，调用non_max_suppression函数对预测结果进行非极大值抑制处理。该函数会根据置信度阈值（conf_thres）、IoU阈值（iou_thres）、类别列表（classes）、是否使用类别不可知的NMS（agnostic_nms）以及最大检测数（max_det）等参数进行NMS操作，过滤掉重叠度高的边界框。
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"
# 定义了一个函数write_to_csv，用于将图像名称、预测结果和置信度写入CSV文件。如果CSV文件不存在，则会创建文件并写入表头；如果文件已存在，则会在文件末尾追加数据。
        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
                # 如果使用摄像头数据源（webcam=True），则将当前图像的路径（path[i]）、原始图像（im0s[i].copy()）和数据集的帧数计数（dataset.count）分别赋值给p、im0和frame变量，并更新字符串S。
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                # 如果不使用摄像头数据源，则将图像路径（path）、原始图像（im0s.copy()）和数据集的帧数计数（getattr(dataset, “frame”, 0)）分别赋值给p、im0和frame变量。
            p = Path(p)  # to Path
            # 对路径进行处理并生成保存路径和文本文件路径，输出图像尺寸信息，进行坐标归一化处理，以便后续保存图像文件、标签文件和处理边界框坐标等操作
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 程序遍历检测结果中的每个类别，统计每个类别的检测数量，并将类别名称和对应的检测数量添加到字符串s中，用于打印输出检测结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results 打印目标检测结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)
                        # 如果save_csv为True，则调用write_to_csv函数将图像名称、类别标签和置信度写入CSV文件
                    if save_txt:  # Write to file 如果save_txt为True，则将归一化后的边界框坐标、类别、置信度写入文本文件
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 绘制边界框和标签： 如果save_img、save_crop或view_img为True，则在图像上绘制边界框和标签。
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        # 如果save_crop为True，则将裁剪的边界框保存为单独的图像文件，文件名包括类别信息和图像名称
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            #  使用annotator.result()获取绘制了边界框和标签的图像im0。
            # 如果view_img为True，则将处理后的图像展示出来。
            if view_img: # 如果view_img为True，根据操作系统类型和窗口列表windows，判断是否需要创建新窗口并展示图像。
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections) 保存带有检测结果的图像或视频流，
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolo11.pt", help="model path or triton URL")
    # 指定网络模型以及下载
    parser.add_argument("--source", type=str, default=ROOT / "ultralytics-main/data/NEU-DET/train/images/bus.jpg", help="file/dir/URL/glob/screen/0(webcam)")
    # 指定检测的东西 添加 --source 参数，指定输入数据源，可以是文件、目录、URL、屏幕或摄像头（0 表示默认摄像头）。
    # 默认值为 ROOT / "data/images/bus.jpg"，即默认检测 bus.jpg 图像。
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # 添加 --data 参数，指定数据集配置文件的路径。
    # 默认值为 ROOT / "data/coco128.yaml"，即使用 coco128.yaml 作为数据集配置。
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    # 添加 --imgsz、--img 或 --img-size 参数，指定推理图像的尺寸（高度和宽度）。
    # 默认值为 [640]，即推理图像的尺寸是 640x640。
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    # 添加 --conf-thres 参数，设置置信度阈值。
    # 默认值为 0.25，即只有当检测框的置信度大于 25% 时，才会被认为是有效检测。
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    # 添加 --iou-thres 参数，设置 NMS（非极大值抑制）的IoU（Intersection over Union）阈值。
    # 默认值为 0.45，即IoU大于45%的框会被认为是重复的，从而抑制。
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    # 添加 --max-det 参数，设置每张图片最多进行的检测数量。默认值为 1000。
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 添加 --device 参数，设置使用的计算设备。
    # 可以是 CUDA 设备编号（如 0）或 cpu。默认使用所有可用设备。
    parser.add_argument("--view-img", action="store_true", help="show results")
    # 添加 --view-img 参数，如果该标志被设置，则显示推理结果图像。
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 添加 --save-txt 参数，如果设置该标志，则会将检测结果保存到 .txt 文件中。
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    # 添加 --save-format 参数，指定保存框坐标的格式。0 表示YOLO格式，1 表示Pascal-VOC格式。
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    # 添加 --save-csv 参数，如果设置该标志，则将结果保存为CSV格式。
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 添加 --save-conf 参数，如果设置该标志，则在 --save-txt 中保存检测框的置信度。
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    # 添加 --save-crop 参数，如果设置该标志，则保存裁剪后的预测框。添加 --save-crop 参数，如果设置该标志，则保存裁剪后的预测框。
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    # 添加 --nosave 参数，如果设置该标志，则不会保存检测图像或视频。
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    # 添加 --classes 参数，允许用户按类别过滤检测结果，用户可以指定要检测的类（如 --classes 0 或 --classes 0 2 3）。
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    # 添加 --agnostic-nms 参数，启用类无关的NMS，即不考虑类别之间的关系进行NMS。
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 添加 --augment 参数，启用增强推理，使推理过程包含数据增强。
    parser.add_argument('--visualize', default=True, action='store_true', help='visualize features') # 热力图修改
    # parser.add_argument("--visualize", action="store_true", help="visualize features")
    # 添加 --visualize 参数，启用特征可视化（如热力图）。默认为启用。
    parser.add_argument("--update", action="store_true", help="update all models")
    # 添加 --update 参数，如果设置该标志，程序会更新模型目录中的所有模型。
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    # 添加 --project 参数，设置结果保存的项目目录。默认保存到 ROOT / "runs/detect"。
    parser.add_argument("--name", default="exp", help="save results to project/name")
    # 添加 --name 参数，设置结果保存的子目录名称。默认保存到 exp 子目录。
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 可以帮结果保存到一个文件夹 添加 --exist-ok 参数，如果设置该标志，则允许覆盖已存在的项目和子目录。
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    # 添加--line-thickness参数，设置相似的线宽，单位为像素。默认值为3。
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    # 添加--hide-labels参数，设置是否隐藏标签。
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    # 添加--hide-conf参数，设置是否隐藏置信度。
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 添加--half参数，赋予FP16半精度推理作用以加速计算。
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 添加--dnn参数，实现OpenCV DNN进行ONNX模型推理。
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # 添加--vid-stride参数，设置视频帧率步进，表示每隔几帧进行一次推理。
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
    # 解析命令行输入的参数，可视化大小图像进行扩展（如果只有一个维度的话），然后打印解析后的参数，并返回结果。

def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
