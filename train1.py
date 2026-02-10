# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import os
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = r'E:\app\1\git\git\Git\bin\git.exe'  # 替换为你的 Git 路径


try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    # hyp: 超参数文件路径或字典，用于设置训练的超参数。
    # opt：命令行解析的训练选项，包含训练的基本配置信息。
    # device: 指定训练的设备，通常为'cuda'或'cpu'。
    # callbacks：回调函数，用于在训练中的不同阶段触发事件。
    """
    Train a YOLOv5 model on a custom dataset using specified hyperparameters, options, and device, managing datasets,
    model architecture, loss computation, and optimizer steps.

    Args:
        hyp (str | dict): Path to the hyperparameters YAML file or a dictionary of hyperparameters.
        opt (argparse.Namespace): Parsed command-line arguments containing training options.
        device (torch.device): Device on which training occurs, e.g., 'cuda' or 'cpu'.
        callbacks (Callbacks): Callback functions for various training events.

    Returns:
        None

    Models and datasets download automatically from the latest YOLOv5 release.

    Example:
        Single-GPU training:
        ```bash
        $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
        $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
        ```

        Multi-GPU DDP training:
        ```bash
        $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights
        yolov5s.pt --img 640 --device 0,1,2,3
        ```

        For more usage details, refer to:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),# 保存模型权重及训练日志的路径。
        opt.epochs,# 训练轮数。
        opt.batch_size,# 批量大小。
        opt.weights,# 预训练权重路径。
        opt.single_cls,# 指定是否为单类别训练。
        opt.evolve,# 指定是否有超参数。
        opt.data,# 数据集配置文件路径。
        opt.cfg,# 模型配置文件路径。
        opt.resume,# 是否从断点继续训练。
        opt.noval,#  是否跳过验证。
        opt.nosave,# 是否不保存模型。
        opt.workers,# 工作进程数量。
        opt.freeze, # 冻结的层数量。
    )
    callbacks.run("on_pretrain_routine_start")
    # 在预训练实例过程中开始时触发回调事件，用于在训练前执行初始化操作。

    # Directories
    w = save_dir / "weights"  # weights dir 在保存路径下创建权限重目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # 若evolve为真，则在weights的父目录创建目录（用于此时），直接否则在weights目录创建。
    last, best = w / "last.pt", w / "best.pt" # 定义存储最新和最佳模型权重的路径。

    # Hyperparameters
    if isinstance(hyp, str):# 如果hyp是字符串类型，表示其为YAML文件路径。
        with open(hyp, errors="ignore") as f: # 读取YAML文件，将内容加载到字典中。
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    # 记录加载的超参数，然后进行调试和检查。

    # Save run settings
    if not evolve:# 如果没有指定超参数，则保存当前的超参数和选项配置文件。
        yaml_save(save_dir / "hyp.yaml", hyp) # 将超参数hyp保存到hyp.yaml文件。
        yaml_save(save_dir / "opt.yaml", vars(opt)) # 将opt转换为字典并保存到opt.yaml文件。

    # Loggers
    data_dict = None # 初始化数据字典为None，用于存储日志器中的数据。
    if RANK in {-1, 0}: # 只在主进程或单节点上初始化日志器。
        include_loggers = list(LOGGERS) #初始化日志器列表。
        if getattr(opt, "ndjson_console", False): # 如果在选项中启用了ndjson_console，则添加到日志器列表。
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False): #如果启用了ndjson_file，则也添加到日志器列表。
            include_loggers.append("ndjson_file")

        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        ) #创建Loggers实例，用于记录和管理训练期间的日志信息。

        # Register actions
        for k in methods(loggers): # 遍历loggers其中的方法。
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset # 从日志器中获取自定义数据集的远程路径。
        if resume:  # If resuming runs from remote artifact
            # 如果指定了断点恢复，则从选项中重新读取训练的基本配置
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    # 当evolve为False且`选择opt.noplots未设置时，plots对于True，用于是否生成训练过程的可视化图表。
    cuda = device.type != "cpu"# 根据device判断是否使用CUDA（GPU设备）。
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 初始化随机种子，以保证训练的可恢复性，并为循环训练中的每个节点设置不同的种子。
    with torch_distributed_zero_first(LOCAL_RANK):
        # 使用torch_distributed_zero_first上下文确保在循环训练时，只有第一个节点会首先检查并下载数据集。
        data_dict = data_dict or check_dataset(data)  # check if None
        #如果data_dict为None，则通过check_dataset函数检查并加载数据集。
    train_path, val_path = data_dict["train"], data_dict["val"]
    # 从data_dict获取训练和验证数据集的路径。
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    # 设置类别数量，如果single_cls为True则类别数为1，否则data_dict则获取类别数量。
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    # 如果是单类别且数据集中类名列表长度不为1，则将类名设置为{0: "item"}，否则使用data_dict中的类名列表。
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset
    # 检查验证数据集路径是否指向COCO数据集的验证集。
    # Model
    check_suffix(weights, ".pt")  # check weights
    # 检查权重文件是否以.pt结尾，确保是模型的权重文件。
    pretrained = weights.endswith(".pt")
    # 如果权重文件是.pt格式，则表示使用预训练模型
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        # 若pretrained为True，且权重文件在本地不存在，则下载该权重文件。
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        # 将预训练权重加载到CPU，避免CUDA内存溢出。
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 根据配置文件或检查点中的模型配置创建YOLO模型，指定输入通道数ch=3（RGB），类别数nc和节点配置anchors，然后将模型加载到指定设备上。
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        # 如果使用自定义配置或定义了某个点且不能继续训练，则将anchor要排除的键。
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # 获取预训练权重的状态字典，并在csd和新模型的状态字典之间取交集，只保留公共键，排除exclude中的键。
        model.load_state_dict(csd, strict=False)  # load
        # 将csd中的参数加载到模型中，strict=False表示不要求完全匹配。
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
        # 输出加载参数的数量，报告成功加载的权重项数量。
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 如果未加载预训练模型，则直接创建模型。
    amp = check_amp(model)  # check AMP
    # 检查是否支持自动混合精度（AMP），用于提升模型训练效率。
    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 需要生成上面的层名称列表，freeze列表中的每个元素都会被初始化为model.<层号>.。
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers 确保所有层都可训练。
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False
            # 若参数名称在freeze列表中，将其requires_grad设置为False冻结该层，避免更新权限重。

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 计算模型的最大步长，并保证步长至少为32，则适合模型的网格大小。
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # 使用check_img_size函数确保输入图像尺寸imgsz是步长gs的倍数。
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        # 如果是单个 GPU 并且批量大小未指定，则使用check_train_batch_size函数给出最佳批量大小。
        loggers.on_params_update({"batch_size": batch_size})
        # 更新日志器参数，用于记录批量大小

    # Optimizer 优化器设置
    nbs = 64  # nominal batch size 定义标称批量大小为64，用于调整权重衰减和累积步长。
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 计算累积步数，将损失累积到指定步数后再更新梯度，以适应小批量训练的设置。
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    # 根据实际批量大小和累计步数调整权重衰减，在不同批量大小下保持一致。
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    # 使用smart_optimizer函数创建优化器，根据超参数中的学习率 ( lr0)、动量 ( momentum) 和重衰减 ( weight_decay) 进行初始化。
    # Scheduler 学习率调度器
    # 根据opt.cos_lr学习率决定的调度方式。
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
        # 如果为True，使用余弦偏置调度器 ( one_cycle)，将学习率从 1 缩放到hyp['lrf']。
    else:

        def lf(x):
            """Linear learning rate scheduler function with decay calculated by epoch proportion."""
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
        # 如果为False，则定义线性调度器 ( lf) 函数，学习率随训练进程逐步衰减。
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    # 使用LambdaLR将定义的学习率调度器lf应用到优化器中，控制每个epoch的学习率变化。
    # EMA 模型值滑动
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    # 为模型创建指数滑动平均 (EMA) 实例，以平滑模型参数并提升推理稳定性。仅在单 GPU 或主进程中执行。
    # Resume 断点恢复
    best_fitness, start_epoch = 0.0, 0
    # 初始化最佳适应度 ( best_fitness) 和起始纪元，为断点恢复做准备。
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd
        # 如果加载预训练模型并需要断点恢复，使用smart_resume函数加载断点，恢复最佳适应度、初始历元和总历元数。另外删除检查点和状态字典，释放内存。
    # DP mode 多GPU模式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)
        # 当CUDA可用且在单GPU模式（RANK == -1）下检测到多GPU时，使用数据任务模式（DataParallel）来进行任务化模型。
        # 不过官方推荐使用多个数据任务（DDP）以获得更好的性能，并提供相关教程链接。
    # SyncBatchNorm 同步批归一化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")
        # 如果在全局训练中实现同步批量归一化（opt.sync_bn），则使用SyncBatchNorm对模型进行批量归一化转换，以同步不同GPU的批量统计。
    # Trainloader 训练数据加载器
    # 使用create_dataloader函数创建训练数据加载器和数据集。
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
         # 设置图像输入路径 ( train_path)、图像大小 ( imgsz)、批量大小 ( batch_size // WORLD_SIZE)、网格大小 ( gs) 等参数。
        hyp=hyp,
        augment=True, # 表示采用数据增强。
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect, # 是否使用推测推理。
        rank=LOCAL_RANK,
        workers=workers,# 表示使用的数据加载线程数。
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,# 表示打乱数据集顺序。
        seed=opt.seed, # 设置随机种子确保数据顺序一致性。
    )
    labels = np.concatenate(dataset.labels, 0)
    # 获取数据集中所有标签将其拼接为一个阵列，用于后续的标签检查。
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"
    # 计算数据集中最大标签值并保证其不超过类别数量nc，若超出则触发断言错误，提示用户类别数设置不正确。
    # Process 0 验证数据加载器（仅主进程）
    # 设置验证路径 ( val_path)、图像大小 ( imgsz)、批量大小（batch_size // WORLD_SIZE * 2）。
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,# 使用推理，保持图像长宽比。
            rank=-1,
            workers=workers * 2,# 表示使用双倍的线程数来加载数据。
            pad=0.5,# 用于填充图像边界。
            prefix=colorstr("val: "),
        )[0]

        if not resume: # 如果不是继续训练 ( resume=False)：
            if not opt.noautoanchor: #检查并以车辆数据集自动调整框架尺寸。
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision
            #首先将模型转换为半精度 ( half())，然后还原为浮点精度 ( float())，这一步会预先减少框的精度。
        callbacks.run("on_pretrain_routine_end", labels, names)
        # 执行回调函数on_pretrain_routine_end，确定标签和类别名称，用于在训练前执行特定的回调操作。

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
        # 如果使用了多个训练（DDP模式），则将模型包装为多个数据模型（smart_DDP），以支持多GPU良好训练。
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    # 设置模型的相关属性，如检测层数nl、类别权重、标签平滑值、类别数nc和超参数hyp，并根据图像大小和层数调整损失系数。
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    # 初始化训练参数，如批次数nb、热体迭代次数nw、早停机制、损失函数和加速器。设置初始学习率调度器last_epoch为开始的迭代数。
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )
    # 打印训练配置信息，包括图像大小、数据加载器工作线程数和保存结果的目录。
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()
        # 开始按迭代进行训练，在每个新迭代开始时调用回调。
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # 如果实现了权重调整，则根据类别权重cw和每类mAP更新图像的权重iw，用于平衡数据。
        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        # 初始化mloss存储平均损失；设置遍布训练的采样器，更新训练详细条显示信息。
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        # 如果是主进程，则启用进度条pbar，并清除间隙。
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
            # 按批处理进行训练；imgs转换到GPU并标准化为浮点。
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])
                        # 热身阶段通过插值逐步提高学习率和行动量。
            # Multi-scale 如果启用了多尺度训练，则动态调整图像大小以增强模型的视力。
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward 通过放大进行模型前向传播并计算损失，每个模式下累积增量。
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward 执行逆向传播，使用scaler的浮点精度缩放以优化模型，确保梯度更新。
            scaler.scale(loss).backward()
            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log 在主进程上记录损失值，更新进度条和内存使用情况，并检查是否需要提前停止。
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        # 获取当前优化器每个参数组的学习率，存储在lr列表中，用于记录日志和输出。
        scheduler.step()
        # 调用调度器scheduler的step()方法更新学习率，按照预设的学习率调整策略调整优化器的学习率。
        if RANK in {-1, 0}:
            # 仅在主进程（非每隔训练时RANK = -1，每隔时RANK = 0）中执行后续代码。确保多GPU情况下不会重复执行评估和保存模型。
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            # 执行训练结束允许回调on_train_epoch_end，将epoch作为参数，用户在训练结束时插入自定义操作。
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            # 更新ema（指数移动平均模型）属性，导出与当前模型的一些重要属性同步，包括类别数nc、超参数hyp、类别名称names等。
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # 检查是否是最后一个训练轮次，或者通过提前终止条件stopper.possible_stop判断是否停止训练。
            # 如果满足这两个条件中的任意一个，就将final_epoch设置为True。
            if not noval or final_epoch:  # Calculate mAP
                # 如果不跳过验证集评估（noval=False）
                # 或当前为最后一个训练轮次final_epoch，则执行验证集评估计算mAP。
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )
                # 调用验证函数validate.run，使用验证数据集val_loader计算模型在验证集上的各项指标（如准确率P、识别率R、mAP），
                # 并返回评估结果results。该函数还会生成maps，即每一类的mAP。
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # 计算当前 epoch 的适应度fi，将results转换为 numpy 负载并重整为 1xN，fitness函数计算 [P, R, mAP@.5, mAP@.5-.95] 的加权组合值，通常用于比较模型的性能。
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            # 调用stopper的方法，输入当前纪元和fi，判断是否触发早停止条件，将结果赋值给stop。
            if fi > best_fitness:
                best_fitness = fi
                # 如果当前fi比记录的最佳适应度best_fitness更高，则更新best_fitness
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)
            # 将本轮次的损失mloss、评估指标results以及学习率lr汇总为log_vals，
            # 执行回调on_fit_epoch_end并收集这些信息，允许在每个epoch结束时做一些记录或日志更新。
            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                # 判断是否需要保存模型，当nosave=False或当前为final_epoch且非展开搜索阶段时，将保存模型。
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }
                # 创建模型检查点ckpt，包含当前历元、最佳适应度best_fitness、模型和ema状态字典，以及其他相关的训练配置信息，用于保存模型状态和恢复训练。
                # Save last, best and delete
                torch.save(ckpt, last)
                # 将ckpt保存为last，表示最新的模型权重。
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    # 如果当前适应度fi达到最佳适应度best_fitness，则将ckpt保存为best，即保存最优模型。
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                    # 根据save_period设置的保存间隔周期，按纪元保存模型。例如，每个save_period轮次保存一次。
                del ckpt # 删除ckpt，释放内存。
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)
                # 执行回调on_model_save，将模型保存的路径、当前轮次、最终轮次标记final_epoch、最佳适应度best_fitness等信息确定，方便记录或进一步操作。
        # EarlyStopping
        if RANK != -1:  # if DDP training
            # 检查是否有多个任务数据（DDP）模式，仅在多个GPU任务训练时执行接下来的同步代码。
            broadcast_list = [stop if RANK == 0 else None]
            # 定义一个列表broadcast_list，主进程（RANK=0）将停止信号stop置入该列表中，其他进程则初始化为空
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            # 使用broadcast_object_list将主进程的stop值广播给其他进程，使所有进程同步stop的值，确保在多 GPU 训练中提前停止状态的一致性
            if RANK != 0:
                stop = broadcast_list[0]
                # 如果当前进程不是主进程，broadcast_list则读取主进程训练的stop值，以便决定是否提前停止。
        if stop:
            break  # must break all DDP ranks
            # 如果stop=True，则跳出当前纪元循环，结束所有进程的训练。确保在所有进程上统一停止。
        # end epoch ----------------------------------------------------------------------------------------------------
    # 表示 epoch 循环的结束，继续进入下一个 epoch 或跳出训练。
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        # 训练过程结束，仅在主进程上执行以下清理和记录操作。
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        # 记录训练的纪元数和所用时间，并将输出结果写入日志。
        for f in last, best: # 遍历最后一次保存的模型last和最佳模型best，对它们进行进一步处理。
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                # 检查文件是否存在，存在则调用strip_optimizer移除优化器信息，仅保留模型参数，以减少模型文件体积。
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    # 如果当前模型是最佳模型best，输出日志信息提示即将对该模型进行验证。
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    # 调用validate.run函数验证最佳模型的性能，利用验证集val_loader进行评估，计算各项指标并生成相关图表。确定参数包括模型、批量大小、输入图像大小、iou_thres（COCO数据集时为0.65，其他情况为0.60） ，是否以及保存结果为JSON文件等配置。
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)
                        # 若数据集为COCO，调用on_fit_epoch_end回调，得出损失、验证结果、学习率、当前历元、最佳适应度best_fitness等信息用于记录。
        callbacks.run("on_train_end", last, best, epoch, results)
        # 训练结束时，调用on_train_end回调，创建last模型best文件路径、最终epoch及验证结果，允许用户进行自定义清理和记录操作。
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    # 定义parse_opt函数，用于解析命令行参数。参数known指示是否忽略未知参数。
    """
    Parse command-line arguments for YOLOv5 training, validation, and testing.

    Args:
        known (bool, optional): If True, parses known arguments, ignoring the unknown. Defaults to False.

    Returns:
        (argparse.Namespace): Parsed command-line arguments containing options for YOLOv5 execution.

    Example:
        ```python
        from ultralytics.yolo import parse_opt
        opt = parse_opt()
        print(opt)
        ```

    Links:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    # 创建一个ArgumentParser对象，用于添加和解析命令行参数。
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    # 设置模型初始化参数yolov5s.pt，yolov5l.pt,yolov5m.pt,yolov5x.pt
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    # 设置模型 设置模型配置文件的路径，例如自定义模型结构。
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    # 设置数据集配置文件的路径，用于加载训练、验证数据
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    # 设置超参数文件路径，用于配置学习率、权重衰减等训练参数
    parser.add_argument("--epochs", type=int, default=3, help="total training epochs")
    # 设置训练的总epoch数，即模型迭代次数，default=100。
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    # 设置批量大小，指定-1表示自动调整批量大小
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    # 设置训练和验证图像的分辨率（以像素为单位）
    parser.add_argument("--rect", action="store_true", help="rectangular trai ning")
    # 使用长方形训练，使批次中的图像保持原始长宽比
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    # 恢复最近一次的训练，从上次中断的地方继续训练
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    # 设置为仅保存最终的检查点，减少中间保存文件
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    # 设置为仅在最后一个epoch进行验证
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    # 取消自动框生成功能
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    # 设置为不保存图纸文件，以减少存储开销
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    # 用于自动进化超参数，设置迭代次数const=300
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    # 设置超参数进化的群体数据加载路径
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    # 恢复进化过程，从最后一个进化代开始继续
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    # 设置用于 Google Cloud Storage 存储桶的存储模型或数据
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ra")
    # 缓存数据集到内存或磁盘，加快加载速度                                                                      "m/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    # 利用图像加权选择策略，增强模型对罕见类别的训练
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 设置用于训练的计算设备，例如GPU数量或CPU
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    # 使用多尺测量，随机调整图像尺寸
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    # 将多类别数据视为单个类别进行训练
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    # 设置优化器类型，可选择SGD、Adam 或 AdamW
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    # 使用同步BatchNorm，仅适用于多种训练模式
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 设置数据加载器的最大工作者数量
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    # 设置保存项目结果的路径
    parser.add_argument("--name", default="exp", help="save to project/name")
    # 设置保存项目的实验名称
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    # 如果目录存在则不自动增加版本号
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    # 启用四路数据加载器，以加快加载速度
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    # 启用余弦学习率调度器
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    # 设置平滑系数标签，以减少过度
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    # 设置早停止的耐心值，即在多少个epoch无改进后停止训练
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    # 冻结部分模型层数，适用于迁移学习
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    # 设置检查点保存周期，低于1则取消
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    # 设置全民随机种子，保证结果可重复
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    # 自动分散训练的GPU进程分配标志，用户消耗修改
    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    # 设置相关的实体名称
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    # 用于上传数据集的选项
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    # 设置边界框图像记录间隔
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")
    # 设置数据集版本别名
    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    # 启用控制台的NDJSON格式日志记录
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")
    # 启用文件的NDJSON格式日志记录
    return parser.parse_known_args()[0] if known else parser.parse_args()
# 根据known参数返回选择已知参数或所有解析的参数

def main(opt, callbacks=Callbacks()):
    # 定义main函数，主入口用于训练或超参数进化。opt包含命令行解析的选项，callbacks用于各阶段的回调函数。
    """
    Runs the main entry point for training or hyperparameter evolution with specified options and optional callbacks.

    Args:
        opt (argparse.Namespace): The command-line arguments parsed for YOLOv5 training and evolution.
        callbacks (ultralytics.utils.callbacks.Callbacks, optional): Callback functions for various training stages.
            Defaults to Callbacks().

    Returns:
        None

    Note:
        For detailed usage, refer to:
        https://github.com/ultralytics/yolov5/tree/master/models
    """
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")
        # 判断是否在主进程（RANK即为-1或0）中执行。print_args打印命令行参数；
        # check_git_status检查代码是否为最新版本；check_requirements检查所需依赖

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        # 判断是否恢复训练。若resume参数启用且未进行超参数进化，尝试从指定模型或最新的last.pt文件恢复训练。
        # last表示最新模型文件路径，opt_yaml为训练选项的YAML配置文件路径，opt_data保存原始数据集路径。
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        # 检查并读取opt.yaml文件内容（若存在），否则从模型文件加载opt配置，使用Namespace更新命令行参数。
        # 清空cfg，放置重路径并resume设置为last模型文件
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
            # 如果路径为URL，则检查文件缺陷，使用在线资源可能产生的授权数据超时。
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        # 否则，逐一检查数据路径、模型配置、超参数文件和权限重路径的有效性，并确保cfg或weights至少设置一个
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
            # 若启用超参数进化模式，将默认的项目路径改为runs/evolve，同时保留resume设置给exist_ok参数，取消resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
            #若name参数为cfg，则使用cfg文件名作为项目名称
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        # 使用increment_path确保保存目录唯一化，为每次实验生成不同的目录
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 选择训练设备，根据device参数选择GPU或CPU，支持多GPU设置
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        # 在多个数据玩具（DDP）模式下，进行必要的参数兼容性检查：确保image_weights、、evolve的batch_size设置适用于多GPU，batch_size且能被WORLD_SIZE整除。同时检查CUDA设备数量是否足够
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )
        # 设置 CUDA 设备编号，根据LOCAL_RANK为每个进程指定 GPU。
        # 初始化 DDP 进程组，使用 NCCL 或 Gloo 作为耳机，并设置超时时间为 3 小时

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        # 判断是否进行超参数进化。如果opt.evolve为False，则直接调用train函数，启动模型训练。
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mixup (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),
        }  # segment copy-paste (probability)
        # meta这些超参数涵盖了模型训练中的学习率、动量、损失函数等关键参数

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10
        # 定义遗传算法（GA）的一些配置项，包括种群大小（pop_size），突变率和交叉率的上下限，最强数量范围，以及集群选择的竞争者数量范围。这些参数控制 GA 的进化过程
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
                # 文件opt.hyp（超参数文件），读取并加载为hyp字典。若超参数文件中未定义anchors参数，则将anchors设置为默认值3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
            # 如果命令行参数noautoanchor为真，则从超参数和meta中删除anchors，取消自动生成锚点
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # 将noval和nosave设置True，仅在最后一个纪元进行验证和保存；将保存路径save_dir设置opt.save_dir
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        # 设置超参数进化文件的保存路径。hyp_evolve.yaml保存进化后的超参数，evolve.csv保存进化的每一代结果数据
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )
            # 如果指定了bucket参数，则从 Google Cloud Storage (GCS) 下载现有的evolve.csv文件，用于在演化前继续过去的演化记录。gsutil cp命令将文件复制到evolve_csv的本地路径
        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary
            # 首先过滤掉meta字典中不参与进化的超参数（即False），并从hyp_GA（超参数字典的副本）中删除相应的项，以简化优化的超参数集合

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])
        # 为进化超参数设置边界，lower_limit和upper_limit存储了每个超参数的搜索空间上下限

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]
        # 创建gene_ranges列表，每个基因（超参数）包含搜索范围，用于生成初始种群
        # Initialize the population with initial_values or random values
        initial_values = [] # 初始化initial_values列表，用于存储仓库中的个体

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            # 这行代码是一个条件判断语句的开始，用于检查是否要从上一次进化的检查点恢复进化过程。
            # 如果opt.resume_evolve的值不为None，表示需要从之前的检查点继续进行进化操作
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            # 当满足从检查点恢复的条件时，这行代码使用assert语句进行断言检查。它会验证opt.resume_evolve所指定的文件路径是否存在且是一个文件，
            # 如果不满足该条件（即文件不存在或不是文件），就会抛出异常并显示指定的错误信息 "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                # 在确认文件路径正确后，这行代码使用with语句打开opt.resume_evolve所指定的文件，以便后续读取文件内容。
                # 设置errors="ignore"参数是为了在读取文件过程中忽略可能出现的编码错误等问题。
                evolve_population = yaml.safe_load(f)
                #读取打开的文件内容，并使用yaml.safe_load函数将文件内容（假设是 YAML 格式）解析为一个 Python 对象，
                # 然后将解析后的结果赋值给evolve_population变量，该变量可能存储了之前进化过程中的相关种群数据等信息
                for value in evolve_population.values():
                    # 开始遍历evolve_population对象中的每个值。这里的每个值可能对应着之前进化过程中保存的个体相关的数据结构
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    # 对于遍历到的每个值，通过列表推导式从该值中提取出与hyp_GA字典的键相对应的值，并将这些值转换为一个numpy数组，然后重新赋值给value变量。
                    initial_values.append(list(value))
                    # 将经过处理后的value（此时已转换为列表形式）添加到initial_values列表中。initial_values列表可能用于存储初始化种群所需的一些初始值，在这里就是从恢复的检查点文件中读取并整理好的个体数据
                    # 如果指定了从上一次进化检查点恢复，则读取resume_evolve文件，将已保存的群体加载到initial_values列表
        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            # 使用列表推导式遍历opt.evolve_population指定的文件夹中的所有文件，筛选出以.yaml结尾的文件，并将这些文件名存储在yaml_files列表中
            # 这些.yaml文件可能包含了用于生成初始种群的相关数据。
            for file_name in yaml_files:
                 # 开始遍历yaml_files列表中的每个文件名，以便逐个处理这些.yaml文件。
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    # 对于每个文件名，使用with语句打开对应的.yaml文件，准备读取文件内容。
                    value = yaml.safe_load(yaml_file)
                    #取打开的.yaml文件内容，并使用yaml.safe_load函数将文件内容解析为一个 Python 对象，然后将解析后的结果赋值给value变量
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    # 与前面从恢复检查点文件读取数据时类似，这里也是从解析后的value对象中提取出与hyp_GA字典的键相对应的值，并将这些值转换为一个numpy数组，然后重新赋值给value变量
                    initial_values.append(list(value))
                    # 将经过处理后的value（此时已转换为列表形式）添加到initial_values列表中。
                    # 同样，initial_values列表用于存储初始化种群所需的一些初始值，这里是从.yaml文件中读取并整理好的个体数据。
                    # 如果没有恢复进化，则从指定的evolve_population文件夹中读取.yaml文件，生成初始种群

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            # 这是一个新的条件判断语句的开始，用于检查initial_values列表是否为空。如果为空，表示可能还没有通过前面两种方式（从检查点恢复或从.yaml文件读取）获取到足够的初始种群数据，需要随机生成种群。
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
            # 当initial_values为空时，这行代码使用列表推导式调用generate_individual函数（假设该函数已在别处定义且功能是在给定的基因范围gene_ranges内生成指定长度len(hyp_GA)的个体），循环pop_size次来随机生成整个种群，并将生成的种群存储在population变量中
        elif pop_size > 1:
            # 是与前面if语句配套的else if分支，当initial_values不为空且pop_size大于1时，会执行这个分支的代码。
            # 它的目的是将initial_values中的个体加入到种群中，并根据需要随机生成剩余部分的个体，以填满到指定的pop_size
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            # 首先，这行代码使用列表推导式调用generate_individual函数，循环pop_size - len(initial_values)次来随机生成一部分个体，这些个体将与initial_values中的个体一起组成完整的种群。生成的这部分个体数量是为了补足到指定的pop_size，减去了已经从initial_values中获取到的个体数量。
            for initial_value in initial_values:
                # 开始遍历initial_values列表中的每个初始值（即之前获取到的个体数据）
                population = [initial_value] + population
                # 将遍历到的每个初始值添加到population变量所存储的种群列表的开头，这样就将initial_values中的个体逐个加入到了种群中
        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        #创建一个列表list_keys，通过将hyp_GA字典的键转换为列表形式来获取
        for generation in range(opt.evolve):
            # 开始一个循环，循环次数由opt.evolve指定。这个循环的目的是运行遗传算法指定的代数，在每一代中都会进行一系列的操作，如评估个体适应度、选择个体进行繁殖、生成下一代种群等
            if generation >= 1:
                # 在每一代的循环内部，这是一个条件判断语句，当generation大于等于1时，会执行下面的代码。它的目的可能是在第一代之后进行一些特定的操作，比如保存种群数据等。
                save_dict = {}
                # 当满足generation大于等于1的条件时，初始化一个空字典save_dict，这个字典将用于保存当前代种群的相关数据，以便后续可以将这些数据保存到文件中，实现每代结果的可追溯性
                for i in range(len(population)):
                    # 开始一个内层循环，循环次数为种群population的长度。这个循环的目的是遍历种群中的每个个体，以便为每个个体构建一个字典，并将这些字典添加到save_dict中
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    # 使用列表推导式为种群中的第i个个体构建一个小字典little_dict。通过遍历个体中的每个元素（对应hyp_GA字典的键），将键和对应的元素值（转换为浮点数）组成键值对添加到小字典中。
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict
                    #将构建好的小字典little_dict添加到save_dict大字典中，键为gen{str(generation)}number{str(i)}，这样就可以通过这个键在保存的文件中找到对应代、对应个体的相关数据。
                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    # 当完成对种群中所有个体数据的整理并添加到save_dict后，使用with语句以写入模式打开save_dir目录下的evolve_population.yaml文件，准备将save_dict中的数据保存到该文件中。
                    yaml.dump(save_dict, outfile, default_flow_style=False)
                    # 使用yaml.dump函数将save_dict字典中的数据以 YAML 格式写入到打开的evolve_population.yaml文件中，设置default_flow_style=False参数是为了让输出的 YAML 格式更易读，采用块状格式而不是流式格式。
                    # 运行遗传算法，在每一代进化时将当前种群保存到evolve_population.yaml，确保每代结果可追溯
            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # 在每一代的循环内部，这行代码用于计算当前代的精英个体数量elite_size。它根据当前的代数generation、总进化代数opt.evolve以及预先设定的最小精英数量min_elite_size和最大精英数量max_elite_size，通过一个线性计算公式来动态调整精英个体的数量，使得精英数量随着代数的增加而适当变化。
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            # 初始化一个空列表fitness_scores，这个列表将用于存储种群中每个个体的适应度得分。适应度得分通常是用来衡量个体在遗传算法中的优劣程度，可能通过对个体进行某种评估（如训练结果等）来获取
            for individual in population:
                # 开始遍历种群population中的每个个体，以便对每个个体进行适应度评估
                for key, value in zip(hyp_GA.keys(), individual):
                    # 对于遍历到的每个个体，通过zip函数将hyp_GA字典的键和个体中的元素一一对应起来，然后进行循环遍历，以便为每个键值对进行相应的操作。
                    hyp_GA[key] = value
                    # 将个体中的元素值赋给hyp_GA字典中对应的键，这样就更新了hyp_GA字典的内容，可能是为了根据个体的具体情况设置相关的超参数等
                hyp.update(hyp_GA)
                # 使用hyp对象（假设已在别处定义）的update方法，将更新后的hyp_GA字典内容更新到hyp对象中，这可能是为了将新的超参数设置应用到后续的训练或评估等操作中
                results = train(hyp.copy(), opt, device, callbacks)
                # 调用train函数（假设已在别处定义），传入更新后的超参数hyp.copy()、配置选项opt、设备信息device和回调函数callbacks，并将函数返回的结果赋值给results变量。这里的train函数可能是进行实际训练操作的函数，通过训练来评估个体的适应度。
                callbacks = Callbacks()
                # 重新初始化callbacks对象（假设已在别处定义）为一个新的Callbacks类的实例。这可能是因为在前面调用train函数后，callbacks对象的状态可能已经发生了变化，需要重新初始化以便进行下一次的训练或评估操作
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                # 定义一个元组keys，其中包含了一些用于记录训练结果的指标名称，如精度、召回率、平均精度等。这些指标将用于后续打印和保存突变结果等操作
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                # 调用print_mutation函数（假设已在别处定义），传入定义好的指标元组keys、训练结果results、复制后的超参数hyp.copy()、保存目录save_dir和桶名称opt.bucket，用于打印和保存与突变相关的结果。
                fitness_scores.append(results[2])
                # 将训练结果results中的第2个元素（具体含义可能取决于train函数的返回值结构）作为个体的适应度得分，添加到fitness_scores列表中，完成对一个个体适应度的评估并记录其得分
                # 评估种群中每个个体的适应度（即训练结果），把这些结果添加到fitness_scores列表中

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            # 初始化一个空列表selected_indices，这个列表将用于存储通过自适应锦标赛选择方法选出的最适合繁殖的个体的索引。
            for _ in range(pop_size - elite_size):
                # 开始一个循环，循环次数为pop_size - elite_size，即除了精英个体之外需要选择的个体数量。这个循环的目的是通过自适应锦标赛选择方法来选择出这些个体
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # 在每一轮选择个体的循环内部，这行代码用于计算当前轮的锦标赛规模tournament_size。它根据预先设定的最小锦标赛规模tournament_size_min、最大锦标赛规模tournament_size_max以及当前的代数generation和总进化代数opt.evolve，通过一系列比较和计算来动态调整锦标赛规模，使得锦标赛规模随着代数的增加而适当变化。
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                # 根据计算出的锦标赛规模tournament_size，使用random.sample函数从种群population的索引范围（0到pop_size - 1）中随机抽取tournament_size个索引，这些索引组成的列表就是tournament_indices，用于表示参与本次锦标赛的个体索引。
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                # 通过列表推导式，根据tournament_indices中的索引从fitness_scores列表中提取对应的适应度得分，生成一个新的列表tournament_fitness。这个列表包含了参与本次锦标赛选择的个体的适应度得分，用于后续确定本次锦标赛的获胜者。
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                # 首先在tournament_fitness列表中找到最大值，通过index方法获取该最大值在列表中的索引，然后将这个索引作为tournament_indices的索引，从而得到在本次锦标赛中具有最高适应度得分的个体在种群中的索引，将该索引赋值给winner_index变量。这个索引所对应的个体就是本次锦标赛选择出的最适合繁殖的个体。
                selected_indices.append(winner_index)
                # 将通过锦标赛选择出的获胜个体的索引winner_index添加到selected_indices列表中。selected_indices列表用于存储经过各种选择方式选出的适合繁殖的个体的索引，以便后续基于这些个体创建下一代种群。

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            # 通过列表推导式遍历从0到pop_size - 1的索引范围，对于每个索引i，检查其对应的fitness_scores中的适应度得分是否在经过排序后的fitness_scores列表的最后elite_size个元素中（即是否属于精英个体的适应度得分范围）。如果满足条件，则将该索引添加到elite_indices列表中。这样就找到了所有精英个体在种群中的索引。
            selected_indices.extend(elite_indices)
            # 将elite_indices列表中的所有精英个体的索引添加到selected_indices列表中。这一步确保了精英个体的索引也被包含在用于创建下一代种群的选择索引列表中，使得精英个体能够直接进入下一代种群，以保留优秀的基因特性。
            # 将精英个体加入到selected_indices中，确保他们直接进入下一代
            # Create the next generation through crossover and mutation
            next_generation = []
            # 初始化一个空列表next_generation，用于存储通过交叉和变异操作生成的下一代种群中的个体。
            for _ in range(pop_size):
                # 开始一个循环，循环次数为pop_size，即要生成的下一代种群的个体数量。在每次循环中，将执行一系列操作来生成一个新的个体并添加到next_generation列表中
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                # 从selected_indices列表中随机选择一个索引作为第一个父代个体的索引，将其赋值给parent1_index变量。这个父代个体将参与后续的交叉和变异操作来生成新的个体。
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # 同样从selected_indices列表中再次随机选择一个索引作为第二个父代个体的索引，将其赋值给parent2_index变量。这两个父代个体将一起用于生成新的个体。
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                #根据当前的代数generation、总进化代数opt.evolve以及预先设定的最小交叉率crossover_rate_min和最大交叉率crossover_rate_max，通过一系列比较和计算来动态调整交叉率crossover_rate。使得交叉率随着代数的增加而适当变化，以适应遗传算法在不同阶段的需求
                if random.uniform(0, 1) < crossover_rate:
                    # 生成一个在0到1之间的随机数，并与当前计算出的交叉率crossover_rate进行比较。如果随机数小于交叉率，说明满足交叉条件，将执行下面的交叉操作来生成新的个体；否则，将直接使用第一个父代个体作为新的个体（即不进行交叉操作）
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    # 当满足交叉条件时，随机生成一个整数作为交叉点。这个交叉点的取值范围是从1到hyp_GA字典长度减1，用于确定在两个父代个体上进行交叉操作的位置
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                    # 根据生成的交叉点，将第一个父代个体population[parent1_index]从开头到交叉点位置的部分与第二个父代个体population[parent2_index]从交叉点位置到末尾的部分拼接起来，生成一个新的个体child。这就是通过交叉操作生成新个体的过程。
                else:
                    child = population[parent1_index]
                    # 当不满足交叉条件（即随机数大于等于交叉率）时，直接将第一个父代个体population[parent1_index]赋值给child变量，作为新生成的个体。这意味着在这种情况下，新个体直接继承了第一个父代个体的基因，没有进行交叉操作。
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                # 类似于计算交叉率的方式，根据当前的代数generation、总进化代数opt.evolve以及预先设定的最小变异率mutation_rate_min和最大变异率mutation_rate_max，通过一系列比较和计算来动态调整变异率mutation_rate。使得变异率随着代数的增加而适当变化，以适应遗传算法在不同阶段的需求
                for j in range(len(hyp_GA)):
                    # 开始一个循环，循环次数为hyp_GA字典的长度。这个循环的目的是遍历新生成的个体（child）中的每个基因位置（对应hyp_GA字典的键），以便对每个基因进行变异操作（如果满足变异条件）。
                    if random.uniform(0, 1) < mutation_rate:
                        # 生成一个在0到1之间的随机数，并与当前计算出的变异率mutation_rate进行比较。如果随机数小于变异率，说明满足变异条件，将执行下面的变异操作来改变个体的基因值；否则，不进行变异操作，直接保留当前基因值
                        child[j] += random.uniform(-0.1, 0.1)
                        # 当满足变异条件时，对新生成的个体child在第j个基因位置上的值进行变异操作。通过添加一个在-0.1到0.1之间的随机数来改变该基因的值，实现基因的变异。
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                        # 在对基因值进行变异操作后，为了确保变异后的基因值仍然在预先设定的基因范围gene_ranges内，通过取最大值和最小值的操作来限制基因值。先取child[j]与基因范围下限gene_ranges[j][0]中的最大值，再取这个结果与基因范围上限gene_ranges[j][1]中的最小值，将最终结果重新赋值给child[j]，从而保证基因值在合理范围内。
                next_generation.append(child)
                # 将经过交叉和变异操作（如果有）生成的新个体child添加到next_generation列表中。经过多次循环后，next_generation列表将包含完整的下一代种群个体。
            # Replace the old population with the new generation
            population = next_generation
            # 将生成的下一代种群next_generation赋值给population变量，从而完成用新一代种群替换旧种群的操作。这样，遗传算法就进入了下一轮迭代，基于新的种群继续进行进化操作。
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        # 在整个遗传算法运行结束后，通过在fitness_scores列表中找到最大值，再使用index方法获取该最大值在列表中的索引，将这个索引赋值给best_index变量。这个索引对应的个体在种群中具有最高的适应度得分，即被认为是找到的最佳解决方案中的个体索引。
        best_individual = population[best_index]
        # 根据找到的最佳个体索引best_index，从最终的种群population中获取对应的个体，将其赋值给best_individual变量。这个个体就是在整个遗传算法运行过程中找到的最佳解决方案所对应的个体。
        print("Best solution found:", best_individual)
        # 将找到的最佳个体best_individual打印出来，输出信息为 "Best solution found:" 加上最佳个体的具体内容，以便用户直观地看到遗传算法最终找到的最佳解决方案。
        # Plot results
        plot_evolve(evolve_csv)
        # 调用plot_evolve函数（假设已在别处定义），传入evolve_csv参数（具体含义可能取决于函数定义），用于绘制与遗传算法进化过程相关的结果图表，以便更直观地展示算法的运行效果和数据变化情况。
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'# 超参数进化已经完成了opt.evolve代
            f"Results saved to {colorstr('bold', save_dir)}\n"# 告知用户结果已经保存到了通过colorstr('bold', save_dir)指定的目录中
            f'Usage example: $ python train.py --hyp {evolve_yaml}' # 给出了一个使用示例，即如何通过命令行运行python train.py --hyp {evolve_yaml}来进行相关操作
        )



def generate_individual(input_ranges, individual_length):
    """
    Generate an individual with random hyperparameters within specified ranges.

    Args:
        input_ranges (list[tuple[float, float]]): List of tuples where each tuple contains the lower and upper bounds
            for the corresponding gene (hyperparameter).
        individual_length (int): The number of genes (hyperparameters) in the individual.

    Returns:
        list[float]: A list representing a generated individual with random gene values within the specified ranges.

    Example:
        ```python
        input_ranges = [(0.01, 0.1), (0.1, 1.0), (0.9, 2.0)]
        individual_length = 3
        individual = generate_individual(input_ranges, individual_length)
        print(individual)  # Output: [0.035, 0.678, 1.456] (example output)
        ```

    Note:
        The individual returned will have a length equal to `individual_length`, with each gene value being a floating-point
        number within its specified range in `input_ranges`.
    """
    individual = []
    for i in range(individual_length):
        # 开始一个循环，循环次数由individual_length决定。这个循环的目的是为了逐个生成个体中的每个基因（超参数）的值。
        lower_bound, upper_bound = input_ranges[i]
        # 在每次循环中，从input_ranges列表中取出第i个元组，将元组中的两个值分别赋给lower_bound和upper_bound，这两个值分别代表当前要生成的超参数的下限和上限。
        individual.append(random.uniform(lower_bound, upper_bound))
        # 使用random.uniform函数生成一个在lower_bound和upper_bound之间的随机浮点数，并将其添加到individual列表中。这样就完成了个体中一个超参数值的生成
    return individual
# 循环结束后，将生成好的包含随机超参数值的individual列表作为函数的返回值返回。

def run(**kwargs):
    """
    Execute YOLOv5 training with specified options, allowing optional overrides through keyword arguments.

    Args:
        weights (str, optional): Path to initial weights. Defaults to ROOT / 'yolov5s.pt'.
        cfg (str, optional): Path to model YAML configuration. Defaults to an empty string.
        data (str, optional): Path to dataset YAML configuration. Defaults to ROOT / 'data/coco128.yaml'.
        hyp (str, optional): Path to hyperparameters YAML configuration. Defaults to ROOT / 'data/hyps/hyp.scratch-low.yaml'.
        epochs (int, optional): Total number of training epochs. Defaults to 100.
        batch_size (int, optional): Total batch size for all GPUs. Use -1 for automatic batch size determination. Defaults to 16.
        imgsz (int, optional): Image size (pixels) for training and validation. Defaults to 640.
        rect (bool, optional): Use rectangular training. Defaults to False.
        resume (bool | str, optional): Resume most recent training with an optional path. Defaults to False.
        nosave (bool, optional): Only save the final checkpoint. Defaults to False.
        noval (bool, optional): Only validate at the final epoch. Defaults to False.
        noautoanchor (bool, optional): Disable AutoAnchor. Defaults to False.
        noplots (bool, optional): Do not save plot files. Defaults to False.
        evolve (int, optional): Evolve hyperparameters for a specified number of generations. Use 300 if provided without a
            value.
        evolve_population (str, optional): Directory for loading population during evolution. Defaults to ROOT / 'data/ hyps'.
        resume_evolve (str, optional): Resume hyperparameter evolution from the last generation. Defaults to None.
        bucket (str, optional): gsutil bucket for saving checkpoints. Defaults to an empty string.
        cache (str, optional): Cache image data in 'ram' or 'disk'. Defaults to None.
        image_weights (bool, optional): Use weighted image selection for training. Defaults to False.
        device (str, optional): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu'. Defaults to an empty string.
        multi_scale (bool, optional): Use multi-scale training, varying image size by ±50%. Defaults to False.
        single_cls (bool, optional): Train with multi-class data as single-class. Defaults to False.
        optimizer (str, optional): Optimizer type, choices are ['SGD', 'Adam', 'AdamW']. Defaults to 'SGD'.
        sync_bn (bool, optional): Use synchronized BatchNorm, only available in DDP mode. Defaults to False.
        workers (int, optional): Maximum dataloader workers per rank in DDP mode. Defaults to 8.
        project (str, optional): Directory for saving training runs. Defaults to ROOT / 'runs/train'.
        name (str, optional): Name for saving the training run. Defaults to 'exp'.
        exist_ok (bool, optional): Allow existing project/name without incrementing. Defaults to False.
        quad (bool, optional): Use quad dataloader. Defaults to False.
        cos_lr (bool, optional): Use cosine learning rate scheduler. Defaults to False.
        label_smoothing (float, optional): Label smoothing epsilon value. Defaults to 0.0.
        patience (int, optional): Patience for early stopping, measured in epochs without improvement. Defaults to 100.
        freeze (list, optional): Layers to freeze, e.g., backbone=10, first 3 layers = [0, 1, 2]. Defaults to [0].
        save_period (int, optional): Frequency in epochs to save checkpoints. Disabled if < 1. Defaults to -1.
        seed (int, optional): Global training random seed. Defaults to 0.
        local_rank (int, optional): Automatic DDP Multi-GPU argument. Do not modify. Defaults to -1.

    Returns:
        None: The function initiates YOLOv5 training or hyperparameter evolution based on the provided options.

    Examples:
        ```python
        import train
        train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
        ```

    Notes:
        - Models: https://github.com/ultralytics/yolov5/tree/master/models
        - Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        - Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    opt = parse_opt(True)
    # 调用parse_opt函数（假设该函数已在别处定义）并传入参数True，该函数可能用于解析命令行参数或配置文件等，返回一个包含各种配置选项的对象opt
    for k, v in kwargs.items():
        # 开始遍历传入函数的关键字参数kwargs中的每个键值对。kwargs是一个字典形式的参数，允许调用者通过关键字传递任意数量的参数来覆盖默认配置
        setattr(opt, k, v)
        # 对于遍历到的每个键值对，使用setattr函数将键k对应的属性值设置为值v，也就是用传入的关键字参数的值来更新opt对象中的相应属性，实现对默认配置的覆盖。
    main(opt)
    # 调用main函数（假设该函数已在别处定义）并传入更新后的配置对象opt，main函数可能是实际执行 YOLOv5 训练或超参数进化核心逻辑的函数。
    return opt
# 将更新后的配置对象opt作为函数的返回值返回，调用者可能会根据需要进一步处理这个配置对象。


if __name__ == "__main__":
    # 这是 Python 脚本的主入口部分。当脚本作为主程序直接运行时（而不是作为模块被导入到其他程序中），会执行这部分代码。
    opt = parse_opt()
    # 调用parse_opt函数（这里没有传入参数，可能会使用默认配置进行解析），得到一个配置对象opt
    main(opt)
    # 将opt对象传入main函数（同样假设该函数已在别处定义），启动 YOLOv5 的相关训练或处理流程
