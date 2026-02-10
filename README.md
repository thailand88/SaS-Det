# SaS-Det
# SaS-Det: Sliced Segmentation Integration and Super-Resolution Reconstruction for Long-Range Small Object Detection in Railway Scenarios

This repository provides the official implementation of the paper:

> **SaS-Det: Sliced Segmentation Integration and Super-Resolution Reconstruction for Long-Range Small Object Detection in Railway Scenarios**  
> *[Journal Name] (under review)*

---

## 📌 Overview

In long-range railway imaging scenarios, small objects are constrained by extremely low pixel proportions and significant image degradation, which often leads to irreversible loss of structural and textural information during feature downsampling and poses substantial challenges for accurate detection.  
This project presents a **unified framework for long-range small object detection** (termed SaS-Det), which integrates SSI (Sliced Segmentation Integration), LRST-ESRGAN (Super-Resolution Reconstruction), and STG-YOLO11 in a unified manner.  

The proposed framework leverages **spatial redundancy suppression**, **fine-grained detail restoration**, and **multi-scale contextual modeling enhancement** to mitigate feature degradation of small objects under long-range imaging conditions, thereby improving the discriminability and robustness of sparse small object detection in railway scenarios.

---

## ✨ Key Contributions

- **Sliced Segmentation Integration (SSI)**  
  A SSI module is designed to suppress spatial redundancy in long-range railway images, focusing computational resources on small object regions and reducing irrelevant background interference.

- **LRST-ESRGAN Super-Resolution Reconstruction**  
  A LRST-ESRGAN model is developed to restore fine-grained structural and textural information of small objects, alleviating irreversible feature loss caused by downsampling.

- **STG-YOLO11 Detection Backbone**  
  An enhanced STG-YOLO11 architecture is proposed to strengthen multi-scale contextual modeling, improving the detection accuracy of sparse small objects in railway scenarios.

- **Long-Range Railway Small Object Dataset**  
  A dedicated long-range railway small object detection dataset is constructed to support systematic evaluation and future research in this field.

---

## 🧠 Framework Overview

<p align="center">
  <img src="1.png" width="800">
</p>

*Figure: Overall framework of the proposed SaS-Det for long-range small object detection in railway scenarios.*

---

## 📊 Experimental Evaluation

The proposed framework is evaluated using:
- **Constructed long-range railway small object detection dataset**
- **Cross-dataset validation (BDD100K, SRSDD-V1.0, AI-TOD)**

Experimental results demonstrate:
- Approximately 21% improvement in mAP@0.5 compared with the YOLO11s baseline
- Consistent gains in Precision, Recall, and other evaluation metrics
- Strong generalization capability across diverse scenes and data distributions
- Stable performance for sparse small object detection under long-range imaging conditions

Detailed quantitative results and ablation studies can be found in the paper.

---

## 📁 Dataset

The dataset contains:
- Long-range railway imaging scenes
- Diverse small objects (typical railway scenario targets)
- Multi-scale and multi-degradation image samples
- Annotated labels for small object detection

> ⚠️ Due to privacy and institutional regulations, the dataset is **not publicly downloadable** at this stage.  
> Please contact the authors for academic collaboration.

---

## 🚀 Getting Started

### Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- torchvision  
- timm  
- numpy, scipy, opencv-python  
- ultralytics (for YOLO11)
- scikit-image (for super-resolution)

Install dependencies:
```bash
pip install -r requirements.txt
