#**HDDI-Net: Lightweight Dual-Domain Interaction for Robust Ultrasound Lesion Segmentation**


This repository is the official implementation of the paper: "HDDI-Net: Lightweight Dual-Domain Interaction for Robust Ultrasound Lesion Segmentation".

📌 Introduction

HDDI-Net is a lightweight, noise-aware segmentation network designed for B-mode ultrasound. It addresses the challenges of speckle interference and ambiguous margins by integrating spatial and frequency-domain reasoning within a mobile-scale computational budget (~1.7M parameters).

Key Features

Hierarchical Dual-Domain Interaction (HDDI): A block that jointly learns multi-kernel spatial morphology and DCT-domain frequency gating to suppress speckle noise.


Prototype-Guided Semantic Consistency (PGSC): A module that approximates long-range context using learnable semantic anchors without the quadratic cost of self-attention.


Macro-Micro ROI Framework: A coarse-to-fine pipeline that geometrically normalizes inputs to improve cross-site robustness.

🛠️ Environment and Installation
The code is implemented using PyTorch and was tested on an NVIDIA RTX 4090 GPU.

Requirements
Python 3.x

PyTorch >= 1.10

Albumentations (for data augmentation) 

Torchvision

Numpy, OpenCV, Matplotlib

📊 Datasets
Our model is evaluated on three diverse ultrasound datasets:


BUSI (Breast): 647 images for training and internal validation.


TN3K (Thyroid): 3,073 images for large-scale multi-organ benchmarking.


UDIAT (Breast): Used for Zero-Shot cross-dataset generalization testing.
