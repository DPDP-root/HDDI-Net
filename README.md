# HDDI-Net: Hierarchical Dual-Domain Interaction Network for Robust and Efficient Ultrasound Lesion Segmentation

This is the official repository for **HDDI-Net**, an extremely lightweight (**1.71M parameters**, ~95% fewer than TransUnet) and efficient deep learning framework for robust breast ultrasound lesion segmentation. HDDI-Net combines a **Hierarchical Dual-Domain Interaction (HDDI) Block** for simultaneous spatial texture capture and frequency-domain speckle noise removal, with a **Coarse-to-Fine ROI-aware Framework** to suppress complex background interference.

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg'></a>
<a href='#'><img src='https://img.shields.io/badge/Journal-The%20Visual%20Computer-orange'></a>
<a href='#'><img src='https://img.shields.io/badge/Status-Under%20Review-yellow'></a>
</div>

---

<p align="center" width="100%">
<a target="_blank"><img src="HDDI/imgs/figure1.1.png" alt="HDDI-Net Architecture" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## 🔥 Updates
* **[2026-03]** 🚀 Official code and training/evaluation scripts released.
* **[2026-02]** ⭐️ HDDI-Net achieves competitive SOTA on BUSI and superior **zero-shot generalization** on the unseen UDIAT dataset with only **1.71M parameters**.

---

## 🎯 Overview

- We propose **HDDI-Net**, a lightweight yet robust network that achieves an optimal trade-off between segmentation accuracy, computational efficiency, and cross-center generalizability.
- **Coarse-to-Fine ROI Framework**: A two-stage pipeline that first roughly localizes the lesion to filter background noise, then performs fine-grained segmentation within the localized region, significantly improving Precision.
- **Hierarchical Dual-Domain Interaction (HDDI) Block**: Integrates multi-kernel spatial convolution (3×3, 5×5, 7×7 depthwise separable conv) with a DCT-based frequency gating mechanism to perform **physiologically-inspired adaptive speckle denoising**.
- **Prototype-Guided Semantic Consistency**: Implements intra-image self-clustering at the bottleneck to model long-range dependencies without heavy Transformer attention, ensuring compact lesion representations even at batch size = 1.
- **Multi-Scale Deep Supervision**: Applies auxiliary losses at each decoder stage with dynamic weights to accelerate convergence and improve feature learning in the lightweight backbone.

---

## 🕹️ Usage

### 1. Environment Setup

```bash
conda create -n hddinet python=3.9 -y
conda activate hddinet
pip install -r requirements.txt
```

### 2. Data Preparation

Download datasets (e.g., [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset), [UDIAT](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)) and organize them as follows:

```
.
└── data
    ├── busi
    │   ├── images
    │   │   ├── benign (10).png
    │   │   ├── malignant (17).png
    │   │   └── ...
    │   └── masks
    │       └── 0
    │           ├── benign (10).png
    │           └── ...
    └── udiat
        ├── images
        └── masks
```

### 3. Training

Train HDDI-Net on the BUSI dataset (internal domain):

```bash
conda activate hddinet
python main.py --max_epochs 100 --gpu 0 --batch_size 8 \
               --model HDDI_Net \
               --base_dir ./data/busi \
               --dataset_name busi
```

Or use the provided Windows batch script:

```bash
run_train.bat
```

### 4. In-Domain Inference

```bash
conda activate hddinet
python main.py --max_epochs 100 --gpu 0 --batch_size 8 \
               --model HDDI_Net \
               --base_dir ./data/busi \
               --dataset_name busi \
               --just_for_test True
```

### 5. Zero-Shot Cross-Dataset Evaluation (UDIAT)

Directly evaluate the BUSI-trained model on the completely unseen UDIAT dataset:

```bash
conda activate hddinet
python main.py --gpu 0 --batch_size 8 \
               --model HDDI_Net \
               --base_dir ./data/busi \
               --dataset_name busi \
               --zero_shot_base_dir ./data/udiat \
               --zero_shot_dataset_name udiat \
               --just_for_zero_shot
```

Or use the provided script:

```bash
run_udiat_verification.bat
```

---

## 🏅 Experiments

### Internal Validation — BUSI Dataset

| Method        | Params (M) ↓ | FLOPs (G) ↓ | Val IoU ↑  | Val F1 ↑   |
|:--------------|:------------:|:-----------:|:----------:|:----------:|
| TransUnet     | 93.23        | 24.67       | **0.6831** | 0.9317     |
| SwinUnet      | 41.34        | 8.69        | 0.6519     | 0.8759     |
| AttU_Net      | 34.88        | 66.63       | 0.6406     | 0.7452     |
| U_Net         | 34.53        | 65.52       | 0.6428     | 0.7277     |
| EMCAD         | 26.76        | 5.60        | 0.6393     | 0.8214     |
| **HDDI-Net (Ours)** | **1.71** | **3.99** | **0.6709** | **0.8774** |
| Tinyunet      | 0.48         | 1.66        | 0.5624     | 0.6546     |

*HDDI-Net is the most efficient model that outperforms all lightweight/medium models while closing the gap with heavy models like TransUnet by only 1.2% IoU.*

### Zero-Shot Generalization — UDIAT Dataset (Unseen)

| Method        | Params (M) ↓ | FLOPs (G) ↓ | Zero-Shot IoU ↑ | Zero-Shot F1 ↑ |
|:--------------|:------------:|:-----------:|:---------------:|:--------------:|
| TransUnet     | 93.23        | 24.67       | **0.7868**      | 0.8764         |
| **HDDI-Net (Ours)** | **1.71** | **3.99** | **0.7504**   | **0.8309**     |
| EMCAD         | 26.76        | 5.60        | 0.7248          | 0.8217         |
| SwinUnet      | 41.34        | 8.69        | 0.7210          | 0.8224         |
| AttU_Net      | 34.88        | 66.63       | 0.7046          | 0.8130         |
| U_Net         | 34.53        | 65.52       | 0.7040          | 0.8127         |
| Tinyunet      | 0.48         | 1.66        | 0.6664          | 0.7931         |

*With only 1.71M parameters, HDDI-Net surpasses SwinUnet (41M) and EMCAD (26M) in zero-shot generalization, demonstrating the superiority of the Dual-Domain design.*

### Ablation Study — BUSI Dataset

| Model Variant                         | IoU ↑  | F1 ↑   | ΔIoU  |
|:--------------------------------------|:------:|:------:|:-----:|
| **Full Model (HDDI-Net)**             | **0.6709** | **0.8774** | —  |
| w/o Deep Supervision (No DS)          | 0.6421 | 0.7962 | ↓2.8% |
| w/o Frequency Gating (No Freq)        | 0.6370 | 0.8074 | ↓3.3% |
| w/o Attention Gate (No AG)            | 0.6267 | 0.7970 | ↓4.3% |

---

## 📑 Citation

If you find our work useful for your research, please cite:

```bibtex
@misc{hddinet2026,
      title={Lightweight yet Robust: Frequency-Gated Interaction Network for Generalized Ultrasound Lesion Segmentation},
      author={Author Names},
      year={2026},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

---

## 📝 Related Projects

- [U-Bench](https://github.com/FengheTan9/U-Bench): A Comprehensive Understanding of U-Net through 100-Variant Benchmarking
- [TransUnet](https://github.com/Beckschen/TransUNet): Transformers Make Strong Encoders for Medical Image Segmentation
- [SwinUnet](https://github.com/HuangBo-Terraman/SwinE-Net): Swin Transformers for Medical Image Segmentation
- [AttU_Net](https://github.com/ozan-oktay/Attention-Gated-Networks): Attention U-Net: Learning Where to Look for the Pancreas

---

## 📬 Contact

For questions or collaborations, feel free to open a GitHub Issue.

⭐ Star this repo if you find it useful!
