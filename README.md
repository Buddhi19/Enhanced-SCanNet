# Enhanced SCanNet: CBAM and Dice Loss for Semantic Change Detection

This repository contains the official PyTorch implementation of **Enhanced SCanNet**, an improved Semantic Change Detection (SCD) framework for multi-temporal remote sensing imagery. The proposed method enhances the original **SCanNet** architecture by integrating the **Convolutional Block Attention Module (CBAM)** into the decoder and incorporating **Dice Loss** to better address class imbalance.

The work is based on the paper:

> **Enhanced SCanNet with CBAM and Dice Loss for Semantic Change Detection**
> R.M.A.M.B. Ratnayake et al., University of Peradeniya

---

## 🔍 Overview

Semantic Change Detection (SCD) aims to identify *what* has changed and *how* land-cover semantics evolve between two temporal images. Despite recent advances, SCD models still face challenges such as:

* Noisy and misaligned remote sensing imagery
* Subtle class boundaries
* Severe class imbalance between changed and unchanged regions

To address these, this work proposes:

* **CBAM-enhanced decoder blocks** for refined spatial and channel-wise feature attention
* **Dice Loss** to improve sensitivity to minority (change) classes

The resulting architecture achieves consistent performance improvements on the **SECOND dataset**, both quantitatively and qualitatively.

---

## 🧠 Methodology

### 1. Semantic Change Network (SCanNet)

Enhanced SCanNet builds upon the original **SCanNet** framework, which employs a **Triple Encoder–Decoder (TED)** design:

* Two weight-sharing encoders extract semantic features from pre- and post-change images
* A third encoder focuses on change-specific information
* A transformer-based **SCanFormer** module enables cross-temporal attention to capture subtle changes

Additional semantic constraints are used during training:

* **Pseudo-label loss** for unchanged regions
* **Semantic consistency loss** to enforce temporal coherence

---

### 2. CBAM Integration

To improve feature representation, **CBAM** is integrated into each decoder block. CBAM sequentially applies:

#### Channel Attention (CA)

Highlights informative feature channels using global average and max pooling followed by a shared MLP.

#### Spatial Attention (SA)

Emphasizes important spatial regions using pooled channel descriptors and convolutional filtering.

#### Attention Fusion

Low-level encoder features and high-level decoder features are fused using CBAM-guided attention, enabling:

* Better preservation of fine spatial details
* Stronger alignment between semantic and spatial information

---

### 3. Loss Functions

The total loss is a weighted combination of:

* **Cross-Entropy Loss** for semantic supervision
* **Dice Loss** for changed regions (mitigates class imbalance)
* **Pseudo-label Loss** for unchanged areas
* **Semantic Consistency Loss** across temporal images

This formulation improves boundary delineation and robustness against skewed class distributions.

---

## 📊 Dataset

### SECOND Dataset

Experiments are conducted on the **SECOND** dataset, which contains:

* 4,662 bi-temporal image pairs (512×512)
* Spatial resolution: 0.5–3 m
* Six land-cover classes:

  * Non-vegetated ground
  * Trees
  * Low vegetation
  * Water
  * Buildings
  * Playgrounds

Following standard protocol:

* **3,729** image pairs for training
* **933** image pairs for testing

---

## ⚙️ Training Settings

* Framework: **PyTorch**
* Backbone: **ResNet-34** (ImageNet pretrained)
* Optimizer: **SGD with Nesterov momentum**
* Batch size: **6**
* Initial learning rate: **0.1**
* Learning rate schedule: Polynomial decay (50 epochs)

---

## 📈 Results

### Ablation Study (SECOND Dataset)

| Method                          | CBAM | Dice | OA (%)    | Fscd (%)  | mIoU (%)  | SeK (%)   |
| ------------------------------- | ---- | ---- | --------- | --------- | --------- | --------- |
| TED                             | ✗    | ✗    | 87.39     | 61.59     | 72.49     | 22.17     |
| SCanNet                         | ✗    | ✗    | 87.86     | 63.66     | 73.42     | 23.94     |
| SCanNet + CBAM                  | ✓    | ✗    | 87.98     | 63.81     | 73.51     | 24.11     |
| **Enhanced SCanNet (Proposed)** | ✓    | ✓    | **88.12** | **64.31** | **73.63** | **24.25** |

### Key Observations

* CBAM improves spatial precision and suppresses false positives
* Dice Loss significantly boosts performance on minority change classes
* The proposed model outperforms existing SOTA methods on all evaluation metrics

---

## 🖼️ Qualitative Results

Compared to the baseline SCanNet, Enhanced SCanNet:

* Recovers small and thin change regions more accurately
* Produces sharper segmentation boundaries
* Reduces spurious predictions caused by noise and shadows

---

## 📌 Citation

If you find this work useful, please consider citing:

```
@article{ratnayake2025enhancedscannet,
  title={Enhanced SCanNet with CBAM and Dice Loss for Semantic Change Detection},
  author={Ratnayake, R.M.A.M.B. and Wijenayake, W.M.B.S.K. and others},
  journal={arXiv preprint arXiv:2505.04199},
  year={2025}
}
```

---

## 🙏 Acknowledgements

* Original **SCanNet** implementation by Ding et al.
* **CBAM** module by Woo et al.
* SECOND dataset by Shi et al.

---

## 📬 Contact

For questions or discussions:

* **R.M.A.M.B. Ratnayake**
  Department of Electrical and Electronic Engineering
  University of Peradeniya
  📧 [athulya@eng.pdn.ac.lk](mailto:athulya@eng.pdn.ac.lk)
