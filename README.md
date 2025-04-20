# Enhanced‑SCanNet 🔍 🌍

 **Semantic Change Detection with CBAM & Composite Loss**  
A lightweight upgrade of [SCanNet](https://github.com/DingLei14/SCanNet) that fuses **Convolutional Block Attention Modules (CBAM)** into each decoder stage and trains with a **composite loss** (Cross‑Entropy + Dice + Lovász‑Softmax) to sharpen boundaries and fight class imbalance.

### 🎯 Loss Function

$
L_\text{total}= 
\;\alpha\!\bigl(L^{\text{sem}}_{\text{CE}}+\lambda_1L^{\text{sem}}_{\text{Dice}}+\lambda_2L^{\text{sem}}_{\text{Lovász}}\bigr)\\
+&\;\beta\!\bigl(L^{\text{psd}}_{\text{CE}}+\lambda_3L^{\text{psd}}_{\text{Dice}}+\lambda_4L^{\text{psd}}_{\text{Lovász}}\bigr)\\
+&\;\gamma\,L_{\text{consistency}}
$

## 🚀 Quick Start

```bash
# clone & install
git clone https://github.com/Buddhi19/SCanNet.git
cd SCanNet
pip install -r requirements.txt

# training
python SCD_train.py

```

## 📊 Benchmark Highlights

| Dataset      | Metric | SCanNet [1] | **Enhanced** |
|--------------|--------|-------------|--------------|
| SECOND       | OA     | 87.05 | **87.80** |
|              | F<sub>scd</sub> | 62.09 | **63.33** |
| Landsat‑SCD  | OA     | 96.26 | **97.21** |
|              | mIoU   | 88.96 | **89.73** |

See `/docs/results/` for full tables & plots.

## ❤️ Acknowledgements

* **Original SCanNet** – massive thanks to Lei Ding *et al.* (🔗 <https://github.com/DingLei14/SCanNet>) for releasing the baseline code and datasets.  
* **CBAM** implementation adapted from Woo *et al.*, “Convolutional Block Attention Module,” ECCV 2018.  


---
