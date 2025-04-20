# Enhanced‑SCanNet 🔍 🌍

 **Semantic Change Detection with CBAM & Composite Loss**  
A lightweight upgrade of [SCanNet](https://github.com/DingLei14/SCanNet) that fuses **Convolutional Block Attention Modules (CBAM)** into each decoder stage and trains with a **composite loss** (Cross‑Entropy + Dice + Lovász‑Softmax) to sharpen boundaries and fight class imbalance.

## 🚀 Quick Start

```bash
# clone & install
git clone https://github.com/Buddhi19/SCanNet.git
cd SCanNet
pip install -r requirements.txt

# training
python SCD_train.py

```

## ❤️ Acknowledgements

* **Original SCanNet** – massive thanks to Lei Ding *et al.* (🔗 <https://github.com/DingLei14/SCanNet>) for releasing the baseline code and datasets.  
* **CBAM** implementation adapted from Woo *et al.*, “Convolutional Block Attention Module,” ECCV 2018.  
