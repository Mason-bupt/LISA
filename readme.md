# LISA

Official PyTorch implementation of **LISA: Language-guided Interference-aware Spatial-Frequency Attention for Driver Gaze Estimation**.

> Accepted by **IJCAI 2026**.

[[arXiv](https://arxiv.org/abs/2605.17287)] [[Code](https://github.com/Mason-bupt/LISA)]

## Overview

LISA is a lightweight driver gaze estimation framework that combines spatial attention, frequency-domain fusion, and language-guided disentanglement. The model is designed to improve robustness under common in-cabin interference such as occlusion, illumination variation, blur, facial appearance changes, and accessories.

The framework contains three main components:

- **Feature Alignment Module**: aligns shallow detail features and deeper semantic guidance features from a truncated ResNet-18 backbone.
- **FAM Fusion**: injects stable low-frequency semantic cues in the Fourier domain and applies spatial saliency gating to emphasize gaze-relevant regions.
- **SDM**: a training-time semantic disentanglement module that uses frozen CLIP text features to push gaze features away from gaze-irrelevant appearance attributes.

## Repository structure

```text
.
├── dataset.py                 # Dataset loading and preprocessing
├── data_normalization.py      # Gaze normalization utilities
├── train_by_subject.py        # Subject-independent training script
├── test_by_subject.py         # Evaluation script
└── models
    ├── resnet18_backbone.py   # Truncated ResNet-18 feature extractor
    ├── resnet_fam_lite.py     # LISA / ResNet-FAM-Lite model
    └── sdm.py                 # Semantic Disentanglement Module
```

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n lisa python=3.10 -y
conda activate lisa

pip install torch torchvision timm transformers opencv-python pillow numpy
```

If you use CUDA, please install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Data preparation

Organize the dataset root with subject folders following the format below:

```text
DATA_ROOT/
├── Subject1_xxx_data/
│   ├── face_ims/
│   │   └── 000001_face.png
│   └── gaze_info/
│       └── 000001_gaze.txt
├── Subject2_xxx_data/
│   ├── face_ims/
│   └── gaze_info/
└── ...
```

Each gaze label file should contain the required eye location and gaze direction fields, such as:

```text
Right_3D_Eye_Loc: [...]
Right_Gaze_Dir: [...]
Left_3D_Eye_Loc: [...]
Left_Gaze_Dir: [...]
```

The dataset loader automatically normalizes face images and converts gaze vectors to pitch-yaw angles.

## Training

Run subject-independent training:

```bash
python train_by_subject.py \
  --data_root /path/to/DATA_ROOT \
  --train_subjects 1-22 \
  --val_subjects 23-25 \
  --test_subjects 26-28 \
  --batch_size 64 \
  --epochs 50 \
  --device cuda \
  --amp
```

The script saves:

- the best checkpoint to `checkpoints/best.pth`;
- the test split indices to `test_indices_by_subject.json`;
- training logs and history files to `logs/`.

By default, SDM is enabled. To train without SDM:

```bash
python train_by_subject.py \
  --data_root /path/to/DATA_ROOT \
  --no-use-sdm
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python test_by_subject.py \
  --data_root /path/to/DATA_ROOT \
  --checkpoint checkpoints/best.pth \
  --test_indices_file test_indices_by_subject.json \
  --batch_size 128 \
  --device cuda
```

Alternatively, specify a test subject range directly:

```bash
python test_by_subject.py \
  --data_root /path/to/DATA_ROOT \
  --checkpoint checkpoints/best.pth \
  --test_subjects 26-28
```

## Model notes

- The visual backbone uses a truncated ResNet-18 feature extractor.
- SDM uses frozen CLIP text features as auxiliary supervision during training.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{ma2026lisa,
  title={LISA: Language-guided Interference-aware Spatial-Frequency Attention for Driver Gaze Estimation},
  author={Ma, Jun and Yang, Zhenye and Zhou, Ruichen and Zhang, Pei and Li, Huan and Chen, Jinpeng},
  journal={arXiv preprint arXiv:2605.17287},
  year={2026}
}
```

## Acknowledgements

This project uses PyTorch, timm, Hugging Face Transformers, and CLIP pretrained representations.
