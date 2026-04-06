# PANav: Robust Embodied Navigation under Sensor Degradation

**Cross-Domain Physical Perception Distillation for Zero-Shot Navigation**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Habitat-Sim](https://img.shields.io/badge/Habitat--Sim-0.2.4-green.svg)](https://github.com/facebookresearch/habitat-sim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

PANav++ addresses a critical yet overlooked problem: **zero-shot image-goal navigation collapses under real-world sensor degradation** (noise, blur, low light, fog, etc.). We diagnose that the root cause is *retrieval collapse*—CLIP features become non-discriminative under degradation, causing the agent to lose track of its goal.

Our solution transfers degradation-invariant **pseudo-infrared (IR) physical priors** from aerial RGB-IR data to indoor navigation via a lightweight Teacher-Student distillation framework with degradation-augmented training.

### Key Results

- **Retrieval**: 0 defeats across all 52 degraded conditions on MP3D; 1.74x CLIP's S4 accuracy
- **Navigation**: Positive SR improvement across all 5 severity levels on both HM3D and MP3D
- **Efficiency**: Only 17.67M additional parameters (~20% of CLIP ViT-B/32), 14 min training on a single RTX 4090

<p align="center">
  <img src="assets/framework.png" width="90%" alt="PANav Framework"/>
</p>

## Architecture


## Installation

### Prerequisites

- Python 3.9
- CUDA 12.1+ (for GPU training)
- Conda (for Habitat-Sim installation)

### Quick Setup

```bash
# 1. Clone repository
cd PANav_plus_plus

# 2. Run automated setup
bash setup_env.sh

# 3. Or manual setup
conda create -n panav python=3.9 -y
conda activate panav
conda install -c aihabitat -c conda-forge habitat-sim=0.2.4 headless --no-update-deps -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Dataset Preparation

```bash
# HM3D scenes (requires Matterport agreement)
python -m habitat_sim.utils.datasets_download --uids hm3d_minival_v0.2 --data-path ./data

# AVIID dataset (for Teacher pretraining)
# Download from: https://github.com/zonghaohan/USTNet
# Place under: ./data/AVIID/
```

## Project Structure

```
PANav_plus_plus/
├── models/                     # Core network modules
│   ├── vir_generator.py        # VIRGenerator: RGB → Pseudo-IR (U-Net + Transformer)
│   ├── multi_degradation.py    # 13-type degradation simulation engine
│   ├── enhanced_distillation.py # Multi-scale & uncertainty-weighted distillation
│   ├── fusion_encoder.py       # DTAF-Lite: RGB + Pseudo-IR dual-stream fusion
│   ├── clip_adapter.py         # Projection to CLIP embedding space
│   ├── noise_aware_fusion.py   # Learnable noise-aware fusion module
│   ├── goal_encoder.py         # Unified text/image goal encoding via CLIP
│   ├── nav_policy.py           # Goal-conditioned navigation policy (MLP/GRU)
│   ├── frontier_nav.py         # PANav++ perception + frontier navigation
│   ├── semantic_map.py         # Top-down semantic map + frontier scoring
│   ├── hierarchical_policy.py  # Hierarchical explore/exploit policy
│   ├── depth_estimator.py      # Depth estimation module
│   ├── panav.py                # Full PANav model pipeline
│   └── panav_plus.py           # PANav++ with semantic map & hierarchical policy
├── losses/
│   └── panav_losses.py         # Perception + PPO training losses
├── envs/
│   ├── nav_env.py              # Abstract navigation environment API
│   └── habitat_wrapper.py      # Habitat-Sim environment wrapper
├── utils/
│   ├── checkpoint.py           # Checkpoint loading utilities
│   └── nav_metrics.py          # Navigation metrics (SR, SPL, DTS)
├── data/
│   └── perception_dataset.py   # AVIID/DroneVehicle paired dataset loader
├── configs/
│   └── default.yaml            # Default hyperparameters
├── scripts/                    # Training, evaluation & demo scripts
│   ├── train_perception.py     # Stage 1: VIR + Fusion perception pretraining
│   ├── train_nav.py            # Stage 2: Navigation policy training (PPO)
│   ├── train_habitat.py        # End-to-end Habitat ObjectNav training
│   ├── train_full_gpu.py       # Full GPU training on RTX 4090
│   ├── eval.py                 # General evaluation
│   ├── eval_habitat.py         # Habitat-Sim evaluation (SR/SPL/DTS)
│   ├── exp_panav_distill.py    # Teacher-Student distillation pipeline
│   └── demo_perception.py      # Perception pipeline demo
├── tools/                      # Visualization tools
│   ├── process_robot_video.py  # Real robot video processing
│   ├── gen_framework_fig.py    # Framework diagram generator
│   └── gen_nav_process_fig.py  # Navigation process figure generator
├── requirements.txt
├── setup_env.sh
└── README.md
```

## Usage

### 1. Teacher Pretraining (Aerial RGB → IR)

```bash
cd scripts
python train_perception.py \
    --stage teacher \
    --data_root ../data/AVIID \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-4
```

### 2. Student Distillation (with Degradation Augmentation)

```bash
python train_perception.py \
    --stage student \
    --teacher_ckpt ../checkpoints/teacher_best.pth \
    --epochs_stage1 30 \
    --epochs_stage2 20 \
    --degrade_prob 0.5
```

### 3. Navigation Evaluation

```bash
# Evaluate on Habitat (HM3D / MP3D)
python eval_habitat.py \
    --ckpt ../checkpoints/panav_plus.pth \
    --scenes 10
```

### 4. Demo: Perception Only

```bash
python demo_perception.py --input ../demo_images/ --ckpt ../checkpoints/panav_plus.pth
```

## Degradation Benchmark

We provide a systematic benchmark covering **13 degradation types × 4 severity levels** across MP3D and HM3D (~28,900 evaluation episodes).

| Category | Degradation Types |
|----------|------------------|
| **Sensor** | Gaussian Noise, Salt & Pepper, Motion Blur, Low Light, Fog, JPEG Compression |
| **Color** | Color Jitter, Red/Blue/Green/Warm/Cool Filters |
| **Combined** | Multi-type composite |

### Key Findings

1. **Retrieval follows stretched-exponential decay** (R² > 0.999)
2. **Navigation is 1.6× more robust than retrieval** (NAR ≈ 1.6)
3. **Sensor degradation is 1.5× more damaging than color shifts**
4. **All pure-visual methods collapse at S4**


## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for the simulation environment
- [CLIP](https://github.com/openai/CLIP) for vision-language features
- [AVIID](https://github.com/zonghaohan/USTNet) for aerial RGB-IR training data
- [HM3D](https://aihabitat.org/datasets/hm3d/) and [MP3D](https://niessner.github.io/Matterport/) for indoor scene datasets
