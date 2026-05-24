# Flower-ResNet: A flower image recognition network based on the ResNet architecture
This repository provides the official implementation of a floral image classification system based on the Deep Residual Network (ResNet) architecture. The project focuses on effectively extracting high-dimensional visual features from fine-grained natural images, overcoming the degradation problem inherent in deep convolutional networks.

## Overview
Fine-grained image classification in natural environments presents significant challenges due to high intra-class variance and low inter-class variance among different flower species. 

This project leverages the ResNet architecture, utilizing its core identity shortcut connections (residual blocks) to train a deep convolutional neural network efficiently. By incorporating transfer learning from pre-trained weights on ImageNet, the model achieves rapid convergence and high classification accuracy on the target floral dataset, demonstrating strong robustness against complex background noise and scale variations.

This repository implements a high-precision, end-to-end flower recognition system (14 classes) based on the PyTorch framework. Beyond achieving high validation accuracy, this project specifically investigates the critical trade-offs between **network depth, data augmentation, and model robustness** in open-world scenarios. 

## 🌟 Key Features

- **End-to-End Pipeline**: Modular code for custom dataset loading, training (`train.py`), inference (`predict.py`), and evaluation (`confusion_matrix.py`).
- **Robustness against Spurious Correlations**: Demonstrates how advanced data augmentation (`RandAugment`) forces neural networks to bypass color/background bias and focus on geometric morphological features (e.g., petal edges).
- **Model Calibration**: Implements Label Smoothing to mitigate the "overconfidence" issue commonly found in models trained with hard labels.
- **Visual Interpretability**: Integrates `pytorch-grad-cam` to generate high-resolution attention maps, providing visual evidence of the model's decision-making process and revealing visual blind spots.

## Datasets preparation
Flower Image Classification Dataset includes images of 14 types of flowers, consisting of 13,618 training images and 98 validation images. Each image has a size of 256*256, and the total dataset size is 202MB. It supports the recognition of fourteen types of flowers: carnation, iris, hyacinth, daffodil, rose, exiled nephew, tulip, marigold, dandelion, chrysanthemum, black-eyed daisy, water lily, sunflower, and daisy.

## 📁 Project Structure
```text
├── dataset/
│   └── flower/               # Dataset directory (10k+ images across 14 classes)
├── model/                    
│   ├── model_resnet18.py     # ResNet18 architecture definition
│   └── model_resnet34.py     # ResNet34 architecture with modified classification head
├── predict/
│   ├── predict_r18.py        # Single-image inference script for ResNet18
│   └── predict_r34.py        # Single-image inference script for ResNet34
├── cam_visualize/
│   ├── cam_visualize_R18.py  # Grad-CAM tool to analyze ResNet18 spatial attention
│   └── cam_visualize_R34.py  # Grad-CAM tool to analyze ResNet34 spatial attention
├── confusion_matrix/         # Directory for evaluation scripts and matrix outputs
├── train_resnet18.py         # Training script for ResNet18 (featuring RandAugment)
├── train_resnet34.py         # Training script for ResNet34 (Transfer Learning focused)
├── dataset.py                # Custom dataset loading and processing utility
├── utils.py                  # Auxiliary functions for training pipelines and metrics
└── requirements.txt          # Frozen environment dependencies
