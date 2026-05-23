# Flower-Resnet: A flower image recognition network based on the ResNet architecture
This repository provides the official implementation of a floral image classification system based on the Deep Residual Network (ResNet) architecture. The project focuses on effectively extracting high-dimensional visual features from fine-grained natural images, overcoming the degradation problem inherent in deep convolutional networks.
## Overview
Fine-grained image classification in natural environments presents significant challenges due to high intra-class variance and low inter-class variance among different flower species. 

This project leverages the ResNet architecture, utilizing its core identity shortcut connections (residual blocks) to train a deep convolutional neural network efficiently. By incorporating transfer learning from pre-trained weights on ImageNet, the model achieves rapid convergence and high classification accuracy on the target floral dataset, demonstrating strong robustness against complex background noise and scale variations.


## Datasets preparation
Flower Image Classification Dataset includes images of 14 types of flowers, consisting of 13,618 training images and 98 validation images. Each image has a size of 256*256, and the total dataset size is 202MB. It supports the recognition of fourteen types of flowers: carnation, iris, hyacinth, daffodil, rose, exiled nephew, tulip, marigold, dandelion, chrysanthemum, black-eyed daisy, water lily, sunflower, and daisy.


