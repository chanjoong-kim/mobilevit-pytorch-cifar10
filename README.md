# MobileViT-PyTorch-CIFAR10

A PyTorch implementation of ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/abs/2110.02178) (arXiv, 2021),
with custom modifications and trained on the CIFAR-10 dataset. This implementation is optimized for execution in the Google Colab environment.

## Implementation Details
### This repository provides:
1. Complete Training Pipeline: From data loading to model evaluation
2.  Custom MobileViT Blocks: Implementation of key components:
    * ConvNormAct: Streamlined convolution blocks with SiLU activation
    * InvertedResidual: MobileNetV2-style blocks for efficient feature extraction
    * MobileViTBlock: Core transformer-based processing unit with local-global fusion
3. Model Size Variants: Configurable model dimensions to balance size and performance
    * **Model Variants via dims Parameter:**  
      This implementation supports three variants of MobileViT that can be selected by specifying the `dims` parameter in the `MobileViT` class. The `dims` parameter controls the channel dimensions in MobileViT blocks and allows you to choose a model size that best fits your needs:
        - **MobileViT-XXS:** `dims=[32, 64, 80]`
        - **MobileViT-XS:** `dims=[64, 80, 96]`
        - **MobileViT-S:** `dims=[96, 120, 144]`


## Training Configuration
### Our implementation uses:

* Optimizer: AdamW with weight decay of 0.01
* Learning Rate: Initial lr of 0.001 with cosine annealing
* Data Augmentation: Random crop, horizontal flip, and normalization
* Input Size: 224×224 (resized from CIFAR-10's 32×32)
* Batch Size: 64 images

## Results on CIFAR-10
### Below are the results from a sample training run. Note that due to random initialization and data augmentation, your results may vary slightly:

* Overall Accuracy: 85.12% after 10 epochs
* Model Size: 5,327,610 parameters

## Experiment
![Image](https://github.com/user-attachments/assets/e37f38b7-7f72-42c6-8ed3-cf8540b0e461)

### Class-wise Accuracy
|Class|Accuracy|
|---|-----|
|car|94.20%|
|truck|92.20%|
|ship|91.70%|
|frog|91.40%|
|deer|86.20%|
|plane|85.40%|
|horse|85.30%|
|dog|77.80%|
|bird|77.00%|
|cat|70.00%|

## Model Size Comparison
|Model|Parameters|
|----------|----------|
|MobileNetV2|2,236,682|
|MobileViT-XXS|5,327,610|
|MobileViT-XS|8,674,362|
|ResNet-18|11,181,642|
|MobileViT-S|19,678,954|

## Citation
``` bibtex
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```
