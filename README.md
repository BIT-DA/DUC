<div align="center">  
  
**Dirichlet-based Uncertainty Calibration for Active Domain Adaptation [ICLR 2023 Spotlight]**

*by [Mixue Xie](https://scholar.google.com/citations?user=2NHj3GsAAAAJ&hl=zh-CN&oi=ao), [Shuang Li](https://shuangli.xyz), Rui Zhang and [Chi Harold Liu](https://scholar.google.com/citations?user=3IgFTEkAAAAJ&hl=en)*

[![arXiv](https://img.shields.io/badge/Paper-arXiv-%23B31B1B?style=flat-square)](https://arxiv.org/abs/2302.13824)&nbsp;&nbsp;


</div>
<!-- TOC -->

- [Overview](#overview)
- [Prerequisites Installation](#prerequisites-installation)
- [Datasets Preparation](#datasets-preparation)
- [Code Running](#code-running)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Contact](#contact)

<!-- /TOC -->

## Overview

We propose a Dirichlet-based Uncertainty Calibration (DUC) approach for active domain adaptation (DA). It provides a novel perspective for active DA by introducing the Dirichlet-based evidential model and designing an uncertainty origin-aware selection strategy to comprehensively evaluate the value of samples.

![image](Figures/Fig_method.jpg)

## Prerequisites Installation

* For cross-domain *image classification* tasks, this code is implemented with `Python 3.7.5`, `CUDA 11.4` on `NVIDIA GeForce RTX 2080 Ti`. To try out this project, it is recommended to set up a virtual environment first.

    ```bash
    # Step-by-step installation
    conda create --name DUC_cls python=3.7.5
    conda activate DUC_cls

    # this installs the right pip and dependencies for the fresh python
    conda install -y ipython pip

    # this installs required packages
    pip install -r requirements_cls.txt
    ```

* For cross-domain *semantic segmentation* tasks, this code is implemented with `Python 3.7.5`, `CUDA 11.2` on `NVIDIA GeForce RTX 3090`. To try out this project, it is recommended to set up a virtual environment first.

    ```bash
    # Step-by-step installation
    conda create --name DUC_seg python=3.7.5
    conda activate DUC_seg

    # this installs the right pip and dependencies for the fresh python
    conda install -y ipython pip

    # this installs required packages
    python3 -m pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements_seg.txt
    ```


## Datasets Preparation

* **Image Classification Datasets**

    - Download [Office-Home Dataset](http://hemanthdv.org/OfficeHome-Dataset/)
    - Download [VisDA-2017 Dataset](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
    - Download [DomainNet dataset](http://ai.bu.edu/M3SDA/#dataset) and [miniDomainNet's split files](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#miniDomainNet)

    Symlink the required dataset by running

    ```bash
    ln -s /path_to_home_dataset data/home
    ln -s /path_to_visda2017_dataset data/visda2017
    ln -s /path_to_domainnet_dataset data/domainnet
    ```

    The data folder should be structured as follows:
    ```
    ├── data/
    │   ├── home/     
    |   |   ├── Art/
    |   |   ├── Clipart/
    |   |   ├── Product/
    |   |   ├── RealWorld/
    │   ├── visda2017/
    |   |   ├── train/
    |   |   ├── validation/
    │   ├── domainnet/	
    |   |   ├── clipart/
    |   |   |—— infograph/
    |   |   ├── painting/
    |   |   |—— quickdraw/
    |   |   ├── real/	
    |   |   ├── sketch/	
    |   |——
    ```

* **Semantic Segmentation Datasets**

    - Download [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
    - Download [GTAV Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
    - Download [SYNTHIA Dataset](https://synthia-dataset.net/)

    Symlink the required dataset by running

    ```bash
    ln -s /path_to_cityscapes_dataset datasets/cityscapes
    ln -s /path_to_gtav_dataset datasets/gtav
    ln -s /path_to_synthia_dataset datasets/synthia
    ```

    Generate the label static files for GTAV/SYNTHIA Datasets by running

    ```bash
    python datasets/generate_gtav_label_info.py -d datasets/gtav -o datasets/gtav/
    python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
    ```

    **The data folder should be structured as follows:**

    ```
    ├── datasets/
    │   ├── cityscapes/     
    |   |   ├── gtFine/
    |   |   ├── leftImg8bit/
    │   ├── gtav/
    |   |   ├── images/
    |   |   ├── labels/
    |   |   ├── gtav_label_info.p
    │   └──	synthia
    |   |   ├── RAND_CITYSCAPES/
    |   |   ├── synthia_label_info.p
    │   └──	
    ```

## Code Running

* for cross-domain image classification:

    ```bash
    # running for cross-domain image classification:
    sh script_cls.sh
    ```

* for cross-domain semantic segmentation:
    ```bash
    # go to the directory of semantic segmentation tasks
    cd ./segmentation

    # running for GTAV to Cityscapes
    sh script_seg_gtav.sh

    # running for SYNTHIA to Cityscapes
    sh script_seg_syn.sh
    ```

## Acknowledgments

This project is based on the following open-source projects. We thank their authors for making the source code publicly available.

- [Transferable-Query-Selection](https://github.com/thuml/Transferable-Query-Selection)
- [EADA](https://github.com/BIT-DA/EADA)
- [RIPU](https://github.com/BIT-DA/RIPU)


## Citation

If you find this work helpful to your research, please consider citing the paper:

```bibtex
@inproceedings{xie2023DUC,
  title={Dirichlet-based Uncertainty Calibration for Active Domain Adaptation},
  author={Xie, Mixue and Li, Shuang and Zhang, Rui and Liu, Chi Harold},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## Contact
If you have any problem about our code, feel free to contact mxxie@bit.edu.cn or describe your problem in Issues.

