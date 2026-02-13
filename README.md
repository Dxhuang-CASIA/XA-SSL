<div align="center">

<h1>Large-scale X-ray Angiogram Pre-training</h1>

<a href="https://arxiv.org/pdf/2602.11536"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://arxiv.org/pdf/2508.10794"><img src='https://img.shields.io/badge/AAAI_2026-Conference-blue' alt='Paper PDF'></a>
<a href="https://huggingface.co/datasets/waha2000huang/XA-170K"><img src='https://img.shields.io/badge/Dataset-XA--170K-green' alt='Dataset'></a>

<p>
    <b>Official implementation of ''<a href='https://arxiv.org/pdf/2602.11536'>Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis</a>''</b>
</p>

<p>
    <a href="https://dxhuang-casia.github.io/">De-Xing Huang</a><sup>1,2</sup>, 
    <a href="https://richardych.github.io/">Chaohui Yu</a><sup>3</sup>, 
    <a href="https://scholar.google.com/citations?user=e9_7q0gAAAAJ&hl">Xiao-Hu Zhou</a><sup>1,2</sup>, 
    <a href="https://garyxty.github.io/">Tian-Yu Xiang</a><sup>1,2</sup>, 
    <a href="https://scholar.google.com/citations?user=NePbSy4AAAAJ&hl">Qin-Yi Zhang</a><sup>1,2</sup>, 
    <a href="https://scholar.google.com/citations?user=_oQTTCAAAAAJ&hl">Mei-Jiang Gui</a><sup>1,2</sup>, 
    Rui-Ze Ma<sup>1</sup>, Chen-Yu Wang<sup>1</sup>, Nu-Fang Xiao<sup>1</sup>, 
    <a href="https://scholar.google.com/citations?user=WCRGTHsAAAAJ&hl">Fan Wang</a><sup>3</sup>, 
    and <a href="https://scholar.google.com/citations?user=tU0CwGwAAAAJ&hl">Zeng-Guang Hou</a><sup>1,2</sup>
</p>

<p>
    <sup>1</sup> Institute of Automation, Chinese Academy of Sciences<br>
    <sup>2</sup> University of Chinese Academy of Sciences<br>
    <sup>3</sup> DAMO Academy, Alibaba Group
</p>
<!-- <p>
    <sup>*</sup> Work done during De-Xing Huang's internship at DAMO Academy, Alibaba Group.
</p> -->

</div>

## 📖 TL;DR
This work introduces **VasoMIM**, a vascular anatomy-aware self-supervised learning framework designed specifically for X-ray angiogram pre-training. To support this, we curated **XA-170K**, the largest existing X-ray angiogram dataset. VasoMIM is validated on four downstream tasks crucial for X-ray angiogram analysis, demonstrating superior performance.

## 🛠️ Method
<div align="center">
  <img src="src/Fig_VasoMIM.png" width="100%" alt="VasoMIM Framework">
</div>

## ⚙️ Requirements
* This repository is a modification of the official <a href="https://github.com/facebookresearch/mae">MAE repository</a>. Installation and environment preparation steps follow the original repo.
* **Note on Timm:** This code relies on [`timm==1.0.20`](https://github.com/rwightman/pytorch-image-models).

## 💾 Datasets

### 1. Pre-training Dataset (XA-170K)

XA-170K aggregates data from four publicly available sources: CADICA, SYNTAX, XCAD, and CoronaryDominance. 

**Option A: Direct Download (Recommended)**
You can download the curated XA-170K dataset directly from our <a href="https://huggingface.co/datasets/waha2000huang/XA-170K">Hugging Face repo</a>.

**Option B: Manual Collection**
Alternatively, you can collect the raw data from the original sources:

| Dataset | Images | Link |
| :--- | :--- | :--- |
| CADICA | 6,594 | <a href="https://data.mendeley.com/datasets/p9bpx9ctcv/5">Download</a> |
| SYNTAX | 2,943 | <a href="https://figshare.com/articles/dataset/X-Ray_Angiography_Images_and_SYNTAX-Score_Dataset/25801447">Download</a> |
| XCAD | 1,621 | <a href="https://github.com/AISIGSJTU/SSVS">Download</a> |
| CoronaryDominance | 160,320 | <a href="https://huggingface.co/datasets/BearSubj13/CoronaryDominance">Download</a> |
| **Total** | **171,478** | - |

**Directory Structure**
The XA-170K dataset should be organized as follows:

```text
/path/to/XA-170K/
  ├── cadica/
  │    ├── image1.png
  │    └── ...
  ├── cadica_frangi/
  │    ├── image1.png
  │    └── ...
  ├── syntax/
  ├── syntax_frangi/
  ├── xcad/
  ├── xcad_frangi/
  ├── coronarydominance/
  └── coronarydominance_frangi/
```

### 2. Downstream Datasets

<div align="center"> <img src="src/Fig_Downstream.png" width="100%" alt="Downstream Tasks"> </div>

| Dataset | Train | Test | Link | Task |
| :--- | :--- | :--- | :--- | :--- |
| ARCADE-V | 1,000 | 3,00 | <a href="https://zenodo.org/records/10390295">Download</a> | Vessel Segmentation |
| CAXF | 337 | 201 | In-house* | Vessel Segmentation |
| XCAV | 175 | 46 | <a href="https://drive.google.com/file/d/11e5SmynT8qitWwSGBj5nn3JVYTNG5VZP/view">Download</a> | Vessel Segmentation |
| ARCADE-S | 1,000 | 3,00 | <a href="https://zenodo.org/records/10390295">Download</a> | Stenosis Segmentation |
| ARCADE-VS | 1,000 | 3,00 | <a href="https://zenodo.org/records/10390295">Download</a> | Vessel Segment Segmentation |
| Stenosis | 7,492 | 833 | <a href="https://data.mendeley.com/datasets/ydrm75xywg/1">Download</a> | Stenosis Detection |

*Note: Please contact De-Xing Huang (huangdexing2022@ia.ac.cn) if you wish to use CAXF for research purposes.

## 🚀 Pre-training

We pre-trained VasoMIM on 8 x NVIDIA H20 GPUs (96 GB).
```
cd /path/to/this/workspace
./pretrain_vasomim.sh
```

## 📦 Pre-trained Models

Coming Soon!
<!-- | Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VasoMIMv1-B  |    **M | Download  |
| VasoMIM-B  |    **M | Download  |
| VasoMIM-L  |    **M | Download  |
| VasoMIM-H  |    **M | Download  | -->

## 🙌 Acknowledgement
This project is built upon <a href="https://github.com/facebookresearch/mae">MAE</a> and <a href="https://github.com/Haochen-Wang409/HPM">HPM</a>. For segmentation task, our implementations are based on <a href="https://github.com/LeapLabTHU/CheXWorld">CheXWorld</a>. For detection task, we utilize <a href="https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet">ViTDet</a> from <a href="https://github.com/facebookresearch/detectron2">Detectron2</a>. The Frangi filter implementation is adapted from <a href="https://github.com/kirito878/denver">DeNVeR</a>.

We thank the authors of these repositories for their wonderful work.

## ✏️ Citation
If you find VasoMIM useful for your research, please consider citing our paper:
```bibtex
@inproceedings{huang2026vasomim,
  title={{VasoMIM}: Vascular anatomy-aware masked image modeling for vessel segmentation},
  author={Huang, De-Xing and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}

@article{huang2026vascular,
  title={Vascular anatomy-aware self-supervised pre-training for X-ray angiogram analysis},
  author={Huang, De-Xing and others},
  journal={arXiv preprint arXiv:2602.11536},
  year={2026}
}
```

If you utilize the pre-training dataset, please also consider citing the original data sources:
```bibtex
@article{jimenez2024cadica,
  title={CADICA: A new dataset for coronary artery disease detection by using invasive coronary angiography},
  author={Jim{\'e}nez-Partinen and others},
  journal={Expert Systems},
  volume={41},
  number={12},
  pages={e13708},
  year={2024}
}

@article{mahmoudi2025x,
  title={X-ray Coronary Angiogram images and {SYNTAX} score to develop Machine-Learning algorithms for {CHD} Diagnosis},
  author={Mahmoudi, Seyed Sajjad and others},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={471},
  year={2025}
}

@inproceedings{ma2021self,
  title={Self-supervised vessel segmentation via adversarial learning},
  author={Ma, Yuxin and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={7536--7545},
  year={2021}
}

@article{kruzhilov2025coronarydominance,
  title={{CoronaryDominance}: Angiogram dataset for coronary dominance classification},
  author={Kruzhilov, Ivan and others},
  journal={Scientific Data},
  volume={12},
  number={1},
  pages={341},
  year={2025}
}
```
