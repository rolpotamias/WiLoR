<div align="center">

# WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild

[Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>1</sup> &emsp; [Jinglei Zhang]()<sup>2</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>1</sup> &emsp; [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)<sup>1</sup>  

<sup>1</sup>Imperial College London, UK <br>
<sup>2</sup>Shanghai Jiao Tong University, China

<font color="blue"><strong>CVPR 2025</strong></font> 

<a href='https://rolpotamias.github.io/WiLoR/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2409.12259'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/spaces/rolpotamias/WiLoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
<a href='https://colab.research.google.com/drive/1bNnYFECmJbbvCNZAKtQcxJGxf0DZppsB?usp=sharing'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>
</div>

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wilor-end-to-end-3d-hand-localization-and/3d-hand-pose-estimation-on-freihand)](https://paperswithcode.com/sota/3d-hand-pose-estimation-on-freihand?p=wilor-end-to-end-3d-hand-localization-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wilor-end-to-end-3d-hand-localization-and/3d-hand-pose-estimation-on-ho-3d)](https://paperswithcode.com/sota/3d-hand-pose-estimation-on-ho-3d?p=wilor-end-to-end-3d-hand-localization-and)

</div>

This is the official implementation of **[WiLoR](https://rolpotamias.github.io/WiLoR/)**, an state-of-the-art hand localization and reconstruction model:

![teaser](assets/teaser.png)

## Installation
### [Update] Quick Installation
Thanks to [@warmshao](https://github.com/warmshao) WiLoR can now be installed using a single pip command:  
```
pip install git+https://github.com/warmshao/WiLoR-mini
```
Please head to [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) for additional details. 

**Note:** the above code is a simplified version of WiLoR and can be used for demo only. 
If you wish to use WiLoR for other tasks it is suggested to follow the original installation instructued bellow: 
### Original Installation
```
git clone --recursive https://github.com/rolpotamias/WiLoR.git
cd WiLoR
```

The code has been tested with PyTorch 2.0.0 and CUDA 11.7. It is suggested to use an anaconda environment to install the the required dependencies:
```bash
conda create --name wilor python=3.10
conda activate wilor

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
# Install requirements
pip install -r requirements.txt
```
Download the pretrained models using: 
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo
```bash
python demo.py --img_folder demo_img --out_folder demo_out --save_mesh 
```
## Start a local gradio demo
You can start a local demo for inference by running:
```bash
python gradio_demo.py
```
## WHIM Dataset
To download WHIM dataset please follow the instructions [here](./whim/Dataset_instructions.md)

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## License 
WiLoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find WiLoR useful for your research, please consider citing our paper:

```bibtex
@misc{potamias2024wilor,
    title={WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
    author={Rolandos Alexandros Potamias and Jinglei Zhang and Jiankang Deng and Stefanos Zafeiriou},
    year={2024},
    eprint={2409.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
