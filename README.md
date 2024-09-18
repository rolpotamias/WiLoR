<div align="center">

# WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild

[Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>1</sup> &emsp; [Jinglei Zhang]()<sup>2</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>1</sup> &emsp; [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)<sup>1</sup>  

<sup>1</sup>Imperial College London, UK <br>
<sup>2</sup>Shanghai Jiao Tong University, China

<a href='https://rolpotamias.github.io/WiLoR/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/spaces/rolpotamias/WiLoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
<a href='https://colab.research.google.com/'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>
</div>

This is the official implementation of **[WiLoR](https://arc2face.github.io/)**, an state-of-the-art hand localization and reconstruction model:

![teaser](assets/teaser.png)

## Installation
```
git clone --recursive https://github.com/rolpotamias/WiLoR.git
cd WiLoR
```

The code has been tested with PyTorch 2.0.0 and CUDA 11.7. It is suggested to use an anaconda encironment to install the the required dependencies:
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
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/mano/` folder. 
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
The dataset will be released soon. 

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## License 
WiLoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find WiLoR useful for your research, please consider citing our paper:

```bibtex
@article{potamias2024wilor,
      title={WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild},
      author={Potamias, Rolandos Alexandros and Zhang, Jinglei and Deng, Jiankang and Zafeiriou, Stefanos},
      journal={arXiv},
      year={2024}
    }
```
