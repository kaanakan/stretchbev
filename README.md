# StretchBEV: Stretching Future Instance Prediction Spatially and Temporally (ECCV 2022)

<!-- [![report](https://img.shields.io/badge/CVF-paper-orange)](https://openaccess.thecvf.com/content/ICCV2021/html/Akan_SLAMP_Stochastic_Latent_Appearance_and_Motion_Prediction_ICCV_2021_paper.html) -->
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2203.13641)
[![report](https://img.shields.io/badge/Project-Page-blue)](https://kuis-ai.github.io/stretchbev/)
[![report](https://img.shields.io/badge/Pretrained-Models-yellow)](https://github.com/kaanakan/stretchbev/releases/tag/v1.0)
[![report](https://img.shields.io/badge/Presentation-Video-brightgreen)](https://www.youtube.com/watch?v=2SiUNs6BMVk)
<!-- [![report](https://img.shields.io/badge/Supplementary-Material-brightgreen)](https://kuis-ai.github.io/stretchbev/data/docs/StretchBEV_supp.pdf) -->


> [**StretchBEV: Stretching Future Instance Prediction Spatially and Temporally**](https://arxiv.org/abs/2203.13641),            
> [Adil Kaan Akan](https://kaanakan.github.io), 
> [Fatma Guney](https://mysite.ku.edu.tr/fguney/),      
> *European Conference on Computer Vision (ECCV), 2022* 

<p float="center">
  <img src="https://kuis-ai.github.io/stretchbev/data/figures/comp/2sec_gifs/test_4042_2sec.gif" width="100%" />
</p>

## Features

StretchBEV is a future instance prediction network in Bird's-eye view representation. It earns temporal dynamics in a latent space through stochastic residual updates at each time step. By sampling from a learned distribution at each time step, we obtain more diverse future predictions that are also more accurate compared to previous work, especially stretching both spatially further regions in the scene and temporally over longer time horizons


## Requirements

All models were trained with Python 3.7.10 and PyTorch 1.7.0

A list of required Python packages is available in the `environment.yml` file.



## Datasets

For preparations of datasets, we followed [FIERY](https://github.com/wayveai/fiery). Please follow [this link](https://github.com/wayveai/fiery/blob/master/DATASET.md) below if you want to construct the datasets.


## Training

To train the model on NuScenes:

- First, you need to download [`static_lift_splat_setting.ckpt`](https://github.com/wayveai/fiery/releases/download/v1.0/static_lift_splat_setting.ckpt) and copy it to this directory.
- Run `python train.py --config fiery/configs/baseline.yml DATASET.DATAROOT ${NUSCENES_DATAROOT}`.

This will train the model on 4 GPUs, each with a batch of size 2. To train on single GPU add the flag `GPUS 1`, and to change the batch size use the flag `BATCHSIZE ${DESIRED_BATCHSIZE}`.


## Evaluation

To evaluate a trained model on NuScenes:

- Download [pre-trained weights](https://github.com/wayveai/fiery/releases/download/v1.0/stretchbev.ckpt).
- Run `python evaluate.py --checkpoint ${CHECKPOINT_PATH} --dataroot ${NUSCENES_DATAROOT}`.

### Pretrained weights

You can download the pretrained weights from the releases of this repository or the links below.

[Normal setting weight](https://github.com/wayveai/fiery/releases/download/v1.0/stretchbev.ckpt)

[Fishing setting weight](https://github.com/wayveai/fiery/releases/download/v1.0/stretchbev_fishing.ckpt)



## How to Cite

Please cite the paper if you benefit from our paper or the repository:

```
@InProceedings{Akan2022ECCV,
            author    = {Akan, Adil Kaan and G\"uney, Fatma},
            title     = {StretchBEV: Stretching Future Instance Prediction Spatially and Temporally},
            journal = {European Conference on Computer Vision (ECCV)},
            year      = {2022},
            }
```

## Acknowledgments

We would like to thank FIERY and SRVP authors for making their repositories public. This repository contains several code segments from [FIERY's repository](https://github.com/wayveai/fiery) and [SRVP's repository](https://github.com/edouardelasalles/srvp). We appreciate the efforts by [Berkay Ugur Senocak](https://github.com/4turkuaz) for cleaning the code before release.
