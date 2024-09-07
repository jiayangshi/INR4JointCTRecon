# Implicit Neural Representations for Robust Joint Sparse-View CT Reconstruction (TMLR)

This is the official repository for the paper *Implicit Neural Representations for Robust Joint Sparse-View CT Reconstruction*.

> [**Implicit Neural Representations for Robust Joint Sparse-View CT Reconstruction**](https://openreview.net/forum?id=XCzuQI0oXR),  
> Jiayang Shi, Junyi Zhu, Daniel M. Pelt, K. Joost Batenburg, Matthew B. Blaschko 
> *TMLR 2024*  

![](figures/inr4jointrecon.png)
We introduce a novel Bayesian framework for joint reconstruction of multiple objects from sparse-view CT scans using Implicit Neural Representations (INRs) to improve reconstruction quality. By capturing shared patterns across multiple objects with latent variables, our method enhances the reconstruction of each object, increases robustness to overfitting, and accelerates the learning process.

| ![Node 1](figures/node0.gif) | ![Node 5](figures/node4.gif) | ![Node 9](figures/node9.gif) | ![Prior](figures/latent.gif) |
|:----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|
|        **Node 1**            |        **Node 5**            |        **Node 9**            |        **Prior Mean**             |


## Setup
Create and activate a Conda environment:
```bash
conda env create -f environment.yml
conda activate inr4ct
```
## Dataset
We provide 10 slices LungCT images in [imgs.tif](imgs.tif) from [LungCT](http://medicaldecathlon.com) for easy testing. You can also use your own data by loading it accordingly.

## Run 
To run the joint reconstruction code, use:
```bash
python main.py
```

We also provide implementations of several comparison methods for joint reconstruction:
* Meta-Learning (MAML)
* Federated Averaging (FedAvg)
* INR-in-the-Wild (INRWild)
* A single reconstruction method (SingleINR)

You can run these methods using:
```bash
python meta.py
python fedavg.py
python single_inr.py
python inr_wild.py
```

## Citation
If you find our paper useful, please cite
```bibtex
@article{shi2024implicit,
  title={Implicit Neural Representations for Robust Joint Sparse-View CT Reconstruction},
  author={Shi, Jiayang and Zhu, Junyi and Pelt, Daniel M and Batenburg, K Joost and Blaschko, Matthew B},
  journal={arXiv preprint arXiv:2405.02509},
  year={2024}
}
```