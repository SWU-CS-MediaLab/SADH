# Deep-hashing-with-self-supervised-asymmetric-semantic-excavation-and-margin-scalable-constraint

Pytorch implementation of our paper for deep hashing retrieval.

[Deep hashing with self-supervised asymmetric semantic excavation and margin-scalable constraint](https://www.sciencedirect.com/science/article/pii/S0925231222001035)

by Zhengyang Yu, Song Wu, Zhihao Dou and Erwin M.Bakker

Neurocomputing, 2022


## Introduction
This repository is Pytorch implementation of SADH, which mainly deals with deep hashing retrieval under multi-label scenario. The main insights of SADH are: 1) an asymmetric semantic learning strategy and 2) a margin-scalable similarity constraint. The network structure is illustrated as follows:
## How to use
You can easily train and test SADH via running:
> labnet.py
imgnet.py

You can download the datasets via:
>[NUSWIDE](https://github.com/TreezzZ/DSDH_PyTorch)
>[MS-COCO 2014](https://cocodataset.org/#download) 
>[MIR-Flickr25k](https://press.liacs.nl/mirflickr/mirdownload.html)
## Citation

> @article{article,
title = {Deep hashing with self-supervised asymmetric semantic excavation and margin-scalable constraint},
journal = {Neurocomputing},
volume = {483},
pages = {87-104},
year = {2022},
doi = {https://doi.org/10.1016/j.neucom.2022.01.082},
author = {Zhengyang Yu and Song Wu and Zhihao Dou and Erwin M. Bakker},
keywords = {Deep supervised hashing, Asymmetric learning, Self-supervised learning},
}





## Acknowledgement

Thanks for the work of [swuxyj](https://github.com/swuxyj). Our code is heavily borrowed from the implementation of [https://github.com/swuxyj/DeepHash-pytorch].
# SADH
