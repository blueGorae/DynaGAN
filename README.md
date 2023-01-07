# DynaGAN: Dynamic Few-shot Adaptation of GANs to Multiple Domains <br><sub>Official PyTorch Implementation of the SIGGRAPH Asia 2022 Paper</sub>
![Teaser image 1](srcs/teaser.png)
**DynaGAN: Dynamic Few-shot Adaptation of GANs to Multiple Domains**<br>
Seongtae Kim, Kyoungkook Kang, Geonung Kim, Seung-Hwan Baek, Sunghyun Cho<br>
[![arXiv](https://img.shields.io/static/v1?style=for-the-badge&message=arXiv&color=B31B1B&logo=arXiv&logoColor=FFFFFF&label=
)](https://arxiv.org/abs/2211.14554)
[![ACM](https://img.shields.io/static/v1?style=for-the-badge&message=ACM&color=0085CA&logo=ACM&logoColor=FFFFFF&label=)](https://dl.acm.org/doi/abs/10.1145/3550469.3555416)
[![OpenProject Badge](https://img.shields.io/badge/Project%20Page-E2638D?logo=openproject&logoColor=fff&style=for-the-badge)](https://bluegorae.github.io/dynagan/)

Abstract: *Few-shot domain adaptation to multiple domains aims to learn a complex image distribution across multiple domains from a few training images. A naïve solution here is to train a separate model for each domain using few-shot domain adaptation methods. Unfortunately, this approach mandates linearly-scaled computational resources both in memory and computation time and, more importantly, such separate models cannot exploit the shared knowledge between target domains. In this paper, we propose DynaGAN, a novel few-shot domain-adaptation method for multiple target domains. DynaGAN has an adaptation module, which is a hyper-network that dynamically adapts a pretrained GAN model into the multiple target domains. Hence, we can fully exploit the shared knowledge across target domains and avoid the linearly-scaled computational requirements. As it is still computationally challenging to adapt a large-size GAN model, we design our adaptation module light-weight using the rank-1 tensor decomposition. Lastly, we propose a contrastive-adaptation loss suitable for multi-domain few-shot adaptation. We validate the effectiveness of our method through extensive qualitative and quantitative evaluations.*



## Citation

```
@inproceedings{Kim2022DynaGAN,
    title     = {DynaGAN: Dynamic Few-shot Adaptation of GANs to Multiple Domains},
    author    = {Seongtae Kim and Kyoungkook Kang and Geonung Kim and Seung-Hwan Baek and Sunghyun Cho},
    booktitle = {Proceedings of the ACM (SIGGRAPH Asia)},
    year      = {2022}
}
``` 
