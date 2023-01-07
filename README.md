# DynaGAN: Dynamic Few-shot Adaptation of GANs to Multiple Domains <br><sub>Official PyTorch Implementation of the SIGGRAPH Asia 2022 Paper</sub>
![Teaser image 1](srcs/teaser.png)
**DynaGAN: Dynamic Few-shot Adaptation of GANs to Multiple Domains**<br>
Seongtae Kim, Kyoungkook Kang, Geonung Kim, Seung-Hwan Baek, Sunghyun Cho<br>
[![arXiv](https://img.shields.io/static/v1?style=for-the-badge&message=arXiv&color=B31B1B&logo=arXiv&logoColor=FFFFFF&label=
)](https://arxiv.org/abs/2211.14554)
[![ACM](https://img.shields.io/static/v1?style=for-the-badge&message=ACM&color=0085CA&logo=ACM&logoColor=FFFFFF&label=)](https://dl.acm.org/doi/abs/10.1145/3550469.3555416)
[![OpenProject Badge](https://img.shields.io/badge/Project%20Page-E2638D?logo=openproject&logoColor=fff&style=for-the-badge)](https://bluegorae.github.io/dynagan/)

Abstract: *Few-shot domain adaptation to multiple domains aims to learn a complex image distribution across multiple domains from a few training images. A naïve solution here is to train a separate model for each domain using few-shot domain adaptation methods. Unfortunately, this approach mandates linearly-scaled computational resources both in memory and computation time and, more importantly, such separate models cannot exploit the shared knowledge between target domains. In this paper, we propose DynaGAN, a novel few-shot domain-adaptation method for multiple target domains. DynaGAN has an adaptation module, which is a hyper-network that dynamically adapts a pretrained GAN model into the multiple target domains. Hence, we can fully exploit the shared knowledge across target domains and avoid the linearly-scaled computational requirements. As it is still computationally challenging to adapt a large-size GAN model, we design our adaptation module light-weight using the rank-1 tensor decomposition. Lastly, we propose a contrastive-adaptation loss suitable for multi-domain few-shot adaptation. We validate the effectiveness of our method through extensive qualitative and quantitative evaluations.*

## Download pretrained models
Download files under `pretrained_models/`.
| Model | Description
| :--- | :----------
|[FFHQ](https://drive.google.com/file/d/1XQabKtkpMltyZkFYidX4jd8Zrii5eTyI/view?usp=sharing) | StyleGAN model pretrained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) with 1024x1024 output resolution.
|[FFHQ_PCA](https://drive.google.com/file/d/13b81CBny0VgxWJWWEylNJkNbXuQ512ug/view?usp=sharing) | PCA components of the pretrained StyleGAN(FFHQ) latent space.
|[ArcFace](https://drive.google.com/file/d/1bwcB_AvbD0_qHGUoQCxzbp2wEurhjD4c/view?usp=sharing) | Pretrained face recognition model to calculate identity loss.
|[AFHQ_Cat](https://drive.google.com/file/d/17K_U0IKaVKoQT4lJ6zf1h6ijfmrHSB7B/view?usp=sharing) | StyleGAN model pretrained on [AFHQ_Cat](https://github.com/clovaai/stargan-v2) with 512x512 output resolution.
|[AFHQ_Cat_PCA](https://drive.google.com/file/d/1_JiWz-8eiki-LFFF0Aerf8GpM6mpjpYR/view?usp=share_link) | PCA components of the pretrained StyleGAN(AFHQ_Cat) latent space.

## Training
```shell
bash ./scripts/train.sh
```

## Citation

```bibtex
@inproceedings{Kim2022DynaGAN,
    title     = {DynaGAN: Dynamic Few-shot Adaptation of GANs to Multiple Domains},
    author    = {Seongtae Kim and Kyoungkook Kang and Geonung Kim and Seung-Hwan Baek and Sunghyun Cho},
    booktitle = {Proceedings of the ACM (SIGGRAPH Asia)},
    year      = {2022}
}
``` 
