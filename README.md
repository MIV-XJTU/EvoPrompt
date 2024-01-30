# EvoPrompt

The official PyTorch implementation of our **AAAI 2024 (Oral)** paper:

[Evolving Parameterized Prompt Memory for Continual Learning]() 

*Muhammad Rifki Kurniawan, Xiang Song, Zhiheng Ma, Yuhang He, Yihong Gong, Qi Yang, Xing Wei.*

GitHub maintainer: [Muhammad Rifki Kurniawan](https://mrifkikurniawan.github.io/)

## Highlight

<div align=center><img src="assets/evoprompt.svg" width="80%" height="80%"></div>

### :bookmark:Brief Introduction

Recent studies have demonstrated the potency of leveraging prompts in Transformers for continual learning (CL). Nevertheless, employing a discrete key-prompt bottleneck can lead to selection mismatches and inappropriate prompt associations during testing. Furthermore, this approach hinders adaptive prompting due to the lack of *shareability* among nearly identical instances at more granular level. To address these challenges, we introduce the **Evo**lving Parameterized **Prompt** Memory (EvoPrompt), a novel method involving adaptive and continuous prompting attached to pre-trained Vision Transformer (ViT), conditioned on specific instance. We formulate a continuous prompt function as a neural bottleneck and encode the collection of prompts on network weights. We establish a paired prompt memory system consisting of a **stable reference** and a **flexible working prompt memory**. Inspired by *linear mode connectivity*, we progressively **fuse** the working prompt memory and reference prompt memory during inter-task periods, resulting in **continually evolved prompt memory**. This fusion involves aligning functionally equivalent prompts using optimal transport and aggregating them in parameter space with an adjustable bias based on prompt node attribution. Additionally, to enhance backward compatibility, we propose **compositional classifier initialization**, which leverages prior prototypes from pre-trained models to guide the initialization of new classifiers in a subspace-aware manner.

#### The Code Will Be Released Soon

## Citation

If you find any bits of our paper or code useful for your research, we'd be thrilled if you could give us a shout-out by citing our work and leaving a star on our repository.


```
@inproceedings{kurniawan2024evoprompt,
  title = {Evolving Parameterized Prompt Memory for Continual Learning},
  author = {Kurniawan, Muhammad Rifki and Song, Xiang and Ma, Zhiheng and He, Yuhang and Gong, Yihong and Yang, Qi and Wei, Xing},
  year = {2024},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
}
```