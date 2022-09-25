# Awesome Distributed Machine Learning System

![Awesome](https://awesome.re/badge.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/shenggan/awesome-distributed-ml/pulls)

A curated list of awesome projects and papers for distributed training or inference **especially for large model**.

## Contents
- [Awesome Distributed Machine Learning System](#awesome-distributed-machine-learning-system)
  - [Contents](#contents)
  - [Open Source Projects](#open-source-projects)
  - [Papers](#papers)
    - [Survey](#survey)
    - [Pipeline Parallelism](#pipeline-parallelism)
    - [Mixture-of-Experts Parallelism](#mixture-of-experts-parallelism)
    - [Hybrid Parallelism & Framework](#hybrid-parallelism--framework)
    - [Memory Efficient Training](#memory-efficient-training)
    - [Auto Parallelization](#auto-parallelization)
    - [Communication Optimization](#communication-optimization)
    - [Inference](#inference)
    - [Applications](#applications)
  - [Contribute](#contribute)

## Open Source Projects

- [Megatron-LM: Ongoing Research Training Transformer Models at Scale](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed: A Deep Learning Optimization Library that Makes Distributed Training and Inference Easy, Efficient, and Effective.](https://www.deepspeed.ai/)
- [ColossalAI: A Unified Deep Learning System for Large-Scale Parallel Training](https://www.colossalai.org/)
- [OneFlow: A Performance-Centered and Open-Source Deep Learning Framework](www.oneflow.org)
- [Mesh TensorFlow: Model Parallelism Made Easier](https://github.com/tensorflow/mesh)
- [FlexFlow: A Distributed Deep Learning Framework that Supports Flexible Parallelization Strategies.](https://github.com/flexflow/FlexFlow)
- [Alpa: Auto Parallelization for Large-Scale Neural Networks](https://github.com/alpa-projects/alpa)
- [Easy Parallel Library: A General and Efficient Deep Learning Framework for Distributed Model Training](https://github.com/alibaba/EasyParallelLibrary)
- [FairScale: PyTorch Extensions for High Performance and Large Scale Training](https://github.com/facebookresearch/fairscale)
- [Pipeline Parallelism and SPMD for PyTorch](https://github.com/pytorch/tau)

## Papers

### Survey

- [Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis](https://arxiv.org/abs/1802.09941) by Tal Ben-Nun et al., ACM Computing Surveys 2020
- [A Survey on Auto-Parallelism of Neural Networks Training](https://www.techrxiv.org/articles/preprint/A_Survey_on_Auto-Parallelism_of_Neural_Networks_Training/19522414) by Peng Liang., techrxiv 2022

### Pipeline Parallelism

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://proceedings.neurips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf) by Yanping Huang et al., NeurIPS 2019
- [PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/10.1145/3341301.3359646) by Deepak Narayanan et al., SOSP 2019
- [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503v3) by Deepak Narayanan et al., ICML 2021
- [DAPPLE: a pipelined data parallel approach for training large models](https://dl.acm.org/doi/10.1145/3437801.3441593) by Shiqing Fan et al. PPoPP 2021
- [Chimera: efficiently training large-scale neural networks with bidirectional pipelines](https://dl.acm.org/doi/abs/10.1145/3458817.3476145) by Shigang Li et al., SC 2021

### Mixture-of-Experts Parallelism

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://openreview.net/forum?id=qrwe7XHTmYb) by Dmitry Lepikhin et al., ICLR 2021
- [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models](https://dl.acm.org/doi/abs/10.1145/3503221.3508418) by Jiaao He et al., PPoPP 2022
- [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596) by Samyam Rajbhandari et al., ICML 2022
- [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382) by Changho Hwang et al., arxiv 2022

### Hybrid Parallelism & Framework

- [Efficient large-scale language model training on GPU clusters using megatron-LM](https://dl.acm.org/doi/10.1145/3458817.3476209) by Deepak Narayanan et al., SC 2021
- [GEMS: GPU-Enabled Memory-Aware Model-Parallelism System for Distributed DNN Training](https://ieeexplore.ieee.org/document/9355254) by Arpan Jain et al., SC 2020
- [Amazon SageMaker Model Parallelism: A General and Flexible Framework for Large Model Training](https://arxiv.org/abs/2111.05972) by Can Karakus et al., arxiv 2021
- [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032) by Jinhui Yuan et al., arxiv 2021
- [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training](https://arxiv.org/abs/2110.14883) by Zhengda Bian., arxiv 2021

### Memory Efficient Training

- [Training deep nets with sublinear memory cost](https://arxiv.org/abs/1604.06174) by Tianqi Chen et al., arxiv 2016
- [ZeRO: memory optimizations toward training trillion parameter models](https://dl.acm.org/doi/10.5555/3433701.3433727) by Samyam Rajbhandari et al., SC 2020
- [Capuchin: Tensor-based GPU Memory Management for Deep Learning](https://dl.acm.org/doi/10.1145/3373376.3378505) by Xuan Peng et al., ASPLOS 2020
- [SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping](https://dl.acm.org/doi/10.1145/3373376.3378530) by Chien-Chin Huang et al., ASPLOS 2020
- [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://proceedings.mlsys.org/paper/2020/hash/084b6fbb10729ed4da8c3d3f5a3ae7c9-Abstract.html) by Paras Jain et al., MLSys 2020
- [Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616) by Marisa Kirisame et al., ICLR 2021
- [ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://proceedings.mlr.press/v139/chen21z.html) by Jianfei Chen et al., ICML 2021
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/conference/atc21/presentation/ren-jie) by Jie Ren et al., ATC 2021
- [ZeRO-infinity: breaking the GPU memory wall for extreme scale deep learning](https://dl.acm.org/doi/abs/10.1145/3458817.3476205) by Samyam Rajbhandari et al., SC 2021
- [PatrickStar: Parallel Training of Pre-trained Models via Chunk-based Memory Management](https://arxiv.org/abs/2108.05818) by Jiarui Fang et al., arxiv 2021
- [GACT: Activation Compressed Training for Generic Network Architectures](https://proceedings.mlr.press/v162/liu22v.html) by Xiaoxuan Liu et al., ICML 2022
- [MegTaiChi: dynamic tensor-based memory management optimization for DNN training](https://dl.acm.org/doi/10.1145/3524059.3532394) by Zhongzhe Hu et al., ICS 2022

### Auto Parallelization

- [Mesh-tensorflow: Deep learning for supercomputers](https://proceedings.neurips.cc/paper/2018/hash/3a37abdeefe1dab1b30f7c5c7e581b93-Abstract.html) by Noam Shazeer et al., NeurIPS 2018
- [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](https://arxiv.org/abs/1802.04924) by Zhihao Jia et al., ICML 2018
- [Beyond Data and Model Parallelism for Deep Neural Networks](https://proceedings.mlsys.org/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html) by Zhihao Jia et al., MLSys 2019
- [Supporting Very Large Models using Automatic Dataflow Graph Partitioning](https://dl.acm.org/doi/abs/10.1145/3302424.3303953) by Minjie Wang et al., EuroSys 2019
- [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) by Yuanzhong Xu et al., arxiv 2021
- [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) by Lianmin Zheng et al., OSDI 2022
- [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/conference/osdi22/presentation/unger) by Colin Unger, Zhihao Jia, et al., OSDI 2022

### Communication Optimization

- [Blink: Fast and Generic Collectives for Distributed ML](https://proceedings.mlsys.org/paper/2020/hash/43ec517d68b6edd3015b3edc9a11367b-Abstract.html) by Guanhua Wang et al., MLSys 2020
- [GC3: An Optimizing Compiler for GPU Collective Communication](https://arxiv.org/abs/2201.11840) by Meghan Cowan et al., arxiv 2022
- [Breaking the computation and communication abstraction barrier in distributed machine learning workloads](https://dl.acm.org/doi/10.1145/3503222.3507778) by Abhinav Jangda et al., ASPLOS 2022

### Inference

- [DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032) by Reza Yazdani Aminabadi et al., arxiv 2022
- [EnergonAI: An Inference System for 10-100 Billion Parameter Transformer Models](https://arxiv.org/abs/2209.02341) by Jiangsu Du et al., arxiv 2022

### Applications

- [BaGuaLu: targeting brain scale pretrained models with over 37 million cores](https://dl.acm.org/doi/abs/10.1145/3503221.3508417) by Zixuan Ma et al., PPoPP 2022
- [NASPipe: High Performance and Reproducible Pipeline Parallel Supernet Training via Causal Synchronous Parallelism](https://dl.acm.org/doi/abs/10.1145/3503222.3507735) by Shixiong Zhao et al., ASPLOS 2022

## Contribute

All contributions to this repository are welcome. Open an [issue](https://github.com/shenggan/awesome-distributed-ml/issues) or send a [pull request](https://github.com/shenggan/awesome-distributed-ml/pulls).
