# Transition Path Sampling with Diffusion Path Samplers
[![arXiv](https://img.shields.io/badge/arXiv-2405.19961-84cc16)](https://arxiv.org/abs/2405.19961)

This repository contains the code to reproduce the results of [paper](https://arxiv.org/abs/2405.19961) (ICLR 2025) and [project page](https://kiyoung98.github.io/tps-dps/) for 3D videos of transition paths sampled from our diffusion path sampler.

## Installation
We provide a script for installing packages (assuming CUDA 11.x).
```
conda env create -f environment.yml
conda activate tps-dps
```

## Quickstart: Synthetic Double-Well Example
We provide self-contained notebook `double-well.ipynb` for training TPS-DPS and sampling transition paths in a synthetic double-well system. 

## Project Structure
```
tps-dps/
│
├── src/                # Main source code
│   ├── train.py        # Training a model
│   ├── sample.py       # Sampling transition paths using a trained model
│   ├── eval.py         # Evaluation of sampled paths
│   ├── dps.py          # Core DPS modules (DPS, replay buffer, target path measure)
│   ├── bias.py         # Bias force model architecture
│   ├── dynamics/       # Molecular dynamics system definitions
│   └── utils/          # Utilities (metrics, plotting, etc.)
│
├── scripts/            # Shell scripts for batch experiments
│   ├── train/          # Training scripts for each system
│   ├── sample/         # Sampling scripts for each system
│   └── eval/           # Evaluation scripts for each system
│
├── data/               # Input data (pdb file, tica models, pmf, etc.)
├── figures/            # Output figures and animations
├── double-well.ipynb   # Synthetic example notebook
├── environment.yml     # Conda environment specification
└── README.md           
```

## Steps to reproduce the results
We provide instructions to reproduce the results of aldp and train a new model. You can replace aldp with fast-folding proteins: chignolin, trpcage, bba, and bbl.

- **Sampling**: Run the following command to sample transition paths.
    ```
    bash scripts/sample/aldp.sh
    ```
- **Evaluation**: Run the following command to evaluate sampled paths.
    ```
    bash scripts/eval/aldp.sh
    ```
- **Training**: Run the following command to start training from scratch. For better results, please try many seeds.
    ```
    bash scripts/train/aldp.sh
    ```

## Citation

```bibtex
@inproceedings{seong2025transition,
  title={Transition path sampling with improved off-policy training of diffusion path samplers},
  author={Seong, Kiyoung and Park, Seonghyun and Kim, Seonghwan and Kim, Woo Youn and Ahn, Sungsoo},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={93040--93062},
  year={2025}
}
```
