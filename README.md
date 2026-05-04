# SSD Benchmark

Code and evaluation scripts for a fine-grained temporal phase segmentation benchmark on surfing skeleton sequences.

## Overview

This repository provides the benchmark code, evaluation scripts, official split files, and baseline-specific configuration files used for experiments on SSD.

The repository focuses on:
- unified data split protocol,
- unified evaluation protocol,
- benchmark scripts for supervised and unsupervised settings,
- method-specific configuration files.

This repository does **not** redistribute third-party baseline source code. Official implementations should be obtained from the respective original repositories and used together with the configuration files and evaluation scripts provided here.

## Dataset Format

Each sample in SSD is represented as:
- a skeleton sequence of shape `(103, 17, 2)`,
- a frame-wise label file with 103 lines,
- official split files for train/val/test.

## Benchmark Protocol

All methods are evaluated under the same benchmark protocol:
- input modality: 2D skeleton sequences only,
- official train/val/test split,
- no RGB appearance or scene context,
- unified evaluation metrics.

### Supervised metrics
- frame-wise accuracy
- Edit
- F1@10
- F1@25
- F1@50

### Unsupervised metrics
- MoF
- mIoU
- Edit
- F1

## Repository Scope

This repository includes:
- benchmark evaluation scripts,
- official split files,
- configuration files,
- minimal utilities required for benchmark evaluation.

This repository does **not** include:
- original competition videos,
- third-party baseline source code redistribution.

## Baselines

### Supervised baselines
- MS-TCN
- MS-GCN
- ASRF
- ASFormer
- TRG-Net

### Unsupervised baselines
- CTE
- TOT
- ASOT
- HVQ
- SMQ

Official baseline implementations should be obtained from their original repositories.

## Notes

This repository is intended for research and benchmark evaluation only.
