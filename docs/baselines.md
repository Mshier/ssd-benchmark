# Baseline Notes

This benchmark uses the official implementations of the compared baselines whenever available.

This repository does **not** redistribute third-party baseline source code. Reviewers should obtain the official implementations from the corresponding original repositories.

## Included in this repository

This repository provides:
- unified evaluation scripts for SSD,
- dataset split files,
- SSD-specific utility scripts,
- minimal benchmark adaptation notes.

## Not included in this repository

This repository does not include:
- third-party baseline source code,
- original competition videos,
- duplicated configuration files already provided in the original baseline repositories.

## Benchmark setting

All methods are evaluated on SSD under the same protocol:
- input modality: 2D skeleton sequences only,
- official train/val/test split,
- no RGB appearance or scene context,
- unified evaluation metrics.

## Baseline adaptation notes

For baselines whose original repositories already contain method-specific configuration files, we follow the original setup and only apply the minimum modifications required to support SSD.

These modifications may include:
- adapting the input format to SSD skeleton sequences,
- adjusting feature dimensions,
- exporting frame-wise predictions in the SSD evaluation format,
- adding benchmark-specific utility scripts when needed.

Detailed hyperparameter settings are reported in the supplementary material of the submission.
