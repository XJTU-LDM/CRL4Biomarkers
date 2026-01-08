# CRL4Biomarkers
# Decoupling Predictive & Prognostic Signals in Immunotherapy with Continual RL

This repository contains the official implementation of the paper **"Continual reinforcement learning decouples predictive and prognostic genomic signals in immune checkpoint therapy"**.

## Overview
Immune checkpoint therapy (ICT) shows remarkable efficacy in some cancer patients, but robust biomarkers to guide treatment are still lacking. A key challenge is that genomic features often confound **predictive** signals (response to therapy) with **prognostic** signals (inherent disease aggressiveness).

This project introduces a novel **continual reinforcement learning (CRL)** framework to dynamically and sequentially analyze high-dimensional genomic data. Our method successfully **decouples** these entangled signals, leading to more accurate identification of patients who will truly benefit from ICT.

## Key Features
*   **Novel CRL Framework:** Implements a continual learning agent that interacts with a patient cohort simulator, learning to assign predictive and prognostic scores to genomic features over time without catastrophic forgetting.
*   **Benchmark Datasets:** Includes processed and scripted access to public genomic datasets (e.g., from TCGA) relevant to ICT response.
*   **Reproducible Pipeline:** End-to-end code for data preprocessing, model training, evaluation, and visualization of decoupled signal trajectories.
*   **Biomarker Discovery:** Tools to extract and interpret the top predictive features identified by the model as potential novel biomarkers.
## Main scripts:
- `hssltrain2.py` - Training script
- `test.py` - Testing script
## Citation
If you use this code in your research, please cite our paper:
