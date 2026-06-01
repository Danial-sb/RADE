# 🚀 RADE: Random Add-Drop Edge as a Regularizer

**Official implementation of the ICML 2026 paper.**

RADE is a stochastic graph augmentation framework for graph neural networks that jointly addresses **overfitting** and **over-squashing** through random edge deletion, random edge addition, and expectation-preserving aggregation corrections.

This repository includes experiments for **node classification** and **graph classification**, with implementations of RADE variants and common augmentation baselines.

---

## Overview

Message-passing GNNs face two important limitations: **overfitting** and **over-squashing**. Stochastic graph augmentations, such as edge deletion, can regularize training, but they may introduce train-inference aggregation mismatch and do not directly improve long-range communication. Rewiring methods improve connectivity to mitigate over-squashing, but they are not primarily designed as training-time regularizers.

RADE bridges these two directions by combining random edge deletion and random edge addition within a unified augmentation framework. It uses expectation-preserving aggregation corrections to align stochastic training-time aggregation with the intended inference-time aggregation.

The framework supports two variants:

* **RADE-OF** targets overfitting by aligning the expected training-time aggregation with the input-graph inference aggregation.
* **RADE-OFS** targets both overfitting and over-squashing by correcting deletion effects while retaining the expected contribution of added edges at inference, creating additional communication paths for long-range information flow.

RADE also includes adaptive selection of deletion and addition rates through a GradNorm-style controller.

---

## Repository Structure

```text
RADE/
├── RADE_Node_Classification/    # Node-classification experiments
└── RADE_Graph_Classification/   # Graph-classification experiments
```

The node-classification code contains both full-batch and mini-batch training pipelines. The graph-classification code contains the corresponding graph-level training and evaluation pipeline.

---

## Supported Settings

**Tasks**

* Node classification
* Graph classification

**Backbones**

* GCN
* GIN
* GAT

**Augmentation methods**

* RADE
* DropEdge
* DropMessage
* DropNode
* Dropout
* No augmentation

**RADE variants**

* `rade-of`
* `rade-ofs`

---

## Usage

### Node Classification

```bash
cd RADE_Node_Classification/full_batch

python main.py \
  --dataset cora \
  --gnn gcn \
  --aug_tech rade \
  --rade_variant rade-of \
  --ep_correction True \
  --pq_gradnorm True
```

### Graph Classification

```bash
cd RADE_Graph_Classification

python main_gc.py \
  --dataset mutag \
  --gnn gin \
  --aug_tech rade \
  --rade_variant rade-of \
  --ep_correction True \
  --pq_gradnorm True
```

---

## Main Arguments

Common arguments include:

```bash
--dataset          Dataset name
--gnn              GNN backbone
--aug_tech         Augmentation method
--rade_variant     RADE variant: rade-of or rade-ofs
--ep_correction    Whether to use expectation-preserving correction
--pq_gradnorm      Whether to adapt p and q during training
--p                Initial edge-drop probability
--q                Initial edge-add probability
```

---

## Citation

Don't forget to cite our paper!

```bibtex
@inproceedings{rade2026,
  title     = {RADE: Random Add-Drop Edge as a Regularizer},
  author    = {Saber, Danial and Salehi-Abari, Amirali},
  booktitle = {To be added},
  year      = {2026}
}
```

The final citation will be updated once the official proceedings information is available.
