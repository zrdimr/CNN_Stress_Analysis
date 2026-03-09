# CNN Stress Detection Architecture

This repository holds a robust multi-model machine learning architecture for Physiological Heart Rate Variability (HRV) classification schemas designed specifically for analyzing stress dynamics under multiple augmentation structures.

## System Features
**1. Generic Deep Architectures Supported**:
*   1D CNN (Temporal Arrays)
*   2D CNN (Topology Map representations)
*   Multimodal Fusion (Merging dense representations alongside Convolutions)

**2. Distribution Resampling / Augmentation**:
*   Baseline (Unbalanced Original Set)
*   SMOTE (Synthetic Minority Over-sampling Technique)
*   ADASYN / Data Synthesis variants mapping class variances

**3. Execution Orchestrator (`main.py`)**:
*   Automatically ingests the dataset.
*   Triggers `train`, `evaluate`, `test` logic iteratively sequentially modifying models and dataset sampling techniques on the fly across multiple experiment parameter footprints (`config.yaml`).
*   Stores the highest performing configurations continuously via Validation Loss.

**4. Fully Automated Report Generation**:
*   Generates a standalone `PROFESSIONAL_RESEARCH_REPORT.md` combining research logic context and an aggregated statistical `Accuracy`.
*   Includes a visual cross-architecture Graph Comparison.
*   Spawns specific directory instances yielding isolated `confusion matrices` and `classification metric statistics`.

## Execution Guide

Make sure your environment is constructed properly (requires PyTorch):

```bash
pip install -r requirements.txt
```

Kickstart the primary orchestrator that iterates all models automatically:

```bash
python main.py
```
