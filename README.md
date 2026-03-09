# CNN & LSTM Stress Detection Architecture

This repository contains a robust multi-model machine learning architecture for Physiological Heart Rate Variability (HRV) classification schemas. Evaluated architectures run in fully distributed Github Actions utilizing PyTorch.

## Evaluated Neural Architectures

### 1. 1D Convolutional Neural Network (CNN1D)
Provides robust temporal spatial mapping over vectors.

[View CNN1D Architecture Diagram (Draw.io)](images/CNN1D_Architecture.drawio)
![CNN1D Architecture](images/CNN1D_Architecture.png)

### 2. Long Short-Term Memory Network (LSTM)
Utilizes sequence-gates to interpret biological data sequentially.
[View LSTM Cell Architecture Diagram (Draw.io)](images/LSTM_Architecture.drawio)
![LSTM Architecture](images/LSTM_Architecture.png)

### 3. Hybrid Convolutional LSTM (CNN+LSTM)
A pipeline extracting feature dependencies via convolutions sequentially decoded via LSTMs.
[View Hybrid CNN-LSTM Architecture Diagram (Draw.io)](images/CNN_LSTM_Architecture.drawio)
![Hybrid CNN-LSTM Architecture](images/CNN_LSTM_Architecture.png)

## Research Features
**Distribution Resampling / Augmentation**:
*   Baseline (Unbalanced Original Set)
*   SMOTE (Synthetic Minority Over-sampling Technique)
*   ADASYN (Adaptive Synthetic Minority Over-sampling Technique)

**Fully Distributed Continuous Integration**:
*   Using GitHub Actions (`main.yml` strategy configurations), experiments are automatically run simultaneously in parallel.
*   Once datasets (SMOTE/ADASYN vs Normal) and Models are exhaustively trained, the artifacts are grouped.
*   The system consolidates isolated runs via `--aggregate` into `PROFESSIONAL_RESEARCH_REPORT.md` available as a unified workflow ZIP.

## Local Execution
Ensure your environment is constructed properly (requires PyTorch):

```bash
pip install -r requirements.txt
```

Kickstart the primary orchestrator that iterates all models automatically:

```bash
# Run the entire loop locally
python main.py

# Or run isolated instances to debug:
python main.py --model cnn_lstm --balancing smote
```
