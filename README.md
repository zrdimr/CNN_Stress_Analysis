# CNN & LSTM Stress Detection Architecture

This repository contains a robust multi-model machine learning architecture for Physiological Heart Rate Variability (HRV) classification schemas. Evaluated architectures run in fully distributed Github Actions utilizing PyTorch.

## Evaluated Neural Architectures

### 1. 1D Convolutional Neural Network (CNN1D)
Provides robust temporal spatial mapping over vectors.

```mermaid
graph LR
    Input[Input Layer<br>Matrix fingerprints] --> Conv1[Conv1D<br>Filter: 512, Kernel: 3]
    Conv1 --> Pool1[Max Pooling]
    Pool1 --> Conv2[Conv1D<br>Filter: 256, Kernel: 3]
    Conv2 --> Pool2[Max Pooling]
    Pool2 --> Conv3[Conv1D<br>Filter: 128, Kernel: 3]
    Conv3 --> Pool3[Max Pooling]
    Pool3 --> Conv4[Conv1D<br>Filter: 64, Kernel: 3]
    Conv4 --> Pool4[Max Pooling]
    
    Pool4 --> FC1[Fully Connected<br>1024 Neurons<br>Dropout: 0.5]
    FC1 --> FC2[Fully Connected<br>512 Neurons<br>Dropout: 0.5]
    FC2 --> FC3[Fully Connected<br>256 Neurons]
    FC3 --> Sigmoid[Output<br>Multi-class neurons]
    
    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Sigmoid fill:#f9f,stroke:#333,stroke-width:2px
```

### 2. Long Short-Term Memory Network (LSTM)
Utilizes sequence-gates to interpret biological data sequentially.

```mermaid
graph TD
    classDef gate fill:#f9d0c4,stroke:#333,stroke-width:1px;
    classDef mem fill:#c4e1f9,stroke:#333,stroke-width:1px;
    
    subgraph LSTM Cell
        F((Forget Gate<br>σ)):::gate
        I((Input Gate<br>σ)):::gate
        C_cand((Candidate<br>tanh)):::mem
        O((Output Gate<br>σ)):::gate
        
        PrevC((Prev Memory)):::mem --> F
        PrevH((Prev Hidden state)) --> F
        InputX((Input X_t)) --> F
        
        F --> NewC((Cell State)):::mem
        I --> NewC
        C_cand --> NewC
        
        NewC --> O
        O --> NextH((Next Hidden state))
    end
```

### 3. Hybrid Convolutional LSTM (CNN+LSTM)
A pipeline extracting feature dependencies via convolutions sequentially decoded via LSTMs.

```mermaid
graph LR
    Input[Input Series] --> CNN[CNN Feature<br>Extraction]
    CNN --> Pool[Max Pooling]
    Pool --> LSTM[LSTM Sequential<br>Evaluation]
    LSTM --> FC[Fully Connected<br>Output Classifier]
```

## Research Features
**Distribution Resampling / Augmentation**:
*   Baseline (Unbalanced Original Set)
*   SMOTE (Synthetic Minority Over-sampling Technique)
*   ADASYN (EnTDA proxy methods mapping variance synthetically)

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
