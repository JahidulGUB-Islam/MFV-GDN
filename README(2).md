# MFV-GDN — Multi-Dataset Intrusion Detection Framework

This repository contains the **MFV-GDN** implementation for three benchmark intrusion-detection datasets:

- **CICIoT2023**
- **NSL-KDD**
- **UNSW-NB15**

MFV-GDN combines **Harris Hawks Optimization (HHO)**, **WGAN-GP**, **GCN-GAT**, **Multi-Constrained Fuzzy V-Detector NSA (MFV-NSA)**, **Clonal Selection Algorithm (CSA)**, and a **CNN-BiLSTM-Multi-Head Attention (MHA)** classifier in a leakage-aware intrusion-detection pipeline.

The repository supports:

- binary attack detection,
- multi-class attack classification,
- nested/group-aware evaluation,
- final locked-test evaluation,
- class-wise analysis,
- baseline-model comparison,
- SHAP/LIME explainability,
- adversarial robustness experiments.

---

## 1. Notebooks

| Dataset | Notebook |
|---|---|
| CICIoT2023 | `MFV_GDN_CICIoT2023_GitHub.ipynb` |
| NSL-KDD | `MFV_GDN_NSL_KDD_GitHub.ipynb` |
| UNSW-NB15 | `MFV_GDN_UNSW_NB15_GitHub(1).ipynb` |

Each notebook follows the same MFV-GDN methodology while using dataset-specific loading, labels, and train/test organization.

---

## 2. MFV-GDN Pipeline

```text
Dataset
   ↓
Split-first partitioning / locked test setup
   ↓
Exact + near-duplicate audit
   ↓
Training-only preprocessing
   ↓
Harris Hawks Optimization (HHO)
   ↓
WGAN-GP minority-class balancing
   ↓
┌────────────────────────────────────────────┐
│ MFV-NSA        GCN-GAT Graph Model    CSA │
└────────────────────────────────────────────┘
   ↓
Shared feature representation
   ↓
CNN-BiLSTM-Multi-Head Attention
   ↓
Calibrated decision fusion
   ↓
Binary + Multi-class predictions
   ↓
Performance, XAI, and robustness analysis
```

The central design principle is **leakage control**: learned transformations are fitted using training-derived data and then frozen before validation/test transformation.

---

# 3. Datasets

## 3.1 CICIoT2023

### Default dataset directory

```text
/datapool/home/mi03289/Desktop/ResearchJ/CICIoT2023
```

The notebook searches recursively for partition files beginning with:

```text
part-
```

Supported formats include:

```text
CSV
CSV.GZ
Parquet / PQ
Feather
JSON / JSONL
extensionless Spark-style partition files
```

### Classes

| ID | Class |
|---:|---|
| 0 | Benign |
| 1 | DDoS |
| 2 | DoS |
| 3 | Recon |
| 4 | Web-Based |
| 5 | Brute Force |
| 6 | Spoofing |
| 7 | Mirai |

Binary labels:

```text
Benign → 0
Attack → 1
```

### Split strategy

CICIoT2023 uses source-partition information when possible.

- With at least five source partitions, the notebook uses a grouped stratified split based on source-file groups.
- Otherwise, it falls back to a stratified train/test split.
- Exact and near-duplicate collisions between development and test partitions are audited after the split.
- Colliding training rows are removed before model development.

### CICIoT2023-specific runtime options

```bash
export MFV_CIC_MAX_FILES=0
export MFV_CIC_ROWS_PER_FILE=0
export MFV_CIC_TEST_FOLD=0
```

`0` for file/row limits means no limit.

---

## 3.2 NSL-KDD

### Default dataset directory

```text
/datapool/home/mi03289/Desktop/ResearchJ/NSL_KDD
```

Expected files:

```text
kdd_train.csv
kdd_test.csv
```

The loader supports common NSL-KDD CSV layouts, including headered files and standard headerless formats.

### Classes

| ID | Class |
|---:|---|
| 0 | Normal |
| 1 | DoS |
| 2 | Probe |
| 3 | R2L |
| 4 | U2R |

Binary labels:

```text
Normal → 0
Attack → 1
```

Fine-grained NSL-KDD attack names are grouped into the five categories above.

Examples include:

- **DoS:** Neptune, Smurf, Teardrop, Apache2, Back, etc.
- **Probe:** Nmap, Portsweep, Ipsweep, Satan, etc.
- **R2L:** Guess_Passwd, FTP_Write, Warezmaster, Phf, etc.
- **U2R:** Buffer_Overflow, Rootkit, Perl, Xterm, etc.

### Split strategy

The official dataset files are treated as:

```text
kdd_train.csv → development/training data
kdd_test.csv  → locked test data
```

Before model development:

1. exact signatures are computed,
2. near-duplicate signatures are computed,
3. development samples colliding with the locked test set are removed,
4. the cleaned development set is used for fitting and validation.

The locked test partition is not used to fit preprocessing, HHO, WGAN-GP, MFV-NSA, GNN, CSA, or classifier parameters.

---

## 3.3 UNSW-NB15

### Default dataset directory

```text
/datapool/home/mi03289/Desktop/ResearchJ/UNSW_NB15
```

Expected files:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

### Classes

| ID | Class |
|---:|---|
| 0 | Normal |
| 1 | Fuzzers |
| 2 | Analysis |
| 3 | Backdoor |
| 4 | DoS |
| 5 | Exploits |
| 6 | Generic |
| 7 | Reconnaissance |
| 8 | Shellcode |
| 9 | Worms |

Binary labels:

```text
Normal → 0
Attack → 1
```

The notebook uses the `attack_cat` field for multi-class target engineering and supports the standard numeric `label` field for binary labels.

### Split strategy

The official files are treated as:

```text
UNSW_NB15_training-set.csv → development/training data
UNSW_NB15_testing-set.csv  → locked test data
```

The notebook performs a cross-partition exact/near-duplicate audit and removes colliding training samples before model development.

The locked test set remains separate from all model-fitting stages.

---

# 4. Dataset Summary

| Dataset | Multi-Class Categories | Main Input Organization | Default Output Directory |
|---|---:|---|---|
| CICIoT2023 | 8 | Partitioned files (`part-*`) | `MFV_GDN_CICIoT2023_outputs` |
| NSL-KDD | 5 | `kdd_train.csv`, `kdd_test.csv` | `MFV_GDN_NSL_KDD_outputs` |
| UNSW-NB15 | 10 | Official training/testing CSVs | `MFV_GDN_UNSW_NB15_outputs` |

All three implementations perform both:

```text
Binary detection
Multi-class classification
```

---

# 5. Core MFV-GDN Components

## 5.1 HHO Feature Selection

**Harris Hawks Optimization (HHO)** selects a compact subset of features using a fitness objective that balances predictive performance and sparsity.

Shared default settings:

```text
Alpha             = 0.95
Hawks             = 20
Iterations        = 25
RF trees          = 100
```

Feature selection is performed using training-derived data only.

---

## 5.2 WGAN-GP Balancing

A class-specific **Wasserstein GAN with Gradient Penalty** is trained for minority-class augmentation.

Shared default settings:

```text
Noise dimension       = 64
Epochs                = 300
Learning rate         = 2e-4
Gradient penalty λ    = 10
Batch size            = 128
Critic updates        = 5
```

Synthetic samples are added only to the training partition.

Validation and test distributions are not synthetically balanced.

---

## 5.3 GCN-GAT Graph Learning

The graph-learning branch combines **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)**.

Shared default settings:

```text
Hidden dimension      = 64
Embedding dimension   = 32
Attention heads       = 4
Dropout               = 0.30
Training epochs       = 60
Pretraining epochs    = 60
Learning rate         = 1e-3
Graph edge budget     = 4000
```

Training, validation, and test graph transformations are kept separate according to the notebook's leakage-control protocol.

---

## 5.4 MFV-NSA

The **Multi-Constrained Fuzzy V-Detector Negative Selection Algorithm (MFV-NSA)** is the immune-inspired anomaly-detection component.

It contains five main stages:

```text
Stage 1 → Benign self-set construction
Stage 2 → Adaptive self-boundary estimation
Stage 3 → Multi-constrained detector generation
Stage 4 → Fuzzy membership and anomaly scoring
Stage 5 → 14-dimensional immune-feature extraction
```

Shared default settings:

```text
Maximum self samples   = 8000
Boundary percentile    = 95
Candidate detectors    = 120000
Final detectors        = 600
Minimum radius         = 0.15
Maximum radius         = 5.0
Separation beta        = 0.30
Fuzzy gamma            = 6.0
Distance weight        = 0.60
Detector weight        = 0.40
Nearest detectors K    = 10
```

The extracted MFV-NSA representation contains 14 immune/anomaly features for each sample.

---

## 5.5 CSA

The **Clonal Selection Algorithm (CSA)** creates class-specific memory cells.

Shared defaults:

```text
Memory cells/class     = 15
Clones                 = 8
Generations            = 12
Mutation parameter     = 0.25
```

CSA generates class-probability information that is used in the shared representation and decision-fusion stage.

---

## 5.6 CNN-BiLSTM-MHA

The final deep classifier combines:

- Conv1D
- Bidirectional LSTM
- Multi-Head Attention
- dense layers
- dropout
- binary output
- multi-class output

Shared defaults:

```text
Conv1D filters       = 64, 32
BiLSTM units         = 64
Attention heads      = 4
Attention key dim    = 16
Dense units          = 128, 64
Dropout              = 0.25, 0.35, 0.25
Epochs               = 80
Batch size           = 128
Learning rate        = 1e-3
Optimizer            = Adam
```

---

# 6. Installation

A GPU-enabled environment is recommended for the complete experiments.

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib networkx
pip install torch torch-geometric tensorflow
pip install xgboost lightgbm shap lime
```

Then start Jupyter:

```bash
jupyter lab
```

---

# 7. Selecting a Dataset

All notebooks use the same environment variable:

```bash
MFV_DATA_DIR
```

Set it to the dataset required by the notebook you are running.

### CICIoT2023

```bash
export MFV_DATA_DIR="/path/to/CICIoT2023"
jupyter lab MFV_GDN_CICIoT2023_GitHub.ipynb
```

### NSL-KDD

```bash
export MFV_DATA_DIR="/path/to/NSL_KDD"
jupyter lab MFV_GDN_NSL_KDD_GitHub.ipynb
```

### UNSW-NB15

```bash
export MFV_DATA_DIR="/path/to/UNSW_NB15"
jupyter lab "MFV_GDN_UNSW_NB15_GitHub(1).ipynb"
```

Optional project/output locations:

```bash
export MFV_PROJECT_ROOT="/path/to/project"
export MFV_OUTPUT_DIR="/path/to/output"
```

---

# 8. Main Runtime Options

The following options are shared across the notebooks:

```bash
export MFV_RUN_NESTED_CV=1
export MFV_RUN_FINAL_TEST=1
export MFV_RUN_COMPARATIVE_MODELS=1
export MFV_RUN_COMPARATIVE_CV=1
export MFV_RUN_EXPLAINABILITY=1
```

Use `0` to disable an optional component.

Example:

```bash
export MFV_RUN_EXPLAINABILITY=0
export MFV_RUN_COMPARATIVE_CV=0
```

---

# 9. Fast Debug Mode

For a quick pipeline check:

```bash
export MFV_GDN_FAST_DEBUG=1
```

Fast Debug Mode reduces expensive settings such as:

- HHO hawks and iterations,
- WGAN-GP epochs,
- GNN epochs,
- MFV-NSA candidate/detector counts,
- inner cross-fitting folds,
- CNN training epochs.

Use it only for debugging.

For final experiments:

```bash
export MFV_GDN_FAST_DEBUG=0
```

---

# 10. Evaluation

Each dataset notebook supports a common evaluation framework.

## Binary metrics

Typical outputs include:

- Accuracy
- Precision
- Recall / Attack Detection Rate
- F1-score
- True Negative Rate
- False Positive Rate
- ROC curve
- ROC-AUC
- Precision-Recall curve

## Multi-class metrics

The notebooks include:

- overall accuracy,
- macro precision,
- macro recall,
- macro F1,
- weighted metrics,
- class-wise performance,
- multi-class ROC analysis.

## Validation protocol

The notebooks support:

```text
Outer folds = 5
Inner cross-fitting folds = 5
```

where applicable, together with final locked-test evaluation.

---

# 11. Comparative Models

The implementation includes comparison models such as:

- MLP
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- AdaBoost
- Gradient Boosting
- Bagging
- Stacking

The notebooks also report important MFV-GDN component results, including:

- CNN-BiLSTM-MHA
- CSA
- Decision Fusion

---

# 12. Explainability

Explainability can be enabled with:

```bash
export MFV_RUN_EXPLAINABILITY=1
```

The notebooks contain:

### SHAP

SHAP-based feature analysis for supported tree-based comparison models.

### LIME

LIME explanations for selected individual predictions.

To skip explainability:

```bash
export MFV_RUN_EXPLAINABILITY=0
```

---

# 13. Adversarial Robustness

Each dataset notebook contains an adversarial evaluation section after clean MFV-GDN evaluation.

Implemented stress tests include:

### Training-time attacks

- label poisoning,
- duplicate injection,
- WGAN/augmentation contamination.

### Inference-time attacks

- feature perturbation,
- FGSM,
- unseen/held-out attack-family evaluation,
- adaptive black-box probing.

### Graph and model attacks

- graph edge rewiring,
- edge dropping,
- node-feature corruption,
- model extraction,
- membership inference,
- prediction/output tampering.

Important adversarial runtime variables include:

```bash
export MFV_ATTACK_EVAL_N=3000
export MFV_ATTACK_PROBE_N=300
export MFV_ATTACK_QUERY_BUDGET=30
export MFV_ATTACK_RETRAIN_EPOCHS=12
```

---

# 14. Generated Outputs

Each notebook stores results in its dataset-specific output directory.

### CICIoT2023

```text
MFV_GDN_CICIoT2023_outputs/
```

### NSL-KDD

```text
MFV_GDN_NSL_KDD_outputs/
```

### UNSW-NB15

```text
MFV_GDN_UNSW_NB15_outputs/
```

Common result files can include:

```text
nested_binary_results.csv
nested_multiclass_results.csv
final_binary_results.csv
final_multiclass_results.csv

wgan_gp_novelty_audit.csv

binary_model_comparison.csv
multiclass_model_comparison.csv
multiclass_5fold_summary.csv

feature_analysis.csv
graph_stats.csv

classwise_<model>.csv
```

Figures are saved under:

```text
<OUTPUT_DIR>/figures/
```

Adversarial results are saved under:

```text
<OUTPUT_DIR>/adversarial_attack_results/
```

with outputs such as:

```text
attack_results.csv
attack_category_summary.csv
attack_results_table.tex
adr_under_attack.png
fpr_under_attack.png
```

---

# 15. Recommended Repository Structure

```text
MFV-GDN/
│
├── README.md
│
├── MFV_GDN_CICIoT2023_GitHub.ipynb
├── MFV_GDN_NSL_KDD_GitHub.ipynb
├── MFV_GDN_UNSW_NB15_GitHub(1).ipynb
│
├── data/
│   ├── CICIoT2023/
│   ├── NSL_KDD/
│   └── UNSW_NB15/
│
└── outputs/
    ├── CICIoT2023/
    ├── NSL_KDD/
    └── UNSW_NB15/
```

The datasets themselves do not need to be uploaded to the GitHub repository. Keep large datasets external and configure their paths using `MFV_DATA_DIR`.

---

# 16. Reproducibility

The notebooks use:

```text
SEED = 42
```

The seed is applied to major random-number generators, including:

- Python `random`,
- NumPy,
- PyTorch,
- TensorFlow.

For reproducible comparisons, keep the following consistent:

- dataset split,
- random seed,
- HHO settings,
- WGAN-GP settings,
- GCN-GAT settings,
- MFV-NSA settings,
- CSA settings,
- CNN-BiLSTM-MHA settings,
- evaluation protocol.

The environment-check section also reports system and GPU information.

---

# 17. Recommended Run Order

For each dataset notebook, run the cells from top to bottom.

```text
1. Environment check
2. Imports and configuration
3. Dataset loading and label engineering
4. Split / duplicate audit
5. Training-only preprocessing
6. HHO feature selection
7. WGAN-GP augmentation
8. Traffic-to-graph construction
9. GCN-GAT
10. MFV-NSA
11. CSA
12. Shared representation
13. CNN-BiLSTM-MHA
14. Decision fusion
15. Nested/final evaluation
16. Class-wise + ROC/PR analysis
17. SHAP/LIME
18. Save results
19. Adversarial robustness evaluation
```

---

# 18. Research Integrity

For publication-quality experiments:

- keep the final test partition locked during model development;
- fit preprocessing only on training-derived data;
- perform HHO feature selection without test information;
- train WGAN-GP only on training samples;
- do not create synthetic validation/test data;
- generate MFV-NSA self sets and detectors from training-derived data;
- do not update training graph parameters with test samples;
- tune thresholds before final test evaluation;
- report metrics directly from executed experiments;
- preserve generated CSV files and experiment settings.

---

# 19. Citation

If you use this repository in academic work, cite the associated MFV-GDN manuscript/publication.

```bibtex
@article{islam_mfvgdn,
  title   = {MFV-GDN: An Immune-Inspired Graph-Deep Neural Network Framework for Intrusion Detection},
  author  = {Islam, Md. Jahidul and others},
  journal = {To be updated},
  year    = {2026},
  note    = {Final publication information to be updated}
}
```

Replace the placeholder journal information with the final publication metadata when available.

---

## Summary

This repository provides a common MFV-GDN implementation across **CICIoT2023, NSL-KDD, and UNSW-NB15** while preserving dataset-specific label structures and train/test organization.

The shared framework integrates:

```text
HHO
+ WGAN-GP
+ GCN-GAT
+ MFV-NSA
+ CSA
+ CNN-BiLSTM-MHA
+ Decision Fusion
+ Explainability
+ Adversarial Evaluation
```

for reproducible binary and multi-class intrusion-detection research.
