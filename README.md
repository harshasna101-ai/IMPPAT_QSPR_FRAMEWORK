# IMPPAT Topological Descriptor-Based QSPR Modeling

## Overview

This repository contains the Python implementation of a reproducible Quantitative Structure–Property Relationship (QSPR) pipeline developed using the **IMPPAT phytochemical dataset**.

The workflow investigates whether a compact set of topological molecular descriptors can be used to predict important physicochemical properties and the weighted Quantitative Estimate of Drug-likeness (**QEDw**) of phytochemical compounds.

The study uses a leakage-controlled workflow with descriptor reduction, ensemble machine learning, five-fold cross-validation, blind external validation, Y-randomization, applicability-domain analysis, and an independent RDKit descriptor benchmark.

---

## Dataset

The workflow is based on **1,335 IMPPAT phytochemical compounds**.

The dataset is divided into:

* **Working set:** 1,202 compounds
* **Blind external validation set:** 133 compounds

The data assembly stage merges the physicochemical-property and topological-index sources into a validated modeling table. A fixed random seed is used to ensure reproducibility of downstream splits and analyses.

The pipeline deliberately excludes `Monoisotopic_Mass` from the modeling feature matrix because of its potential direct relationship with Molecular Weight and associated target leakage risk.

---

## Workflow

```text
IMPPAT Data
     │
     ▼
Data Assembly & Validation
     │
     ▼
44 Topological Descriptors
     │
     ▼
Zero-Variance Filtering
     │
     ▼
Iterative VIF Pruning
     │
     ▼
9 Final Topological Descriptors
     │
     ▼
RF + GB + XGBoost Ensemble
     │
     ├── 5-Fold Cross-Validation
     │
     └── Blind External Validation
     │
     ▼
Model Diagnostics
     │
     ├── Observed vs Predicted
     └── Residual Analysis
     │
     ▼
Y-Randomization
     │
     ▼
Williams Applicability Domain
     │
     ▼
RDKit Descriptor Benchmark
     │
     ▼
QEDw Modeling
     │
     ▼
Chemical-Space Prioritization
     │
     ▼
Final Integrated Analysis
```

---

## Descriptor Selection

The initial topological descriptor panel contains **44 descriptors**.

Descriptor quality control consists of:

1. Zero-variance descriptor removal
2. Iterative Variance Inflation Factor (VIF) pruning
3. Selection of a compact non-redundant descriptor panel

The VIF procedure uses standardized descriptors for numerical stability while computing VIF values.

The final topological panel contains **9 descriptors**:

* Narumi-Katayama index
* Multiplicative Zagreb1
* Mostar index
* Szeged index
* Spectral radius
* Average eccentricity
* Sigma index
* Balaban J index
* Multiplicative Zagreb2

---

## Machine Learning Models

Three ensemble regression approaches are evaluated:

* **Random Forest (RF)**
* **Gradient Boosting (GB)**
* **XGBoost (XGB)**

An ensemble prediction is obtained by averaging the predictions from the three models.

The main modeling workflow uses:

* 300 estimators
* Five-fold cross-validation
* Random seed = 42

The scaler and models are fitted only on the internal training portion of each fold to maintain a leakage-free validation procedure.

---

## Predicted Properties

The main QSPR models predict eight physicochemical properties:

1. Molecular Weight
2. Polar Area (TPSA)
3. Complexity (BertzCT)
4. XLogP (Crippen)
5. Heavy-Atom Count
6. H-Bond Donor Count
7. H-Bond Acceptor Count
8. Rotatable-Bond Count

QEDw is subsequently modeled as an additional drug-likeness endpoint using the same final 9-descriptor topological panel.

---

## Validation and Reliability Analysis

### Five-Fold Cross-Validation

Five-fold cross-validation is performed on the working set to obtain out-of-fold predictions and evaluate model performance without using held-out observations during model fitting.

### Blind External Validation

A separate set of **133 compounds** is retained as a blind external validation set and is not used for model fitting.

### Y-Randomization

Y-randomization tests whether model performance could arise from chance correlations.

The implementation performs:

* 200 target permutations
* Fixed 80/20 train/holdout split
* Random state = 42
* 30 estimators for RF, GB, and XGB

The reduced estimator count and fixed split are documented as a computational simplification for the permutation analysis.

### Applicability Domain

Williams-plot analysis evaluates:

* Leverage
* Standardized residuals
* High-leverage compounds
* Response outliers
* Compounds outside the applicability domain

The leverage threshold is calculated using:

`h* = 3(p + 1) / n`

where `p` is the number of final descriptors and `n` is the working-set size.

---

## RDKit Benchmark

An independent RDKit descriptor panel is generated for all 1,335 compounds.

Descriptors that are directly equivalent to the eight target properties are deliberately excluded from the candidate RDKit descriptor pool to provide a fair comparison with the topological descriptor model.

The RDKit descriptors undergo:

* Zero-variance filtering
* Iterative VIF reduction
* Ensemble modeling
* Five-fold cross-validation
* Blind validation

The resulting RDKit model is used as a benchmark against the topological descriptor approach.

---

## QEDw Modeling

QEDw is treated as a composite drug-likeness endpoint.

The same nine topological descriptors are used to model QEDw using:

* RF
* Gradient Boosting
* XGBoost
* Five-fold cross-validation
* Blind validation
* Y-randomization
* Williams-plot applicability-domain analysis

---

## Compound Prioritization

The final prioritization stage combines:

* QEDw
* Chemical-space novelty

Structural novelty is calculated using **ECFP4/Morgan fingerprints** and Tanimoto similarity to the five nearest neighbors within the 1,335-compound dataset.

The priority score is defined as:

```text
Priority Score =
0.5 × QEDw percentile rank
+
0.5 × Tanimoto novelty percentile rank
```

This produces a ranked list of structurally interesting and drug-like phytochemical candidates.

---

## Python Scripts

| Script                         | Purpose                                               |
| ------------------------------ | ----------------------------------------------------- |
| `step1_data_assembly.py`       | Data merging, validation and reproducibility manifest |
| `step3_4_pipeline.py`          | Zero-variance filtering and VIF descriptor reduction  |
| `step9_10_11_modeling.py`      | RF/GB/XGB modeling, 5-fold CV and blind validation    |
| `step12_13_diagnostics.py`     | Observed-vs-predicted plots and residual diagnostics  |
| `step14_yrandomization.py`     | Y-randomization analysis                              |
| `step15_williams_ad.py`        | Williams applicability-domain analysis                |
| `step16a_rdkit_descriptors.py` | RDKit descriptor generation                           |
| `step16b_rdkit_vif.py`         | RDKit descriptor VIF reduction                        |
| `step16c_rdkit_models.py`      | RDKit benchmark modeling                              |
| `step17_best_model.py`         | Best-model selection using blind-set performance      |
| `step18_qedw_modelling.py`     | QEDw modeling and validation                          |
| `step19_prioritization_v2.py`  | Chemical-space novelty and compound prioritization    |
| `step20_final_integration.py`  | Final integration of the complete analysis            |
| `bootstrap_blind_r2_full.py`   | Bootstrap confidence intervals for blind-set R²       |
| `yrand_one_property.py`        | Y-randomization for an individual property            |
| `yrand_aggregate.py`           | Aggregation of Y-randomization results                |

---

## Requirements

The scripts use Python scientific and machine-learning libraries including:

```text
numpy
pandas
scikit-learn
scipy
statsmodels
matplotlib
openpyxl
xgboost
rdkit
```

Install the commonly used dependencies with:

```bash
pip install numpy pandas scikit-learn scipy statsmodels matplotlib openpyxl xgboost
```

**RDKit** should be installed using an appropriate RDKit/Conda environment if it is not available through your existing Python installation.

---

## Reproducibility

The primary workflow uses a fixed random seed:

```text
Random seed = 42
```

This seed is used to make dataset splitting, cross-validation, bootstrapping and permutation-based analyses reproducible.

The data assembly stage also records dataset information and file hashes in a manifest for downstream reproducibility.

---

## Running the Pipeline

The scripts are designed as sequential stages. Run them according to their step numbers, ensuring that the required input files generated by earlier stages are available before executing later stages.

Example:

```bash
python step1_data_assembly.py
python step3_4_pipeline.py
python step9_10_11_modeling.py
python step12_13_diagnostics.py
python step14_yrandomization.py
python step15_williams_ad.py
```

The RDKit benchmark and subsequent analysis can then be executed using the corresponding Step 16–20 scripts.

---

## Important Note

The repository contains the **analysis and modeling code**. Input datasets, generated Excel workbooks, figures, and other potentially large or restricted files are not included unless explicitly permitted.

Several scripts currently expect input files to exist in predefined local directories. These paths may need to be modified when running the code on another system.

---

## Project Summary

This project implements a reproducible QSPR framework for IMPPAT phytochemicals by combining compact topological descriptors with ensemble machine learning. The workflow emphasizes descriptor reduction, leakage-controlled validation, independent blind testing, statistical robustness checks, applicability-domain assessment, and comparison with an independent RDKit descriptor representation.
