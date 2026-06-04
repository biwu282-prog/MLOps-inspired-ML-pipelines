# MLOps-Inspired Machine Learning Pipelines

This repository implements two structured machine learning workflows inspired
by selected MLOps lifecycle stages. It focuses on data preparation, feature or
image preprocessing, model training, and model evaluation. It is not a full
production MLOps system and does not include deployment, online monitoring,
CI/CD, drift detection, or automated retraining.

## Repository Structure

```text
classical/
  data/                 Small barley dataset
  cleaning/             Manual and automated cleaning workflows
  feature_engineering/  Leakage-safe next-year prediction features
  training/             Linear Regression, Ridge, and SVR training
  analysis/             Result-analysis notebook

deep_learning/
  data/                 Animals-10 download instructions
  training/             ResNet18 and VGG16 training pipeline
  analysis/             Result-analysis notebook

artifacts/               Generated outputs; ignored by Git
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Classical Machine Learning Pipeline

The classical branch predicts next-year barley yield using annual agricultural
and climate history. Features for year `t` use only information available
before year `t`.

### 1. Clean the data

The final classical workflow uses the configurable manual-cleaning pipeline
with the following settings:

```bash
python classical/cleaning/manual_clean.py \
  --file classical/data/barley_data.csv \
  --output artifacts/classical/cleaned \
  --outlier-detection zscore \
  --z-threshold 3.5 \
  --fix-method median \
  --smooth-method rolling \
  --window 5 \
  --no-plot
```

The feature-engineering pipeline expects:

```text
artifacts/classical/cleaned/barley_data_cleaned.csv
```

An automated cleaning implementation is also included as an alternative:

```bash
python classical/cleaning/auto_clean.py \
  --file classical/data/barley_data.csv \
  --output artifacts/classical/cleaned
```

To build features from its output, explicitly pass:

```bash
python classical/feature_engineering/feature_engineering.py \
  --input artifacts/classical/cleaned/barley_data_auto_cleaned.csv
```

### 2. Build features

```bash
python classical/feature_engineering/feature_engineering.py
```

This stage performs chronological splitting, training-only correlation
selection, and training-only StandardScaler fitting.

### 3. Train models

```bash
python classical/training/train_models.py
```

The script trains:

- Linear Regression
- Ridge Regression
- Support Vector Regression

### 4. Analyse results

Open and run:

```text
classical/analysis/results_analysis.ipynb
```

## Deep Learning Pipeline

The deep-learning branch applies the same process-centred workflow to Animals-10
image classification. It trains and evaluates pretrained ResNet18 and VGG16
models.

### 1. Prepare Animals-10

Follow the instructions in:

```text
deep_learning/data/README.md
```

### 2. Train models

```bash
python deep_learning/training/train_models.py
```

For a quick data-preparation check without model training:

```bash
python deep_learning/training/train_models.py --dry-run
```

### 3. Analyse results

Open and run:

```text
deep_learning/analysis/results_analysis.ipynb
```

## Generated Artifacts

The scripts generate metrics, predictions, metadata, plots, classification
reports, confusion matrices, and model checkpoints under `artifacts/`.
Generated artifacts are intentionally excluded from version control.

## Dataset Sources

- Agricultural and climate data: prepared from FAOSTAT and NASA POWER data.
- Animals-10: https://www.kaggle.com/datasets/alessiocorrado99/animals10

## Project Scope

The purpose of this repository is to demonstrate reproducible implementation of
selected machine learning lifecycle stages across structured numerical data and
image data. The emphasis is on the process and practical findings rather than
proposing a new machine learning model.
