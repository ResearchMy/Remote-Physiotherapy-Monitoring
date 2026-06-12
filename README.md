# A Machine Learning Framework for Human Activity Recognition and Confidence-Based Movement Deviation Assessment for Remote Physiotherapy Monitoring

This repository contains the implementation accompanying the research paper on Human Activity Recognition (HAR), confidence-based movement deviation detection, severity scoring, and remote physiotherapy monitoring.

## Features

- Human Activity Recognition using Random Forest
- Confidence-based potential deviation detection
- Severity scoring and movement quality assessment
- Severity level classification (Good, Moderate, Needs Improvement)
- Rule-based physiotherapy recommendations
- ROC-AUC, Precision-Recall, and Feature Importance analysis

## Dataset

Human Activity Recognition Using Smartphones (HAR) Dataset:

https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

(Original source: UCI Machine Learning Repository)

## Repository Contents

- `banurp1.py` – Python implementation
- `BanuRP1.ipynb` – Jupyter Notebook implementation

## Installation

```bash
pip install -r requirements.txt
```

## Note

This study is based on a benchmark HAR dataset. The proposed deviation detection and severity scoring mechanisms are computational indicators and have not yet been clinically validated using physiotherapy-specific datasets or patient populations.
