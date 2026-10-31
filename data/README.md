# Dataset

## Overview

This project uses the **Bank Marketing Prediction Dataset**, provided through a Kaggle competition.

The dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution. The main objective is to predict whether a client will subscribe to a term deposit based on demographic, financial, and campaign-related features.

## Dataset Source

The dataset is available on Kaggle:

**[Bank Marketing Prediction – Kaggle Competition](https://www.kaggle.com/competitions/bank-marketing-prediction)**

## Download Instructions

1. Visit the Kaggle competition page using the link above.
2. Sign in to your Kaggle account.
3. Join the competition and accept its rules if required.
4. Navigate to the **Data** section.
5. Download the available dataset files.
6. Extract the files and place `train.csv` and `test.csv` inside `data/raw/`.

The project keeps generated datasets in `data/processed/` and prediction files in
`data/submission/`. The raw and generated CSV files are intentionally ignored by
Git; only the folder layout and documentation are tracked.

**Note:** The dataset is not included in this repository. Users must download it directly from Kaggle before running the project notebooks or scripts.