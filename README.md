# Stock Return Prediction V3

An end-to-end machine learning project for predicting stock returns from anonymous factor data.

## Project overview

The dataset contains anonymous stock codes, integer dates, 20 factor features, and a return target `y`. The competition evaluates predictions using Rank Information Coefficient (Rank IC), so the workflow focuses on cross-sectional ranking quality rather than only minimizing RMSE.

## Approach

- Time-aware train/validation split
- Cross-sectional z-score features computed by date
- Quantile binning of continuous factors
- Ordered target encoding for stock codes and binned factors
- LightGBM regression with L1 objective
- Daily Rank IC validation
- Kaggle submission generation

## Best experiment

The current best local model combines continuous and binned factor representations with Ordered Target Encoding and LightGBM.

```text
Local validation Mean Daily Rank IC: 0.114162
Kaggle score: 0.08744
```

The Kaggle score was obtained after the competition deadline and is included as an experimental benchmark.

## Repository contents

```text
pred.ipynb
README.md
requirements.txt
.gitignore
```

The competition data and generated submission files are intentionally excluded from this repository because of their size and competition-data restrictions.

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the competition data from Kaggle and place the following files in the local data directory used by the notebook:

```text
train_data.pkl
test_data.pkl
sample_submission.csv
```

Then open `pred.ipynb` and run the cells from top to bottom.

## Resume bullet

Built a time-aware stock return prediction pipeline using cross-sectional factor normalization, quantile binning, ordered target encoding, and LightGBM; achieved a 0.08744 Kaggle Rank IC score on an anonymous factor prediction competition.
