# KDIL

Implementation of a knowledge distillation-based incremental learning framework for next-activity prediction under unseen events.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the raw CSV event log at:

```text
data/<dataset>/<dataset>.csv
```

The CSV file must contain the following columns:

```text
case:concept:name
concept:name
time:timestamp
```

Run preprocessing:

```bash
python DataPreProcess.py --dataset BPIC15_2
```

The processed file will be saved to:

```text
data/BPIC15_2/processed/prefixes.csv
```

## Run KDIL

Run online prediction and adaptation with monthly windows:

```bash
python KDTest.py --dataset BPIC15_2 --window_type month
```

Supported window types:

```text
day
week
month
```

## Save Results

Save the evaluation results to Excel:

```bash
python KDTest.py --dataset BPIC15_2 --window_type month --save_excel True
```

## Disable Console Output

Disable detailed console output:

```bash
python KDTest.py --dataset BPIC15_2 --window_type month --no-verbose
```
