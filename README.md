# Structural Analysis of FlakyLens Concurrency Classification

This repository contains a small preliminary extension and error analysis of
the FlakyLens artifact from the paper:

**Understanding and Improving Flaky Test Classification**

Official FlakyLens repository:
https://github.com/UT-SE-Research/FlakyLens

The goal of this investigation was to examine the reported difficulty in
distinguishing **Concurrency** from **Async-Wait** flaky tests and to test
whether explicit program-structure and synchronization signals provide
complementary information to the learned code representation.

## Motivation

Using the prediction file provided with the FlakyLens artifact, I observed:

- 37 tests have Concurrency as the ground-truth category.
- 12 are correctly classified as Concurrency.
- 20 are classified as Async-Wait.
- 5 are classified as other categories.

Thus, 20/37 (54.1%) of the Concurrency cases are classified as Async-Wait in
the provided predictions.

An exploratory comparison suggested differences in synchronization and timing
constructs between correctly classified and misclassified Concurrency tests.

Examples include:

| Feature | Correct Concurrency | Concurrency -> Async-Wait |
|---|---:|---:|
| `await` | 50.0% | 15.0% |
| `CountDownLatch` | 33.3% | 0.0% |
| `join` | 33.3% | 5.0% |
| `Thread.sleep` | 16.7% | 50.0% |
| timeout-related signals | 8.3% | 35.0% |

These percentages are exploratory observations from a small number of
Concurrency tests and are not intended as general conclusions.

## Experiment 1: Structural Probe

I extracted lightweight structural features related to timing and concurrency,
including occurrences and positions of constructs such as:

`Thread.sleep`, `await`, `join`, `CountDownLatch`, `Atomic*`, `Thread`,
`Executor`, `Future`, timeout-related constructs, and other synchronization
signals.

A logistic-regression probe was evaluated using the project groups supplied by
the FlakyLens artifact. Each project group was held out in turn.

For the Async-Wait vs Concurrency subset:

| Metric | FlakyLens | Structural Probe |
|---|---:|---:|
| Macro-F1 | 0.5568 | 0.5995 |
| Concurrency Recall | 0.3243 | 0.4324 |

This suggests that explicit structural information contains complementary
signal for distinguishing these two categories.

## Experiment 2: Confidence-Gated Hybrid

I next tested whether the structural signal could complement the original
FlakyLens predictions.

The structural model does not replace the six-class classifier. It is used only
as a conservative second-stage signal when FlakyLens predicts Async-Wait.

A prediction is changed from **Async-Wait -> Concurrency** only when the
structural model predicts Concurrency above a confidence threshold.

Threshold selection uses nested project-group validation: the held-out project
group is not used to choose its threshold.

Final results:

| Metric | FlakyLens | Hybrid |
|---|---:|---:|
| Overall Macro-F1 | 0.6768 | 0.6812 |
| Concurrency F1 | 0.3582 | 0.3944 |
| Async-Wait F1 | 0.5988 | 0.5890 |

The hybrid changed four predictions:

- 2 true Concurrency cases were recovered.
- 2 true Async-Wait cases were incorrectly changed.
- No other categories were modified.

Therefore, this should be viewed as a **preliminary indication of complementary
structural information**, not as a complete solution.

A more principled integration of learned code representations with structural
or semantic program information may be worth investigating further.

## Repository Structure

```text
scripts/
  01_error_analysis.py
  02_structural_probe.py
  03_nested_hybrid.py

results/
  concurrency_feature_comparison.csv
  structural_probe_summary.csv
  structural_probe_per_group.csv
  nested_threshold_selection.csv
  nested_hybrid_class_results.csv
  nested_hybrid_overall.csv

```

## Setup

Clone the official FlakyLens repository and this repository into the same parent directory:

```text
flaky-research/
  FlakyLens/
  flakylens-structural-analysis/
```

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The experiments use the prediction artifact provided by FlakyLens:

```text
FlakyLens/src/FlakyLens_Categorization_PerProject-result/Finetuned_Result_with_tokens.csv
```

No FlakyLens model retraining is required for these analyses.

## Reproduction

Run from this repository:

```bash
python scripts/01_error_analysis.py --flakylens-root ../FlakyLens
python scripts/02_structural_probe.py --flakylens-root ../FlakyLens
python scripts/03_nested_hybrid.py --flakylens-root ../FlakyLens
```

## Scope

This is a preliminary exploratory investigation based on the FlakyLens artifact. The structural features and hybrid strategy are intended to test whether explicit program-structure information provides complementary signal to the original learned representation.

The current results should not be interpreted as a complete replacement for FlakyLens or as evidence of a general solution. The final confidence-gated hybrid changed only four predictions: two Concurrency cases were recovered, while two Async-Wait cases were incorrectly changed.
