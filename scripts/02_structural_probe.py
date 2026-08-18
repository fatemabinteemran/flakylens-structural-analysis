import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project-group-held-out structural probe for "
            "FlakyLens Async-Wait vs Concurrency classification."
        )
    )
    parser.add_argument(
        "--flakylens-root",
        default="../FlakyLens",
        help="Path to the official FlakyLens repository.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for generated results.",
    )
    return parser.parse_args()


PATTERNS = {
    "sleep": r"\bThread\s*\.\s*sleep\s*\(",
    "wait": r"\.wait\s*\(",
    "await": r"\bawait\w*\s*\(",
    "join": r"\.join\s*\(",
    "timeout": r"\btimeout\b|\bTimeUnit\b",
    "eventually_retry_poll":
        r"\beventually\b|\bretry\b|\bpoll\w*\b",

    "synchronized": r"\bsynchronized\b",
    "atomic":
        r"\bAtomic(?:Integer|Long|Boolean|Reference|"
        r"ReferenceArray|IntegerArray|LongArray)?\b",
    "executor":
        r"\bExecutor(?:Service)?\b|\bExecutors\b",
    "thread": r"\bThread\b",
    "future": r"\bFuture\b|\bCompletableFuture\b",
    "latch": r"\bCountDownLatch\b",
    "lock":
        r"\bReentrantLock\b|\bReadWriteLock\b|\bLock\b",
    "semaphore": r"\bSemaphore\b",
    "volatile": r"\bvolatile\b",
    "submit_execute":
        r"\.(?:submit|execute|invokeAll|invokeAny)\s*\(",
    "parallel": r"\bparallel\w*\b",

    "assertion":
        r"\bassert\w*\s*\(|\bAssert\.",
}


def count_pattern(code, pattern):
    return len(
        re.findall(
            pattern,
            str(code),
            flags=re.I,
        )
    )


def first_position(code, pattern):
    match = re.search(
        pattern,
        str(code),
        flags=re.I,
    )
    return match.start() if match else -1


def extract_features(code):
    code = str(code)
    features = {}

    for name, pattern in PATTERNS.items():
        count = count_pattern(code, pattern)

        features[f"{name}_count"] = count
        features[f"{name}_present"] = int(count > 0)

    timing_features = [
        "sleep",
        "wait",
        "timeout",
        "eventually_retry_poll",
    ]

    coordination_features = [
        "await",
        "join",
        "synchronized",
        "atomic",
        "executor",
        "thread",
        "future",
        "latch",
        "lock",
        "semaphore",
        "volatile",
        "submit_execute",
        "parallel",
    ]

    features["timing_signal_count"] = sum(
        features[f"{name}_count"]
        for name in timing_features
    )

    features["coordination_signal_count"] = sum(
        features[f"{name}_count"]
        for name in coordination_features
    )

    features["coord_minus_timing"] = (
        features["coordination_signal_count"]
        - features["timing_signal_count"]
    )

    first_assert = first_position(
        code,
        PATTERNS["assertion"],
    )

    for name in [
        "sleep",
        "await",
        "join",
        "latch",
    ]:
        pos = first_position(
            code,
            PATTERNS[name],
        )

        features[
            f"{name}_before_first_assert"
        ] = int(
            pos >= 0
            and first_assert >= 0
            and pos < first_assert
        )

    features["code_length"] = len(code)

    return features


def make_model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    args = parse_args()

    root = Path(args.flakylens_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_file = (
        root
        / "src"
        / "FlakyLens_Categorization_PerProject-result"
        / "Finetuned_Result_with_tokens.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    df = pd.read_csv(prediction_file)

    feature_df = pd.DataFrame(
        [
            extract_features(code)
            for code in df["test_code"]
        ]
    )

    feature_columns = list(
        feature_df.columns
    )

    # Focus on the two categories involved in the
    # strongest reported confusion.
    binary_mask = df["Ground_Truth"].isin(
        [0, 1]
    )

    y_true = df.loc[
        binary_mask,
        "Ground_Truth",
    ].to_numpy()

    flakylens_pred = df.loc[
        binary_mask,
        "Prediction",
    ].to_numpy()

    print(
        "=== FLAKYLENS BASELINE: "
        "ASYNC-WAIT VS CONCURRENCY ==="
    )

    print(
        "Samples:",
        len(y_true),
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_true,
                flakylens_pred,
            ),
            4,
        ),
    )

    baseline_macro_f1 = f1_score(
        y_true,
        flakylens_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    baseline_recall = recall_score(
        y_true,
        flakylens_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )[1]

    print(
        "Macro-F1:",
        round(
            baseline_macro_f1,
            4,
        ),
    )

    print(
        "Concurrency recall:",
        round(
            baseline_recall,
            4,
        ),
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            flakylens_pred,
            labels=[0, 1],
        )
    )

    # --------------------------------------------------
    # Project-group-held-out structural probe
    # --------------------------------------------------

    structural_prediction = np.full(
        len(df),
        -1,
        dtype=int,
    )

    groups = sorted(
        df["project_group"].unique()
    )

    per_group_rows = []

    for held_out_group in groups:
        train_mask = (
            (df["project_group"] != held_out_group)
            & df["Ground_Truth"].isin([0, 1])
        )

        test_mask = (
            (df["project_group"] == held_out_group)
            & df["Ground_Truth"].isin([0, 1])
        )

        model = make_model()

        model.fit(
            feature_df.loc[
                train_mask,
                feature_columns,
            ],
            df.loc[
                train_mask,
                "Ground_Truth",
            ],
        )

        predictions = model.predict(
            feature_df.loc[
                test_mask,
                feature_columns,
            ]
        )

        structural_prediction[
            test_mask
        ] = predictions

        group_truth = df.loc[
            test_mask,
            "Ground_Truth",
        ].to_numpy()

        group_macro = f1_score(
            group_truth,
            predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )

        per_group_rows.append(
            {
                "held_out_group":
                    held_out_group,
                "samples":
                    int(test_mask.sum()),
                "macro_f1":
                    group_macro,
            }
        )

    structural_binary_pred = (
        structural_prediction[
            binary_mask
        ]
    )

    structural_accuracy = accuracy_score(
        y_true,
        structural_binary_pred,
    )

    structural_macro_f1 = f1_score(
        y_true,
        structural_binary_pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    structural_recall = recall_score(
        y_true,
        structural_binary_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )[1]

    print(
        "\n=== STRUCTURAL PROBE: "
        "PROJECT-GROUP HELD-OUT ==="
    )

    print(
        "Accuracy:",
        round(
            structural_accuracy,
            4,
        ),
    )

    print(
        "Macro-F1:",
        round(
            structural_macro_f1,
            4,
        ),
    )

    print(
        "Concurrency recall:",
        round(
            structural_recall,
            4,
        ),
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            structural_binary_pred,
            labels=[0, 1],
        )
    )

    print(
        "\n=== CHANGE FROM FLAKYLENS ==="
    )

    print(
        "Macro-F1 delta:",
        round(
            structural_macro_f1
            - baseline_macro_f1,
            4,
        ),
    )

    print(
        "Concurrency recall delta:",
        round(
            structural_recall
            - baseline_recall,
            4,
        ),
    )

    summary = pd.DataFrame(
        [
            {
                "method":
                    "FlakyLens",
                "accuracy":
                    accuracy_score(
                        y_true,
                        flakylens_pred,
                    ),
                "macro_f1":
                    baseline_macro_f1,
                "concurrency_recall":
                    baseline_recall,
            },
            {
                "method":
                    "Structural_Probe",
                "accuracy":
                    structural_accuracy,
                "macro_f1":
                    structural_macro_f1,
                "concurrency_recall":
                    structural_recall,
            },
        ]
    )

    summary.to_csv(
        output_dir
        / "structural_probe_summary.csv",
        index=False,
    )

    pd.DataFrame(
        per_group_rows
    ).to_csv(
        output_dir
        / "structural_probe_per_group.csv",
        index=False,
    )

    print(
        "\nSaved:"
    )
    print(
        output_dir
        / "structural_probe_summary.csv"
    )
    print(
        output_dir
        / "structural_probe_per_group.csv"
    )


if __name__ == "__main__":
    main()
