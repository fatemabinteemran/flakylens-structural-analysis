import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confidence-gated structural correction of FlakyLens "
            "predictions using nested project-group validation."
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

    timing = [
        "sleep",
        "wait",
        "timeout",
        "eventually_retry_poll",
    ]

    coordination = [
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
        for name in timing
    )

    features["coordination_signal_count"] = sum(
        features[f"{name}_count"]
        for name in coordination
    )

    features["coord_minus_timing"] = (
        features["coordination_signal_count"]
        - features["timing_signal_count"]
    )

    first_assert = first_position(
        code,
        PATTERNS["assertion"],
    )

    for name in ["sleep", "await", "join", "latch"]:
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

    feature_columns = list(feature_df.columns)

    truth = df["Ground_Truth"].to_numpy()
    baseline = df["Prediction"].to_numpy()

    labels = [0, 1, 2, 3, 4, 5]

    baseline_macro = f1_score(
        truth,
        baseline,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    groups = sorted(
        df["project_group"].unique()
    )

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    hybrid_prediction = baseline.copy()

    selection_rows = []

    # --------------------------------------------------
    # OUTER LOOP:
    # one project group remains completely held out
    # --------------------------------------------------

    for outer_group in groups:
        outer_training_groups = [
            group
            for group in groups
            if group != outer_group
        ]

        # ----------------------------------------------
        # INNER LOOP:
        # choose confidence threshold without using
        # the outer held-out project group
        # ----------------------------------------------

        inner_probability = np.full(
            len(df),
            np.nan,
        )

        for inner_group in outer_training_groups:
            model_training_groups = [
                group
                for group in outer_training_groups
                if group != inner_group
            ]

            train_mask = (
                df["project_group"].isin(
                    model_training_groups
                )
                & df["Ground_Truth"].isin([0, 1])
            )

            # Predict probabilities for every test in
            # the inner validation group. Ground truth
            # is not used to decide which prediction
            # receives a possible correction.
            validation_mask = (
                df["project_group"] == inner_group
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

            inner_probability[
                validation_mask
            ] = model.predict_proba(
                feature_df.loc[
                    validation_mask,
                    feature_columns,
                ]
            )[:, 1]

        inner_evaluation_mask = (
            df["project_group"].isin(
                outer_training_groups
            )
        )

        best_threshold = None
        best_macro = -1.0
        best_inner_overrides = 0

        for threshold in thresholds:
            candidate_prediction = (
                baseline.copy()
            )

            correction_mask = (
                inner_evaluation_mask
                & (baseline == 0)
                & (~np.isnan(inner_probability))
                & (
                    inner_probability
                    >= threshold
                )
            )

            candidate_prediction[
                correction_mask
            ] = 1

            score = f1_score(
                truth[
                    inner_evaluation_mask
                ],
                candidate_prediction[
                    inner_evaluation_mask
                ],
                labels=labels,
                average="macro",
                zero_division=0,
            )

            # On ties, choose the more conservative
            # higher threshold.
            if (
                score > best_macro + 1e-12
                or (
                    abs(
                        score - best_macro
                    ) <= 1e-12
                    and (
                        best_threshold is None
                        or threshold
                        > best_threshold
                    )
                )
            ):
                best_macro = score
                best_threshold = threshold
                best_inner_overrides = int(
                    correction_mask.sum()
                )

        # ----------------------------------------------
        # Train final structural model using the outer
        # training groups only.
        # ----------------------------------------------

        final_train_mask = (
            df["project_group"].isin(
                outer_training_groups
            )
            & df["Ground_Truth"].isin([0, 1])
        )

        outer_mask = (
            df["project_group"]
            == outer_group
        )

        model = make_model()

        model.fit(
            feature_df.loc[
                final_train_mask,
                feature_columns,
            ],
            df.loc[
                final_train_mask,
                "Ground_Truth",
            ],
        )

        # Probability is calculated for every test
        # in the held-out project group.
        outer_probability = (
            model.predict_proba(
                feature_df.loc[
                    outer_mask,
                    feature_columns,
                ]
            )[:, 1]
        )

        outer_indices = np.where(
            outer_mask
        )[0]

        # One-way correction only:
        # FlakyLens Async-Wait -> Concurrency
        # when the structural model is confident.
        local_correction = (
            (baseline[outer_mask] == 0)
            & (
                outer_probability
                >= best_threshold
            )
        )

        corrected_indices = (
            outer_indices[
                local_correction
            ]
        )

        hybrid_prediction[
            corrected_indices
        ] = 1

        if len(corrected_indices):
            corrected_truth = truth[
                corrected_indices
            ]

            true_concurrency = int(
                (
                    corrected_truth
                    == 1
                ).sum()
            )

            true_async = int(
                (
                    corrected_truth
                    == 0
                ).sum()
            )

            other_classes = int(
                (
                    ~np.isin(
                        corrected_truth,
                        [0, 1],
                    )
                ).sum()
            )
        else:
            true_concurrency = 0
            true_async = 0
            other_classes = 0

        selection_rows.append(
            {
                "held_out_group":
                    outer_group,
                "selected_threshold":
                    best_threshold,
                "inner_macro_f1":
                    best_macro,
                "inner_overrides":
                    best_inner_overrides,
                "heldout_overrides":
                    len(corrected_indices),
                "heldout_true_concurrency":
                    true_concurrency,
                "heldout_true_async":
                    true_async,
                "heldout_other_classes":
                    other_classes,
            }
        )

    hybrid_macro = f1_score(
        truth,
        hybrid_prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    print(
        "=== NESTED PROJECT-GROUP "
        "CONFIDENCE-GATED HYBRID ==="
    )

    selection_df = pd.DataFrame(
        selection_rows
    )

    print(
        selection_df.to_string(
            index=False
        )
    )

    print(
        "\n=== FINAL COMPARISON ==="
    )

    print(
        "FlakyLens Macro-F1:",
        round(
            baseline_macro,
            4,
        ),
    )

    print(
        "Hybrid Macro-F1:",
        round(
            hybrid_macro,
            4,
        ),
    )

    print(
        "Macro-F1 delta:",
        round(
            hybrid_macro
            - baseline_macro,
            4,
        ),
    )

    class_names = {
        0: "Async-Wait",
        1: "Concurrency",
        2: "Time",
        3: "Unordered Collections",
        4: "Order Dependency",
        5: "Non-flaky",
    }

    result_rows = []

    for label, name in class_names.items():
        before = f1_score(
            truth == label,
            baseline == label,
            zero_division=0,
        )

        after = f1_score(
            truth == label,
            hybrid_prediction == label,
            zero_division=0,
        )

        result_rows.append(
            {
                "category": name,
                "baseline_f1": before,
                "hybrid_f1": after,
                "delta": after - before,
            }
        )

        print(
            f"{name:24s} "
            f"before={before:.4f} "
            f"after={after:.4f} "
            f"delta={after-before:+.4f}"
        )

    changed = (
        hybrid_prediction
        != baseline
    )

    print(
        "\n=== OVERRIDE SUMMARY ==="
    )

    print(
        "Total predictions changed:",
        int(changed.sum()),
    )

    print(
        "Correct Concurrency recoveries:",
        int(
            (
                truth[changed]
                == 1
            ).sum()
        ),
    )

    print(
        "True Async-Wait harmed:",
        int(
            (
                truth[changed]
                == 0
            ).sum()
        ),
    )

    print(
        "Other true classes changed:",
        int(
            (
                ~np.isin(
                    truth[changed],
                    [0, 1],
                )
            ).sum()
        ),
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    selection_df.to_csv(
        output_dir
        / "nested_threshold_selection.csv",
        index=False,
    )

    pd.DataFrame(
        result_rows
    ).to_csv(
        output_dir
        / "nested_hybrid_class_results.csv",
        index=False,
    )

    overall = pd.DataFrame(
        [
            {
                "method":
                    "FlakyLens",
                "macro_f1":
                    baseline_macro,
            },
            {
                "method":
                    "Structural_Hybrid",
                "macro_f1":
                    hybrid_macro,
            },
        ]
    )

    overall.to_csv(
        output_dir
        / "nested_hybrid_overall.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        output_dir
        / "nested_threshold_selection.csv"
    )
    print(
        output_dir
        / "nested_hybrid_class_results.csv"
    )
    print(
        output_dir
        / "nested_hybrid_overall.csv"
    )


if __name__ == "__main__":
    main()
