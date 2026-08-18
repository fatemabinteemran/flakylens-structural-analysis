import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze FlakyLens Concurrency vs Async-Wait errors."
    )
    parser.add_argument(
        "--flakylens-root",
        default="../FlakyLens",
        help="Path to the official FlakyLens repository.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for generated analysis results.",
    )
    return parser.parse_args()


def has_pattern(code, pattern):
    return int(bool(re.search(pattern, str(code), flags=re.I)))


def main():
    args = parse_args()

    flakylens_root = Path(args.flakylens_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_file = (
        flakylens_root
        / "src"
        / "FlakyLens_Categorization_PerProject-result"
        / "Finetuned_Result_with_tokens.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    df = pd.read_csv(prediction_file)

    # FlakyLens category mapping:
    # 0 = Async-Wait
    # 1 = Concurrency
    # 2 = Time
    # 3 = Unordered Collections
    # 4 = Test Order Dependency
    # 5 = Non-flaky

    concurrency = df[df["Ground_Truth"] == 1].copy()

    correct = concurrency[concurrency["Prediction"] == 1]
    as_async = concurrency[concurrency["Prediction"] == 0]
    other_errors = concurrency[
        ~concurrency["Prediction"].isin([0, 1])
    ]

    print("=== FLAKYLENS CONCURRENCY ERROR ANALYSIS ===")
    print(f"Total actual Concurrency: {len(concurrency)}")
    print(f"Correctly predicted Concurrency: {len(correct)}")
    print(f"Concurrency -> Async-Wait: {len(as_async)}")
    print(f"Concurrency -> other classes: {len(other_errors)}")

    if len(concurrency):
        pct = 100 * len(as_async) / len(concurrency)
        print(
            f"Concurrency misclassified as Async-Wait: {pct:.1f}%"
        )

    patterns = {
        # Timing/wait-oriented signals
        "sleep": r"\bThread\s*\.\s*sleep\s*\(",
        "wait": r"\.wait\s*\(",
        "await": r"\bawait\w*\s*\(",
        "join": r"\.join\s*\(",
        "timeout": r"\btimeout\b|\bTimeUnit\b",
        "eventually_retry_poll":
            r"\beventually\b|\bretry\b|\bpoll\w*\b",

        # Concurrency/synchronization signals
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
    }

    for feature, pattern in patterns.items():
        concurrency[feature] = concurrency["test_code"].map(
            lambda code: has_pattern(code, pattern)
        )

    concurrency["group"] = concurrency["Prediction"].map(
        lambda p: (
            "Correct_Concurrency"
            if p == 1
            else (
                "Misclassified_As_Async"
                if p == 0
                else "Other_Misclassification"
            )
        )
    )

    focus = concurrency[
        concurrency["group"].isin(
            ["Correct_Concurrency", "Misclassified_As_Async"]
        )
    ].copy()

    rows = []

    for feature in patterns:
        for group in [
            "Correct_Concurrency",
            "Misclassified_As_Async",
        ]:
            subset = focus[focus["group"] == group][feature]

            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "count": int(subset.sum()),
                    "total": len(subset),
                    "percent": round(100 * subset.mean(), 1),
                }
            )

    summary = pd.DataFrame(rows)

    pivot = summary.pivot(
        index="feature",
        columns="group",
        values="percent",
    ).fillna(0)

    pivot["Difference_Correct_minus_Async"] = (
        pivot["Correct_Concurrency"]
        - pivot["Misclassified_As_Async"]
    )

    pivot = pivot.sort_values(
        "Difference_Correct_minus_Async",
        ascending=False,
    )

    print("\n=== FEATURE PRESENCE (%) ===")
    print(pivot.to_string())

    pivot.to_csv(
        output_dir / "concurrency_feature_comparison.csv"
    )

    concurrency.to_csv(
        output_dir / "concurrency_cases_with_features.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        output_dir / "concurrency_feature_comparison.csv"
    )
    print(
        output_dir / "concurrency_cases_with_features.csv"
    )


if __name__ == "__main__":
    main()
