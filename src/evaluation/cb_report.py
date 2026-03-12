import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.cb_evaluator import (
    MODEL_REGISTRY,
    KS,
    RELEVANCE_THRESHOLD,
    load_model,
    evaluate_model,
)
from src.evaluation.metrics import aggregate_metrics
from src.logging_utils.logger import logger
from src.config.config import config

ALL_STRATEGIES = config.model_config.get("strategies", ["weighted", "mean_centering"])
OUTPUT_CSV = Path(
    config.evaluation_config.get("results_path", "reports/evaluation_results.csv")
)


def run_report(
    model_keys: list[str],
    split: str = "val",
    ks: list[int] = KS,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    output_csv: Path | None = OUTPUT_CSV,
    max_users: int | None = None,
    strategies: list[str] = ALL_STRATEGIES,
) -> pd.DataFrame:
    """
    Evaluate a list of CB models and return a summary comparison table.
    For each model the function:
        1. Loads the fitted artifact.
        2. Runs the evaluation loop once per strategy in `strategies`.
        3. Aggregates per-user metrics to means.
    Each (model, strategy) pair becomes one row in the output, keyed as
    "<model_key>_<strategy>" (e.g. "cb1_weighted", "cb1_mean_centering").
    When only a single strategy is requested the key stays plain ("cb1").
    Args:
        model_keys (list[str]): Short keys of models to evaluate, e.g. ['cb1', 'cb2'].
        split (str): 'val' or 'test'.
        ks (list[int]): Cut-off values, e.g. [5, 10, 20].
        relevance_threshold (float): Minimum rating to count as a positive item.
        output_csv (Path | None): If provided, save the results DataFrame to this CSV path.
        max_users (int | None): If set, cap the number of evaluated users per model.
        strategies (list[str]): Profile strategies to evaluate. Defaults to both
            'weighted' and 'mean_centering'. Pass a single-element list to skip
            the strategy suffix in the row key.
    Returns:
        pd.DataFrame: DataFrame indexed by "<model>_<strategy>" (or "<model>" when
            only one strategy is requested), one row per (model, strategy) pair.
    """
    rows = []
    use_suffix = len(strategies) > 1

    for key in model_keys:
        logger.info(f"Model: {key.upper()}")

        try:
            model = load_model(key)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            continue

        for strategy in strategies:
            logger.info(f"  Strategy: {strategy}")

            per_user_df = evaluate_model(
                model,
                split=split,
                ks=ks,
                relevance_threshold=relevance_threshold,
                max_users=max_users,
                profile_strategy=strategy,
            )

            if per_user_df.empty:
                logger.warning(f"{key} [{strategy}]: no eligible users — skipping")
                continue

            summary = aggregate_metrics(per_user_df, ks=ks)
            row_key = f"{key}_{strategy}" if use_suffix else key
            row = {"model": row_key, "n_users": len(per_user_df)}
            row.update(summary.to_dict())
            rows.append(row)

    if not rows:
        logger.warning("No results to report.")
        return pd.DataFrame()

    results_df = pd.DataFrame(rows).set_index("model")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv)
        logger.info(f"Saved evaluation results to {output_csv}")

    return results_df


# Markdown table formatter
def format_markdown_table(df: pd.DataFrame, ks: list[int] = KS) -> str:
    """
    Format a results DataFrame (output of run_report) as a Markdown table.
    Args:
        df (pd.DataFrame): DataFrame indexed by model key with metric columns.
        ks (list[int]): Cut-off values used during evaluation — controls column ordering.
    Returns:
        str: Multi-line string with a Markdown-formatted table.
    """
    if df.empty:
        return "No results to display"

    # Build ordered column list: n_users then metrics grouped by K
    metric_cols = [f"{m}@{k}" for k in ks for m in ("hr", "p", "r", "ndcg")]
    ordered_cols = ["n_users"] + [c for c in metric_cols if c in df.columns]
    display = df[ordered_cols].copy()

    # Round metric columns
    for col in ordered_cols[1:]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    display["n_users"] = display["n_users"].astype(int)

    display = display.reset_index()  # model becomes a regular column
    headers = list(display.columns)
    col_widths = [max(len(h), 8) for h in headers]

    def _sep() -> str:
        """
        Generate a separator row for the Markdown table.
        """
        return "| " + " | ".join("-" * w for w in col_widths) + " |"

    def _row(values: list) -> str:
        """
        Format a single row of values with padding for Markdown.
        """
        cells = [str(v).ljust(w) for v, w in zip(values, col_widths)]
        return "| " + " | ".join(cells) + " |"

    lines = [
        _row(headers),
        _sep(),
        *[_row(list(row)) for _, row in display.iterrows()],
    ]
    return "\n".join(lines)


# Command-line interface
def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation report.
    Returns:
        argparse.Namespace with attributes:
            models (list[str]): Models to evaluate.
            skip (list[str]): Models to skip.
            split (str): Evaluation split ('val' or 'test').
            ks (list[int]): Cut-off values for metrics.
            threshold (float): Relevance threshold for ground truth.
            output (Path | None): Optional path to save results as CSV.
            strategies (list[str]): Profile strategies to compare.
            max_users (int | None): Optional cap on number of evaluated users per model.
    """
    all_keys = list(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Evaluate content-based recommender models and print a comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=all_keys,
        choices=all_keys,
        metavar="MODEL",
        help="Models to evaluate (default: all). Choices: " + ", ".join(all_keys),
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=all_keys,
        metavar="MODEL",
        help="Models to skip (subtracted from --models).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Evaluation split to use.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=KS,
        metavar="K",
        help="Cut-off values for metric computation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help="Minimum rating to treat an item as relevant.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        metavar="PATH",
        help="Optional path to save results as CSV.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=ALL_STRATEGIES,
        choices=ALL_STRATEGIES,
        metavar="STRATEGY",
        help="Profile strategies to compare (default: both). "
        "Choices: weighted, mean_centering.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        metavar="N",
        help="Cap on number of evaluated users per model",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to run the evaluation report from the command line.
    Parses arguments, runs the report, and prints results in both plain text and Markdown formats.
    """
    args = _parse_args()
    keys = [k for k in args.models if k not in args.skip]

    if not keys:
        logger.warning("No models selected after applying --skip filter. Exiting.")
        return

    logger.info(
        f"Starting evaluation — models={keys}, split={args.split}, "
        f"strategies={args.strategies}, K={args.ks}, threshold={args.threshold}"
    )

    results = run_report(
        model_keys=keys,
        split=args.split,
        ks=args.ks,
        relevance_threshold=args.threshold,
        output_csv=args.output,
        max_users=args.max_users,
        strategies=args.strategies,
    )

    if results.empty:
        logger.warning("No evaluation results produced. Exiting.")
        return

    print(f"Evaluation Results  |  split={args.split}  |  K={args.ks}")
    print(results.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Markdown ---\n")
    print(format_markdown_table(results, ks=args.ks))


if __name__ == "__main__":
    main()
