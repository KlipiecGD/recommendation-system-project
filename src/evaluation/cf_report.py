import argparse
from pathlib import Path

import pandas as pd

from src.models.collaborative_filtering.cf_model import CFModel, ALGORITHM_REGISTRY
from src.evaluation.cf_evaluator import evaluate_model, KS, RELEVANCE_THRESHOLD
from src.logging_utils.logger import logger
from src.config.config import config

CF_ARTIFACTS_DIR = Path(config.data_config.get("cf_artifacts_dir", "artifacts/cf"))
CF_REPORTS_PATH = Path(
    config.data_config.get("cf_reports_path", "reports/cf_evaluation_results.csv")
)


def format_comparison_table(results: list[dict], ks: list[int] = KS) -> str:
    """
    Format evaluation results as a readable markdown table.
    Columns are ordered: model | n_users | rmse | hr@K | p@K | r@K | ndcg@K
    for each K, grouped by K for easy scanning.
    Args:
        results (list[dict]): List of result dicts from evaluate_model().
        ks (list[int]): Cut-off values used during evaluation.
    Returns:
        str: Multi-line markdown table string.
    """
    if not results:
        return "No results to display."

    df = pd.DataFrame(results).set_index("model")

    # Build ordered column list
    metric_cols = [f"{m}@{k}" for k in ks for m in ("hr", "p", "r", "ndcg")]
    ordered_cols = ["n_users", "rmse"] + [c for c in metric_cols if c in df.columns]
    df = df[ordered_cols]

    # Format values
    display = df.copy().reset_index()
    for col in ordered_cols:
        if col == "n_users":
            display[col] = display[col].astype(int)
        elif col == "rmse":
            display[col] = display[col].map(lambda x: f"{x:.4f}")
        else:
            display[col] = display[col].map(lambda x: f"{x:.4f}")

    headers = list(display.columns)
    col_widths = [max(len(h), 10) for h in headers]

    def _sep() -> str:
        """Generate markdown separator row based on column widths."""
        return "| " + " | ".join("-" * w for w in col_widths) + " |"

    def _row(values: list) -> str:
        """Format a single row of values as a markdown table row."""
        cells = [str(v).ljust(w) for v, w in zip(values, col_widths)]
        return "| " + " | ".join(cells) + " |"

    lines = [
        _row(headers),
        _sep(),
        *[_row(list(row)) for _, row in display.iterrows()],
    ]
    return "\n".join(lines)


def print_summary(results: list[dict], ks: list[int] = KS) -> None:
    """
    Print a human-readable summary highlighting best model per metric.
    Args:
        results (list[dict]): List of result dicts from evaluate_model().
        ks (list[int]): Cut-off values used during evaluation.
    """
    if not results:
        return

    df = pd.DataFrame(results).set_index("model")

    print("BEST MODEL PER METRIC")

    # RMSE — lower is better
    best_rmse_model = df["rmse"].idxmin()
    print(f"RMSE:  {best_rmse_model} ({df.loc[best_rmse_model, 'rmse']:.4f})")

    # Ranking metrics — higher is better
    ranking_metrics = [f"{m}@{k}" for k in ks for m in ("hr", "ndcg")]
    for metric in ranking_metrics:
        if metric in df.columns:
            best_model = df[metric].idxmax()
            print(
                f"{metric.upper():12s}:  {best_model} ({df.loc[best_model, metric]:.4f})"
            )


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluating CF models.
    Returns:
        argparse.Namespace: Parsed arguments with 'models', 'skip', 'ks', 'threshold', 'max_users', and 'output' attributes.
    """
    all_keys = list(ALGORITHM_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Evaluate collaborative_filtering filtering models and print comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=all_keys,
        choices=all_keys,
        metavar="MODEL",
        help=f"Models to evaluate. Choices: {all_keys}",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=all_keys,
        metavar="MODEL",
        help="Models to skip.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=KS,
        metavar="K",
        help="Cut-off values for ranking metrics.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RELEVANCE_THRESHOLD,
        help="Minimum rating to treat an item as relevant.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        metavar="N",
        help="Cap on evaluated users per model. Useful for quick dev checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CF_REPORTS_PATH,
        metavar="PATH",
        help="Optional path to save results as CSV (e.g. reports/cf_results.csv).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for evaluating CF models.
    Parses arguments, loads fitted models, runs evaluation, and prints comparison table and summary.
    """
    args = _parse_args()
    keys_to_eval = [k for k in args.models if k not in args.skip]

    if not keys_to_eval:
        logger.warning("No models selected after applying --skip. Exiting.")
        return

    logger.info(f"Models to evaluate: {keys_to_eval}")
    logger.info(
        f"K={args.ks} | threshold={args.threshold} | max_users={args.max_users}"
    )

    results = []

    for key in keys_to_eval:
        logger.info(f"\nEvaluating {key.upper()} ...")

        # Load fitted model
        try:
            model = CFModel.load_cf(CF_ARTIFACTS_DIR, key)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            continue

        # Run full evaluation
        try:
            result = evaluate_model(
                model=model,
                ks=args.ks,
                relevance_threshold=args.threshold,
                max_users=args.max_users,
            )
            results.append(result)
            logger.info(
                f"{key.upper()} done — "
                f"RMSE: {result['rmse']:.4f} | "
                f"HR@10: {result.get('hr@10', 0):.4f} | "
                f"NDCG@10: {result.get('ndcg@10', 0):.4f}"
            )
        except Exception as exc:
            logger.error(f"{key.upper()} evaluation FAILED: {exc}", exc_info=True)
            continue

    if not results:
        logger.warning("No evaluation results produced.")
        return

    # Print full comparison table
    print(f"CF MODEL COMPARISON  |  K={args.ks}  |  threshold={args.threshold}")
    print(format_comparison_table(results, ks=args.ks))

    # Print best model summary
    print_summary(results, ks=args.ks)

    # Optionally save to CSV
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).set_index("model").to_csv(args.output)
        logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
