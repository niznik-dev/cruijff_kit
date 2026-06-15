#!/usr/bin/env python3
"""
Quick summary of continuous/regression results from inspect-ai .eval files.

Like summary() in R — a quick glance, not a robust analysis tool. The companion
to summary_binary.py: where that builds a 0/1 confusion matrix, this reports the
regression metrics continuous_scorer produces (mae, rmse, r_squared, parse_rate)
plus a distributional glance the aggregate metrics hide. A model that emits one
constant scores R²≈0 just like a noisy model does — but here it shows up as zero
prediction variance, and a low parse_rate shows up as a parse-failure count.

Reads the scorer's own per-sample outputs (prediction / target_value / error
from Score.metadata) via read_eval_log, so the numbers match the scorer exactly
rather than re-parsing the model text. Aggregate metrics are pulled straight
from results.scores (authoritative); the per-sample distribution is derived from
the same metadata the scorer stored.

Usage:
    python -m cruijff_kit.tools.inspect.summary_continuous /path/to/log.eval
    python -m cruijff_kit.tools.inspect.summary_continuous /path/to/logs/       # all .eval in dir
    python -m cruijff_kit.tools.inspect.summary_continuous /path/to/log.eval --json
"""

import argparse
import json
import math
import statistics
from pathlib import Path

# The scorer key under which continuous predictions live, in both the per-sample
# Score.metadata and the aggregate results.scores entry.
SCORER_NAME = "continuous_scorer"


def _round(x, ndigits: int = 4):
    """Round a metric for display; map None and NaN to None so they read as n/a."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return round(x, ndigits)
    except (TypeError, ValueError):
        return x


def _distribution(values: list) -> dict | None:
    """min/max/mean/std for a list of floats. Returns None when empty.

    std is population stdev (pstdev), 0.0 for a single value — a flat
    distribution (std=0) is the tell for a model emitting one constant.
    """
    if not values:
        return None
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def compute_metrics(path: Path) -> dict:
    """Compute a regression summary from a continuous_scorer eval file.

    Returns dict with metrics + distributions or an error status.
    """
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return {
            "status": "error",
            "message": "inspect_ai not installed. Run: pip install inspect-ai",
            "path": str(path),
        }

    try:
        log = read_eval_log(str(path))
    except Exception as e:
        return {"status": "error", "message": str(e), "path": str(path)}

    samples = log.samples or []
    n = len(samples)
    if n == 0:
        return {
            "status": "error",
            "message": f"No samples in {path}",
            "path": str(path),
        }

    # Authoritative aggregate metrics, straight from the scorer's results.
    agg: dict = {}
    scorer_name = SCORER_NAME
    if log.results and log.results.scores:
        score = next(
            (s for s in log.results.scores if s.name == SCORER_NAME),
            log.results.scores[0],
        )
        scorer_name = score.name
        for mname, mval in (score.metrics or {}).items():
            agg[mname] = getattr(mval, "value", mval)

    # Per-sample glance from the scorer's stored metadata.
    preds: list = []
    targets: list = []
    errors: list = []
    n_parsed = 0
    for s in samples:
        scores = getattr(s, "scores", None) or {}
        sc = scores.get(SCORER_NAME)
        meta = (getattr(sc, "metadata", None) or {}) if sc else {}
        pred = meta.get("prediction")
        tgt = meta.get("target_value")
        err = meta.get("error")
        if pred is not None:
            n_parsed += 1
            preds.append(pred)
        if tgt is not None:
            targets.append(tgt)
        if err is not None:
            errors.append(err)

    fallback_parse_rate = n_parsed / n if n else 0.0

    return {
        "status": "success",
        "path": str(path),
        # Output key: the name of the scorer that ran, distinct from the input
        # `scorers` list in eval.yaml (see parse_eval_log.py).
        "scorer": scorer_name,
        "samples": n,
        "parsed": n_parsed,
        "parse_failures": n - n_parsed,
        "metrics": {
            "mae": _round(agg.get("mae")),
            "rmse": _round(agg.get("rmse")),
            "r_squared": _round(agg.get("r_squared")),
            "parse_rate": _round(agg.get("parse_rate", fallback_parse_rate)),
        },
        "prediction_distribution": _distribution(preds),
        "target_distribution": _distribution(targets),
        "residual_distribution": _distribution(errors),
    }


def _fmt(x) -> str:
    """Format a metric value for the human-readable table (n/a for None)."""
    return "n/a" if x is None else f"{x:.4f}"


def _dist_row(label: str, dist: dict | None) -> str:
    if dist is None:
        return f"{label:<12} {'n/a':>9}"
    return (
        f"{label:<12} {dist['min']:>9.3f} {dist['max']:>9.3f} "
        f"{dist['mean']:>9.3f} {dist['std']:>9.3f}"
    )


def print_summary(path: Path) -> dict:
    """Print human-readable summary and return the metrics dict."""
    result = compute_metrics(path)

    if result["status"] == "error":
        print(result["message"])
        return result

    m = result["metrics"]
    print(f"\n{'=' * 60}")
    print(f"{path.name}")
    print(f"{'=' * 60}")
    print(
        f"n = {result['samples']}  (parsed {result['parsed']}, {result['parse_failures']} failed)"
    )

    print(f"\nMAE:        {_fmt(m['mae'])}")
    print(f"RMSE:       {_fmt(m['rmse'])}")
    print(f"R²:         {_fmt(m['r_squared'])}")
    pr = m["parse_rate"]
    print(f"Parse rate: {'n/a' if pr is None else f'{pr:.1%}'}")

    print(f"\n{'':<12} {'min':>9} {'max':>9} {'mean':>9} {'std':>9}")
    print(_dist_row("Prediction", result["prediction_distribution"]))
    print(_dist_row("Target", result["target_distribution"]))
    print(_dist_row("Residual", result["residual_distribution"]))

    pred = result["prediction_distribution"]
    if pred is not None and pred["std"] == 0.0 and result["parsed"] > 1:
        print(
            "\nNote: prediction std = 0 — the model emitted a single constant; "
            "a low R² here reflects that, not noise."
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Quick summary of continuous/regression eval results"
    )
    parser.add_argument("path", type=Path, help=".eval file or directory")
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output JSON instead of human-readable format",
    )
    args = parser.parse_args()

    if args.json:
        if args.path.is_dir():
            results = [compute_metrics(f) for f in sorted(args.path.glob("**/*.eval"))]
            print(json.dumps(results, indent=2))
        else:
            print(json.dumps(compute_metrics(args.path), indent=2))
    else:
        if args.path.is_dir():
            for f in sorted(args.path.glob("**/*.eval")):
                print_summary(f)
        else:
            print_summary(args.path)


if __name__ == "__main__":
    main()
