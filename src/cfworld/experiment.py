"""Generic vs structured prompting: K runs per (rule, method), then score."""
import json
from pathlib import Path

import pandas as pd

from . import metrics
from .generate import GenFn, World, generic_world, structured_world
from .prompts import RULES

METHODS = {"generic": generic_world, "structured": structured_world}


def run_experiment(
    gen: GenFn,
    rules: list[str] = RULES,
    k: int = 5,
    embed: metrics.Embed = metrics.default_embed,
    out_dir: str | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[World]]]:
    """Returns (summary table, raw worlds keyed by (rule, method))."""
    worlds: dict[tuple[str, str], list[World]] = {}
    rows = []
    for rule in rules:
        for method, build in METHODS.items():
            runs = [build(gen, rule, seed=i) for i in range(k)]
            worlds[(rule, method)] = runs

            texts_per_run = [w.texts for w in runs]
            cov = [metrics.domain_coverage(t, embed) for t in texts_per_run]
            rows.append({
                "rule": rule,
                "method": method,
                "consistency": metrics.consistency(runs, embed),
                "semantic_diversity": _mean(metrics.semantic_diversity(t, embed) for t in texts_per_run),
                "distinct_2": _mean(metrics.distinct_n(t, 2) for t in texts_per_run),
                "domain_coverage": _mean(c[0] for c in cov),
                "domain_entropy": _mean(c[1] for c in cov),
            })

    df = pd.DataFrame(rows)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "summary.csv", index=False)
        raw = {f"{r} | {m}": [[c.__dict__ for c in w.consequences] for w in ws]
               for (r, m), ws in worlds.items()}
        (out / "worlds.json").write_text(json.dumps(raw, indent=2))
    return df, worlds


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
