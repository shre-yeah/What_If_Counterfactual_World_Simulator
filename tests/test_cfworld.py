import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfworld import DOMAINS, generic_world, run_experiment, structured_world  # noqa: E402
from cfworld.generate import clean  # noqa: E402
from cfworld.metrics import distinct_n, domain_coverage, semantic_diversity  # noqa: E402


def fake_gen(prompt, n=1, seed=None, **kw):
    return [f"1. Thing {hashlib.md5(f'{prompt}{seed}{i}'.encode()).hexdigest()[:4]} happens. Extra." for i in range(n)]


def fake_embed(texts):
    out = []
    for t in texts:
        h = hashlib.md5(t.encode()).digest()
        out.append(np.frombuffer(h, dtype=np.uint8).astype(float) - 127)
    return np.array(out)


def test_clean():
    assert clean("1. Schools close. They reopen.") == "Schools close."
    assert clean("- Jobs vanish") == "Jobs vanish"


def test_world_sizes_match():
    g = generic_world(fake_gen, "x", seed=0)
    s = structured_world(fake_gen, "x", seed=0)
    assert len(g.consequences) == len(s.consequences) == len(DOMAINS)
    assert {c.domain for c in s.consequences} == {d.name for d in DOMAINS}


def test_ripple_doubles():
    s = structured_world(fake_gen, "x", ripple=True)
    assert len(s.consequences) == 2 * len(DOMAINS)


def test_metrics_bounds():
    assert distinct_n(["a b c", "a b c"], 2) == 0.5
    same = np.ones((3, 4))
    assert abs(semantic_diversity(["a", "b", "c"], lambda t: same)) < 1e-9
    cov, ent = domain_coverage(["a"] * 5, fake_embed)
    assert 0 < cov <= 1 and 0 <= ent <= 1


def test_experiment_runs():
    df, worlds = run_experiment(fake_gen, rules=["r1"], k=3, embed=fake_embed)
    assert set(df.method) == {"generic", "structured"}
    assert len(worlds[("r1", "generic")]) == 3
