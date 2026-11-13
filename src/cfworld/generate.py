"""Two ways to build a counterfactual world from one rule.

Both produce the same number of consequences by default (len(DOMAINS) *
per_domain) so the comparison is about prompting strategy, not output size.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

from .prompts import DOMAIN, DOMAINS, GENERIC, PREMISE, RIPPLE

# gen(prompt, n=..., seed=...) -> list[str]
GenFn = Callable[..., list[str]]


@dataclass
class Consequence:
    text: str
    domain: str | None = None  # None for generic prompting (untagged)
    order: int = 1  # 1 = direct, 2 = ripple


@dataclass
class World:
    rule: str
    method: str
    consequences: list[Consequence] = field(default_factory=list)
    premise: str | None = None

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.consequences]


def clean(text: str) -> str:
    """Strip bullets/numbering and keep the first sentence."""
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text.strip())
    first = re.split(r"(?<=[.!?])\s+", text)[0]
    return first.strip()


def generic_world(gen: GenFn, rule: str, n_items: int | None = None, seed: int | None = None) -> World:
    n_items = n_items or len(DOMAINS)
    outs = gen(GENERIC.format(rule=rule), n=n_items, seed=seed)
    world = World(rule=rule, method="generic")
    world.consequences = [Consequence(clean(o)) for o in outs if clean(o)]
    return world


def structured_world(
    gen: GenFn,
    rule: str,
    per_domain: int = 1,
    ripple: bool = False,
    seed: int | None = None,
) -> World:
    premise = clean(gen(PREMISE.format(rule=rule), n=1, temperature=0.7, seed=seed)[0])
    world = World(rule=rule, method="structured", premise=premise)
    for d in DOMAINS:
        prompt = DOMAIN.format(rule=rule, premise=premise, domain=d.name, hint=d.hint)
        for out in gen(prompt, n=per_domain):
            text = clean(out)
            if not text:
                continue
            world.consequences.append(Consequence(text, d.name, 1))
            if ripple:
                nxt = clean(gen(RIPPLE.format(rule=rule, consequence=text), n=1)[0])
                if nxt:
                    world.consequences.append(Consequence(nxt, d.name, 2))
    return world
