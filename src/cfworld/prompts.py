from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    name: str
    hint: str  # also used to embed the domain for coverage metrics


DOMAINS = [
    Domain("education", "schools, learning, teachers and students"),
    Domain("economics", "jobs, markets, money and industries"),
    Domain("relationships", "family, friendship, love and community"),
    Domain("technology", "inventions, devices and infrastructure"),
    Domain("daily life", "routines, habits, food, housing and leisure"),
]

RULES = [
    "humans never needed sleep",
    "money did not exist",
    "nobody could lie",
    "the internet was never invented",
    "people stopped aging at 30",
]

# --- Baseline: one generic prompt, sampled repeatedly -----------------------
GENERIC = (
    "Imagine a world where {rule}. "
    "Describe one consequence of this world."
)

# --- Structured: multi-step ------------------------------------------------
# Step 1: pin down the premise so later steps stay consistent with it.
PREMISE = (
    "Imagine a world where {rule}. "
    "In one sentence, describe the single biggest change in this world."
)

# Step 2: one focused prompt per domain, conditioned on the premise.
DOMAIN = (
    "World: {rule}. Key change: {premise}\n"
    "How does this change {domain} ({hint})? "
    "Describe one specific consequence in one sentence."
)

# Step 3 (optional): second-order ripple effect.
RIPPLE = (
    "World: {rule}.\n"
    "Consequence: {consequence}\n"
    "What further consequence follows from this? Answer in one sentence."
)
