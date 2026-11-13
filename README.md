# What If? — Counterfactual World Generator

Give a small instruction-tuned model (FLAN-T5) a real-world rule — "humans never needed sleep" — and have it imagine the consequences across education, economics, relationships, technology and daily life.

**Experiment:** generic prompting vs structured multi-step prompting (premise → per-domain consequences → optional ripple effects), scored on consistency and diversity.

## Layout
- `src/cfworld/` — prompts, model wrapper, world generation, metrics, experiment runner
- `notebooks/counterfactual_world.ipynb` — the Colab entry point
- `tests/` — metric/parsing tests using a fake model (no downloads)

## Run on Google Colab
1. Zip this folder as `project.zip` (or push to GitHub and set `REPO_URL` in the notebook).
2. Open `notebooks/counterfactual_world.ipynb` in Colab (File → Upload notebook).
3. Runtime → Change runtime type → T4 GPU, then Run all.

## Run tests locally
```bash
pip install numpy pandas pytest
pytest
```
