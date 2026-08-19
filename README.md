# DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs

This repository contains the implementation of **DRAFT-RL**, a framework that
combines multi-draft Chain-of-Draft reasoning, peer-guided evaluation,
reward-aligned selection, and PPO with imitation learning.

The current release provides executable Qwen3-1.7B and Qwen3-8B profiles. Both
profiles complete the full DRAFT-RL path on one NVIDIA A40 GPU, including
multi-agent generation, peer evaluation, reward-model training, PPO with
imitation, and checkpoint reload.

Paper: [arXiv:2511.20468](https://arxiv.org/abs/2511.20468) ·
[AAAI-26](https://doi.org/10.1609/aaai.v40i35.40195)

## Method

![Overall architecture of the DRAFT-RL framework](assets/draft-rl-architecture.png)

For every query, each agent produces multiple concise drafts. Other agents
score each draft on coherence, step validity, relevance, completeness, and
answer correctness. A learned reward model ranks the candidates. The global
winner produces the displayed answer, while each agent's local winner supplies
its PPO and imitation target.

## Current status

| Profile | Shared policy | Agents | Drafts per agent | Learned RM | PPO + imitation | Save/reload |
| --- | --- | ---: | ---: | --- | --- | --- |
| `qwen3-1.7b` | Qwen3-1.7B | 3 | 2 | PASS | PASS | PASS |
| `qwen3-8b` | Qwen3-8B | 3 | 2 | PASS | PASS | PASS |

The validation gates verify:

- six evaluation drafts and twelve non-self peer reviews per query;
- structured five-criterion review output;
- learned RM optimization and score-preserving reload;
- non-zero clipped-PPO gradients for all three agents;
- isolated LoRA updates with a frozen shared base;
- adapter parameter and fixed-input logit equality after reload;
- finite losses, ratios, rewards, and gradients.

See [Validation results](docs/VALIDATION.md) for execution evidence and observed
capability boundaries.

## Installation

The validated environment used Python 3.12, PyTorch 2.8.0 + CUDA 12.8,
Transformers 4.57.6, and PEFT 0.17.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Set the Hugging Face cache outside the repository if desired:

```bash
export HF_HOME=/path/to/hf-cache
```

## Run

Qwen3-1.7B:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_draftrl.py \
  --config configs/qwen3_1.7b.json
```

Qwen3-8B:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_draftrl.py \
  --config configs/qwen3_8b.json
```

Each run writes an auditable bundle to `artifacts/<run_id>/` containing:

```text
config.json
data_freeze.json
trajectories.jsonl
train_metrics.jsonl
summary.json
demo.md
checkpoints/
```

Run deterministic unit tests with:

```bash
python -m pytest -q
```

## Repository layout

```text
assets/               Original paper architecture figure
configs/              Qwen3 execution profiles
docs/                 Method and validation notes
examples/             Compact successful-run evidence
scripts/              Entry point
src/draftrl/           CoD, peer review, RM, PPO, and orchestration
tests/                 Deterministic unit tests
```

## Scope

The included profiles provide a compact execution of the complete method using
a fixed GSM8K subset, `K=2`, a learned hashed-text reward scorer, and one PPO
iteration. Full paper-scale datasets, multi-seed experiments, and benchmark
tables are outside the current release.

## Citation

```bibtex
@article{li2026draftrl,
  title={DRAFT-RL: Multi-Agent Chain-of-Draft Reasoning for Reinforcement Learning-Enhanced LLMs},
  author={Li, Yuanhao and Liu, Mingshan and Wang, Hongbo and Zhang, Yiding and Ma, Yifei and Tan, Wei},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={35},
  pages={29530--29537},
  year={2026},
  doi={10.1609/aaai.v40i35.40195}
}
```
