# Execution validation

Two DRAFT-RL profiles completed on one NVIDIA A40 GPU with PyTorch 2.8.0 and
CUDA 12.8.

| Item | Qwen3-1.7B | Qwen3-8B |
| --- | ---: | ---: |
| Status | PASS | PASS |
| Trainable parameters per LoRA | 8,716,288 | 21,823,488 |
| Training candidates | 18 | 6 |
| Evaluation candidates | 6 | 6 |
| Peer reviews | 48 | 24 |
| RM initial loss | 0.6933 | 0.6933 |
| RM final loss | 0.0034 | 0.0027 |
| Adapter state reload max difference | 0 | 0 |
| Adapter logit reload max difference | 0 | 0 |
| Passed gates | 16/16 | 16/16 |

For every agent in both profiles:

- clipped-PPO gradients were non-zero;
- the active adapter changed;
- inactive adapters remained bitwise unchanged;
- base-model trainable parameter count was zero;
- adapter and value-head checkpoints were written;
- all recorded numeric metrics were finite.

## Capability boundary

These runs validate the mechanism, not accuracy improvement.

- Qwen3-1.7B produced one correct candidate among six evaluation drafts, but
  that candidate violated the configured CoD constraint. The selected answer
  was incorrect.
- Qwen3-8B produced no correct candidate among six evaluation drafts in its
  single held-out query. The selected answer was incorrect.
- One PPO update did not establish a capability improvement for either profile.
- The low RM training loss reflects calibration on a tiny set and must not
  be interpreted as held-out generalization.

Compact evidence for both runs is stored under `examples/`. Large checkpoints,
optimizer states, model caches, and failed intermediate runs are intentionally
excluded from Git.
