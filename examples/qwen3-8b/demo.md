# DRAFT-RL Method Smoke — Qwen/Qwen3-8B

> This is a small real method demonstration, not a performance claim.

## Query

Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

## Six candidate drafts and RM ranking

### Rank 1 — agent_b / draft 1

- Temperature: 0.2
- CoD valid: True
- RM score: 0.0929
- Selected globally: True
- Post-selection verifier reward: 0.0

```text
STEP 1: Total eggs laid daily
STEP 2: Subtract eggs used
STEP 3: Multiply remaining by price
FINAL: 22
```

Peer reviews:

- agent_a: overall=0.00; No explanation, no calculations, no reasoning provided.
- agent_c: overall=0.00; No explanation, no calculations, no logic.

### Rank 2 — agent_b / draft 2

- Temperature: 0.8
- CoD valid: True
- RM score: 0.0333
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Total eggs laid daily
STEP 2: Subtract breakfast eggs
STEP 3: Subtract muffin eggs
STEP 4: Multiply by price per egg
FINAL: 22
```

Peer reviews:

- agent_a: overall=0.00; No steps or explanation provided.
- agent_c: overall=0.00; No steps or explanation provided.

### Rank 3 — agent_a / draft 2

- Temperature: 0.8
- CoD valid: True
- RM score: 0.0261
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Ducks lay 16 eggs
STEP 2: Eat 3 eggs breakfast
STEP 3: Use 4 eggs muffins
STEP 4: Remaining eggs sold
FINAL: 22
```

Peer reviews:

- agent_b: overall=0.00; No steps, no calculation, no explanation.
- agent_c: overall=0.00; No steps, no calculation, no explanation.

### Rank 4 — agent_a / draft 1

- Temperature: 0.2
- CoD valid: True
- RM score: 0.0128
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Ducks lay 16 eggs daily
STEP 2: Subtract eggs eaten
STEP 3: Subtract eggs used for muffins
STEP 4: Multiply remaining eggs by price
FINAL: 22
```

Peer reviews:

- agent_b: overall=0.00; No steps or explanation provided. Answer is incorrect.
- agent_c: overall=0.00; No steps or explanation provided. Answer is incorrect.

### Rank 5 — agent_c / draft 1

- Temperature: 0.2
- CoD valid: True
- RM score: 0.0091
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Ducks lay 16 eggs daily
STEP 2: Subtract eggs eaten and used
STEP 3: Calculate eggs sold
STEP 4: Multiply eggs by price
FINAL: 22
```

Peer reviews:

- agent_a: overall=0.00; No steps or explanation provided.
- agent_b: overall=0.00; No steps or explanation provided.

### Rank 6 — agent_c / draft 2

- Temperature: 0.8
- CoD valid: True
- RM score: 0.0068
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Ducks lay 16 eggs
STEP 2: Eggs used daily
STEP 3: Remaining eggs sold
STEP 4: Multiply by price per egg
FINAL: 22
```

Peer reviews:

- agent_a: overall=0.00; No steps or explanation provided.
- agent_b: overall=0.00; No steps or explanation provided.

## Reward-model smoke

- Loss: 0.6933 → 0.0027
- Positive mean score: 0.9973
- Negative mean score: 0.0027
- Calibration pair passed: True

## PPO + imitation updates

- agent_a: PPO loss=-1.0978, PPO grad=0.364934, adapter Δ=0.349380, inactive max Δ=0.0
- agent_b: PPO loss=-1.0997, PPO grad=0.045609, adapter Δ=0.347949, inactive max Δ=0.0
- agent_c: PPO loss=-0.9003, PPO grad=0.001421, adapter Δ=0.327579, inactive max Δ=0.0

## One post-update draft per agent

### agent_a

```text
STEP 1: Ducks lay 16 eggs daily
STEP 2: Subtract eggs eaten
STEP 3: Subtract eggs used for muffins
STEP 4: Multiply remaining eggs by price
FINAL: 24
```

### agent_b

```text
STEP 1: Total eggs laid daily
STEP 2: Subtract eggs used
STEP 3: Multiply remaining eggs by price
FINAL: 22
```

### agent_c

```text
STEP 1: Total eggs laid daily
STEP 2: Subtract eggs used
STEP 3: Calculate eggs sold
STEP 4: Multiply by price per egg
FINAL: 22
```
