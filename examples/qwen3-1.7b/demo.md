# DRAFT-RL 1.7B Method Smoke

> This is a small real method demonstration, not a performance claim.

## Query

Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

## Six candidate drafts and RM ranking

### Rank 1 — agent_a / draft 1

- Temperature: 0.2
- CoD valid: True
- RM score: 0.9995
- Selected globally: True
- Post-selection verifier reward: 0.0

```text
STEP 1: 16 eggs per day
STEP 2: 16 - 3 = 13 eggs sold
STEP 3: 13 * 2 = 26 dollars
FINAL: 26
```

Peer reviews:

- agent_b: overall=1.00; The solution is correct and follows logically.
- agent_c: overall=1.00; The solution is correct and follows logically.

### Rank 2 — agent_a / draft 2

- Temperature: 0.8
- CoD valid: True
- RM score: 0.9693
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: 16 eggs per day
STEP 2: 16 - 3 = 13 eggs sold
STEP 3: 13 * 2 = 26 dollars
FINAL: 26
```

Peer reviews:

- agent_b: overall=1.00; The solution is correct and well-structured.
- agent_c: overall=0.00; The calculation is correct, but the steps are not fully explained.

### Rank 3 — agent_b / draft 1

- Temperature: 0.2
- CoD valid: True
- RM score: 0.1207
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Define variables
STEP 2: Calculate eggs laid
STEP 3: Calculate eggs sold
STEP 4: Calculate revenue
FINAL: 64
```

Peer reviews:

- agent_a: overall=0.00; The calculation is incorrect. Janet lays 16 eggs per day, eats 3 for breakfast, and bakes 4 muffins. The remaining eggs sold at the market are 16 - 3 - 4 = 9. At $2 per egg, revenue is 9 * 2 = $18.
- agent_c: overall=0.00; The calculation is incorrect. Janet lays 16 eggs per day, eats 3 for breakfast, and bakes 4 muffins. The remaining eggs sold at the market are 16 - 3 - 4 = 9. At $2 per egg, the revenue is 9 * 2 = $18.

### Rank 4 — agent_c / draft 2

- Temperature: 0.8
- CoD valid: True
- RM score: 0.0079
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: 16 eggs per day
STEP 2: 16 - 3 - 4 = 9
FINAL: 9
```

Peer reviews:

- agent_a: overall=0.00; The calculation is incorrect. Janet sells 9 eggs per day, but the correct amount is 16 - 3 - 4 = 9, which is correct. The answer is 9 dollars.
- agent_b: overall=0.00; The calculation is correct, but the steps are not fully explained. The answer is 9 dollars.

### Rank 5 — agent_c / draft 1

- Temperature: 0.2
- CoD valid: True
- RM score: 0.0002
- Selected globally: False
- Post-selection verifier reward: 0.0

```text
STEP 1: Total eggs laid
STEP 2: Subtract eaten and used eggs
FINAL: 9
```

Peer reviews:

- agent_a: overall=0.00; The calculation is incorrect. The correct answer is $12.
- agent_b: overall=0.00; The calculation is incorrect. Janet lays 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, totaling 7 eggs. The remaining eggs sold at the market are 16 - 7 = 9. The correct daily revenue is 9 * $2 = $18.

### Rank 6 — agent_b / draft 2

- Temperature: 0.8
- CoD valid: False
- RM score: 0.8909
- Selected globally: False
- Post-selection verifier reward: 1.0

```text
STEP 1: Define variables
STEP 2: Eggs laid = 16
STEP 3: Eggs sold = 16 - 3 - 4 = 9
STEP 4: Revenue = 9 * 2 = 18
FINAL: 18
```

Peer reviews:

- agent_a: overall=1.00; The solution is correct and follows logically. All steps are valid and relevant.
- agent_c: overall=1.00; The solution is correct and follows logically. All steps are valid and relevant.

## Reward-model smoke

- Loss: 0.6933 → 0.0034
- Positive mean score: 0.9966
- Negative mean score: 0.0034
- Calibration pair passed: True

## PPO + imitation updates

- agent_a: PPO loss=-0.1685, PPO grad=0.030269, adapter Δ=0.426392, inactive max Δ=0.0
- agent_b: PPO loss=-0.1454, PPO grad=2.721594, adapter Δ=0.371816, inactive max Δ=0.0
- agent_c: PPO loss=-0.2583, PPO grad=0.041363, adapter Δ=0.412799, inactive max Δ=0.0

## One post-update draft per agent

### agent_a

```text
STEP 1: 16 eggs/day
STEP 2: 13 eggs/day (16 - 3)
STEP 3: 26 dollars/day (13 * 2)
STEP 4: 26 dollars/day
STEP 5: 26
FINAL: 26
```

### agent_b

```text
STEP 1: Define variables
STEP 2: Calculate eggs laid
STEP 3: Calculate eggs sold
FINAL: 48
```

### agent_c

```text
STEP 1: Calculate total eggs laid per day
STEP 2: Subtract eggs eaten and used for muffins
FINAL: 16 - 3 - 4 = 9
```
