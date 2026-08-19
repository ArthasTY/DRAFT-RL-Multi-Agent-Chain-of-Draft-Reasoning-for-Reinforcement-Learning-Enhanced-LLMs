# Method mapping

The implementation follows the paper's executable method path. The included
profiles keep the data volume compact while retaining every learning stage.

| Paper component | Implementation |
| --- | --- |
| Agent policy $\pi_{\theta_i}$ | One shared Qwen3 policy with three independent LoRA adapters |
| Multi-draft generation | Two temperature-conditioned drafts per agent in the included profiles |
| Chain-of-Draft constraint | Parsed `STEP n:` lines with a five-word limit and a separate `FINAL:` answer |
| Peer evaluation | The two non-author agents return five scalar criteria and textual feedback |
| Reward model $R_\phi$ | Learned hashed-text sequence scorer with peer and CoD features |
| Global selection | Highest eligible RM score across all agents and drafts |
| Local selection | Highest eligible RM score within each agent's drafts |
| Task reward | Numeric GSM8K verifier applied only after generation and selection |
| Policy update | Token-level clipped PPO, value loss, KL diagnostic, and verified-target imitation |

## Selection decision

Equation 7 and Algorithm 1 admit two readings. This implementation uses both:

- the global highest-scoring eligible draft is the displayed answer;
- each agent's local highest-scoring eligible draft supplies that agent's PPO
  rollout and imitation target.

Gold answers never enter generation prompts, peer-review prompts, or RM ranking
features. They are consulted only by the verifier after the relevant candidates
and rankings are frozen.

## PPO objective

For response token $a_t$ generated from state $s_t$, the implementation stores
the rollout-policy log probability and old value. Terminal reward is propagated
with GAE. The actor uses the standard clipped ratio objective; the value head
uses a clipped value loss. A fixed verified target supplies the imitation NLL.

The optimization objective is:

$$
L = L_{PPO} + c_v L_V + \beta L_{KL} + \alpha L_{NLL}.
$$

The included profiles use $\epsilon=0.2$, $\gamma=1$, $\lambda=1$, and
$\alpha=0.5$. The base policy remains frozen; only the active LoRA and its value
head are optimized.
