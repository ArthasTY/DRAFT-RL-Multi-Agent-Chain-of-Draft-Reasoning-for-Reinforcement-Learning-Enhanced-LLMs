import torch
from draftrl.ppo import clipped_policy_loss, clipped_value_loss, terminal_gae


def test_terminal_return_with_zero_values() -> None:
    returns, advantages = terminal_gae(torch.zeros(3), reward=2.0, gamma=1.0, lam=1.0)
    assert torch.allclose(returns, torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(advantages, returns)


def test_identity_ratio_and_kl() -> None:
    logprobs = torch.tensor([-1.0, -2.0])
    loss, ratio, clip_fraction, kl = clipped_policy_loss(
        logprobs, logprobs, torch.tensor([1.0, -1.0]), epsilon=0.2
    )
    assert torch.allclose(ratio, torch.ones_like(ratio))
    assert float(clip_fraction) == 0.0
    assert abs(float(kl)) < 1e-8
    assert torch.isfinite(loss)


def test_clip_handles_positive_and_negative_advantages() -> None:
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.5, 0.5]))
    loss, ratio, clip_fraction, _ = clipped_policy_loss(
        new, old, torch.tensor([1.0, -1.0]), epsilon=0.2
    )
    assert torch.allclose(ratio, torch.tensor([1.5, 0.5]))
    assert float(clip_fraction) == 1.0
    assert torch.isfinite(loss)


def test_value_loss_is_finite() -> None:
    loss = clipped_value_loss(
        torch.tensor([0.2, 0.4]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, -1.0]),
        epsilon=0.2,
    )
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
