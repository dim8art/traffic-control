"""
Градиент log π(a|obs) по вектору наблюдений — локальная «важность» компонент (sensitivity).

Подходит только для Torch-политик RLlib; даёт вклад именно для выбранного действия ``taken_action``,
а не истинную причинность симулятора.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Порядок как в MultiAgentTrafficEnv._get_local_obs — для чужих размерностей будут суффиксы f{n}.
OBS_LABELS_6 = (
    "local_waiting",
    "local_speed",
    "local_upstream_occ",
    "neighbor_waiting_w",
    "neighbor_speed_w",
    "neighbor_upstream_w",
)

# Суффиксы в том же порядке, что и в MultiAgentTrafficEnv._get_local_obs (после базовых 6).
OBS_LABELS_PED = ("local_ped_count", "neighbor_ped_count_w")
OBS_LABELS_PT = ("local_halting_buses", "neighbor_halting_buses_w")


def _extended_obs_labels(
    *,
    enable_pedestrians: bool,
    enable_public_transport: bool,
) -> tuple[str, ...]:
    names = list(OBS_LABELS_6)
    if enable_pedestrians:
        names.extend(OBS_LABELS_PED)
    if enable_public_transport:
        names.extend(OBS_LABELS_PT)
    return tuple(names)


def observation_feature_labels(
    obs_dim: int,
    *,
    enable_pedestrians: bool | None = None,
    enable_public_transport: bool | None = None,
) -> tuple[str, ...]:
    """
    Имена компонентов наблюдения. Без флагов поведение как раньше: после 6 базовых — f6, f7, …
    С флагами — осмысленные имена для блоков пешеходов и ОТ (размер должен совпадать с env).
    """
    if enable_pedestrians is not None or enable_public_transport is not None:
        ped = bool(enable_pedestrians)
        pt = bool(enable_public_transport)
        base = list(_extended_obs_labels(enable_pedestrians=ped, enable_public_transport=pt))
        while len(base) < obs_dim:
            base.append(f"f{len(base)}")
        return tuple(base[:obs_dim])

    labels = []
    for i in range(obs_dim):
        if i < len(OBS_LABELS_6):
            labels.append(OBS_LABELS_6[i])
        else:
            labels.append(f"f{i}")
    return tuple(labels)


def observation_saliency_logp_gradient(
    policy: Any,
    observation: Sequence[float] | np.ndarray,
    *,
    taken_action: int,
) -> np.ndarray:
    """
    Δ log π(action|obs) / Δ obs — вектор той же длины, что ``observation``.

    ``taken_action`` должен совпадать с тем, который реально применён (например ``explore=False``).
    """
    cfg = policy.config
    framework = (
        cfg.get("framework")
        if isinstance(cfg, dict)
        else getattr(cfg, "framework", None)
    )
    if framework != "torch":
        raise TypeError(
            "saliency поддерживается только для framework=torch, "
            f"сейчас: {framework!r}",
        )

    import torch

    from ray.rllib.policy.sample_batch import SampleBatch

    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    batch_np = SampleBatch(
        {
            SampleBatch.CUR_OBS: obs[np.newaxis, ...],
            SampleBatch.ACTIONS: np.asarray([taken_action], dtype=np.int64),
        },
    )

    laz = policy._lazy_tensor_dict(batch_np)
    obs_t = laz[SampleBatch.CUR_OBS]

    obs_t.requires_grad_(True)

    device = policy.device
    seq_lens = torch.ones(1, dtype=torch.int32, device=device)
    state_batches: list[Any] = []

    was_training = policy.model.training
    try:
        policy.model.train(False)

        dist_inputs: Any
        with torch.enable_grad():
            dist_inputs, _ = policy.model(laz, state_batches, seq_lens)

            policy_dist = policy.dist_class(dist_inputs, policy.model)
            acts = laz[SampleBatch.ACTIONS].view(-1)
            scalar = policy_dist.logp(acts).sum()

        grads = torch.autograd.grad(scalar, obs_t, retain_graph=False, create_graph=False)
        vec = grads[0].detach().cpu().numpy().squeeze(0)
        return np.asarray(vec, dtype=np.float32)
    finally:
        obs_t.requires_grad_(False)
        if was_training:
            policy.model.train(True)


def format_top_saliency(
    grads: np.ndarray,
    *,
    top_k: int = 6,
    abs_values: bool = True,
    enable_pedestrians: bool | None = None,
    enable_public_transport: bool | None = None,
) -> str:
    vals = np.abs(grads) if abs_values else grads
    dim = len(vals)
    labels = observation_feature_labels(
        dim,
        enable_pedestrians=enable_pedestrians,
        enable_public_transport=enable_public_transport,
    )
    order = np.argsort(-vals)
    parts = []
    for i in order[: min(top_k, dim)]:
        parts.append(f"{labels[i]}={vals[i]:.5g}")
    return ", ".join(parts)


def mean_abs_saliency_across_tls(per_tls_gradients: Sequence[np.ndarray]) -> np.ndarray:
    """Усреднить по светофорам |∂ logπ / ∂obs| (батч строк одной размерности)."""
    abs_k = np.stack(
        [np.abs(np.asarray(g, dtype=np.float64)) for g in per_tls_gradients],
        axis=0,
    )
    return np.mean(abs_k, axis=0)


def log_saliency_tensorboard_scalar_groups(
    writer: Any,
    global_step: int,
    mean_abs_grad_per_dim: np.ndarray,
    *,
    prefix: str = "saliency",
    enable_pedestrians: bool | None = None,
    enable_public_transport: bool | None = None,
) -> None:
    """Скаляры в TensorBoard: среднее |∂logπ/∂obs_i| по TLS и доля в сумме (Scalars)."""
    abs_vals = np.asarray(mean_abs_grad_per_dim, dtype=np.float64).flatten()
    labels = observation_feature_labels(
        len(abs_vals),
        enable_pedestrians=enable_pedestrians,
        enable_public_transport=enable_public_transport,
    )
    total = float(np.sum(abs_vals)) + 1e-12
    for i, name in enumerate(labels):
        writer.add_scalar(f"{prefix}/mean_abs/{name}", float(abs_vals[i]), global_step)
        writer.add_scalar(f"{prefix}/share_sum/{name}", float(abs_vals[i] / total), global_step)

    writer.add_scalar(f"{prefix}/aggregate/l1_norm", float(np.sum(abs_vals)), global_step)

