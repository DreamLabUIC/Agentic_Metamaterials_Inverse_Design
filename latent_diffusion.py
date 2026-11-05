"""
latent_diffusion.py
===================

Conditional latent diffusion used for the inverse design step. The model
operates inside the shared VAE latent space: a target-conditioned reverse
chain denoises a standard-normal sample into a latent, which the VAE decoders
then turn into a material-structure design.

The condition is the target stiffness matrix (all 21 components C11 ... C66,
GPa), transformed by ``StiffnessCondition``: log10(C_ii) for the diagonal and
C_ij / sqrt(C_ii C_jj) for the off-diagonal entries, then min-max scaled to
[0, 1]. The diffusion works on the standardised latent (z - mean) / std
(``LatentStandardization``); generated latents are converted back with
z = z_std * std + mean before decoding.

This module is inference-only. It contains the sampling utilities
(``DiffusionSampler`` / ``SubsampledDiffusionSampler``), the beta-schedule and
timestep-spacing helpers, the denoising network ``LatentDenoiser``, and the
condition / latent transforms. All training-loss, DDIM, and Langevin code
paths have been removed.
"""

import enum
import math
import pickle

import numpy as np
import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Beta schedules
# --------------------------------------------------------------------------
def derive_betas_from_alpha_bar(num_steps, alpha_bar_fn, max_beta=0.999):
    """Discretize a continuous alpha-bar(t) function into per-step betas."""
    schedule = []
    for i in range(num_steps):
        t1, t2 = i / num_steps, (i + 1) / num_steps
        schedule.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return np.array(schedule)


def build_beta_schedule(schedule_name, num_steps):
    """Return the beta schedule for the given name ('cosine' or 'linear')."""
    if schedule_name == "linear":
        scale = 1000 / num_steps
        return np.linspace(scale * 1e-4, scale * 0.02, num_steps, dtype=np.float64)
    if schedule_name == "cosine":
        return derive_betas_from_alpha_bar(
            num_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def select_sampling_timesteps(num_timesteps, section_counts):
    """Select the subset of original timesteps to keep.

    Supports the DDIM-style spacing string ``"ddimN"`` used at inference.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            target = int(section_counts[len("ddim"):])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == target:
                    return set(range(0, num_timesteps, stride))
            raise ValueError(
                f"cannot create exactly {target} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]

    per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start, steps = 0, []
    for i, count in enumerate(section_counts):
        size = per + (1 if i < extra else 0)
        if size < count:
            raise ValueError(f"cannot divide {size} steps into {count}")
        stride = 1 if count <= 1 else (size - 1) / (count - 1)
        cur = 0.0
        for _ in range(count):
            steps.append(start + round(cur))
            cur += stride
        start += size
    return set(steps)


def _gather_schedule_values(arr, step, broadcast_shape):
    """Gather 1-D schedule values at ``step`` and broadcast to a shape."""
    out = torch.from_numpy(arr).to(step.device)[step].float()
    while out.dim() < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)


class PredictionTarget(enum.Enum):
    EPSILON = enum.auto()
    START_X = enum.auto()


# --------------------------------------------------------------------------
# Gaussian diffusion (sampling only)
# --------------------------------------------------------------------------
class DiffusionSampler:
    """Forward-noising and reverse-sampling utilities for a diffusion model.

    Only the pieces needed for conditional sampling are kept: the forward
    marginal ``q(x_t | x_0)``, the posterior mean/variance, and the ancestral
    reverse sampler. The denoising network is expected to predict the clean
    latent x_0 directly (``predicts_clean_latent=True``).
    """

    def __init__(self, *, beta_schedule, predicts_clean_latent, rescale_steps=False,
                 clip_range=None, **_ignored):
        self.predicts_clean_latent = predicts_clean_latent
        self.rescale_steps = rescale_steps
        self.clip_range = clip_range

        beta_schedule = np.asarray(beta_schedule, dtype=np.float64)
        assert beta_schedule.ndim == 1 and (beta_schedule > 0).all() and (beta_schedule <= 1).all()
        self.beta_schedule = beta_schedule
        self.step_count = int(beta_schedule.shape[0])

        alpha_schedule = 1.0 - beta_schedule
        self.alpha_bar = np.cumprod(alpha_schedule)
        self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])

        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)
        self.sqrt_inv_alpha_bar = np.sqrt(1.0 / self.alpha_bar)
        self.sqrt_inv_alpha_bar_minus1 = np.sqrt(1.0 / self.alpha_bar - 1)

        self.post_variance = (
            beta_schedule * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        self.post_log_variance_clipped = np.log(
            np.append(self.post_variance[1], self.post_variance[1:]))
        self.post_mean_coef_xstart = (
            beta_schedule * np.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        self.post_mean_coef_xt = (
            (1.0 - self.alpha_bar_prev) * np.sqrt(alpha_schedule)
            / (1.0 - self.alpha_bar))

    def _rescale_step(self, step):
        if self.rescale_steps:
            return step.float() * (1000.0 / self.step_count)
        return step

    def forward_noise(self, clean_latent, step, eps=None):
        """Sample x_t from the forward marginal q(x_t | x_0)."""
        if eps is None:
            eps = torch.randn_like(clean_latent)
        return (_gather_schedule_values(self.sqrt_alpha_bar, step, clean_latent.shape) * clean_latent
                + _gather_schedule_values(self.sqrt_one_minus_alpha_bar, step, clean_latent.shape)
                * eps)

    def _clean_latent_from_eps(self, noisy_latent, step, eps):
        return (_gather_schedule_values(self.sqrt_inv_alpha_bar, step, noisy_latent.shape) * noisy_latent
                - _gather_schedule_values(self.sqrt_inv_alpha_bar_minus1, step, noisy_latent.shape) * eps)

    def posterior_mean_variance(self, clean_latent, noisy_latent, step):
        """Mean/variance of the posterior q(x_{t-1} | x_t, x_0)."""
        post_mean = (_gather_schedule_values(self.post_mean_coef_xstart, step, noisy_latent.shape) * clean_latent
                     + _gather_schedule_values(self.post_mean_coef_xt, step, noisy_latent.shape) * noisy_latent)
        post_var = _gather_schedule_values(self.post_variance, step, noisy_latent.shape)
        post_log_var = _gather_schedule_values(self.post_log_variance_clipped, step, noisy_latent.shape)
        return post_mean, post_var, post_log_var

    def reverse_mean_variance(self, denoiser, noisy_latent, step, condition=None, clip_output=True):
        """Reverse-step distribution p(x_{t-1} | x_t) from the model output."""
        raw_output = denoiser(noisy_latent, self._rescale_step(step), condition)

        log_variance = _gather_schedule_values(
            np.log(np.append(self.post_variance[1], self.beta_schedule[1:])),
            step, noisy_latent.shape)

        def clamp_clean_latent(v):
            if clip_output and self.clip_range is not None:
                return v.clamp(*self.clip_range)
            if clip_output:
                return v.clamp(-1, 1)
            return v

        if self.predicts_clean_latent:
            clean_latent_pred = clamp_clean_latent(raw_output)
        else:
            clean_latent_pred = clamp_clean_latent(
                self._clean_latent_from_eps(noisy_latent, step, raw_output))

        post_mean, _, _ = self.posterior_mean_variance(clean_latent_pred, noisy_latent, step)
        return {"mean": post_mean, "log_variance": log_variance,
                "pred_xstart": clean_latent_pred}

    def reverse_step(self, denoiser, noisy_latent, step, condition=None, clip_output=True):
        """Draw x_{t-1} from the reverse step (ancestral sampling)."""
        step_out = self.reverse_mean_variance(denoiser, noisy_latent, step, condition=condition,
                                              clip_output=clip_output)
        eps = torch.randn_like(noisy_latent)
        step_mask = (step != 0).float().view(-1, *([1] * (noisy_latent.dim() - 1)))
        prev_latent = step_out["mean"] + step_mask * torch.exp(0.5 * step_out["log_variance"]) * eps
        return {"sample": prev_latent, "pred_xstart": step_out["pred_xstart"]}

    def run_reverse_chain(self, denoiser, batch_shape, condition=None, init_noise=None,
                          clip_output=True, device=None, show_progress=False,
                          **_ignored):
        """Run the full reverse chain and return the per-step samples list."""
        return [step_out["sample"] for step_out in self.reverse_chain_steps(
            denoiser, batch_shape, condition=condition, init_noise=init_noise,
            clip_output=clip_output, device=device, show_progress=show_progress)]

    def reverse_chain_steps(self, denoiser, batch_shape, condition=None, init_noise=None,
                            clip_output=True, device=None,
                            show_progress=False, **_ignored):
        """Yield the reverse-step output dict at each timestep, T-1 down to 0."""
        if device is None:
            device = next(denoiser.parameters()).device
        latent = init_noise if init_noise is not None else torch.randn(*batch_shape, device=device)

        step_indices = list(range(self.step_count))[::-1]
        if show_progress:
            from tqdm.auto import tqdm
            step_indices = tqdm(step_indices)

        for step_idx in step_indices:
            step = torch.tensor([step_idx] * batch_shape[0], device=device)
            with torch.no_grad():
                step_out = self.reverse_step(denoiser, latent, step, condition=condition,
                                             clip_output=clip_output)
                yield step_out
                latent = step_out["sample"]


class SubsampledDiffusionSampler(DiffusionSampler):
    """A diffusion process that samples on a subset of the original timesteps."""

    def __init__(self, kept_timesteps, **kwargs):
        self.kept_timesteps = set(kept_timesteps)
        self.step_index_map = []
        self.full_step_count = len(kwargs["beta_schedule"])

        base_diffusion = DiffusionSampler(**kwargs)
        last_alpha_bar, spaced_betas = 1.0, []
        for idx, alpha_bar_val in enumerate(base_diffusion.alpha_bar):
            if idx in self.kept_timesteps:
                spaced_betas.append(1 - alpha_bar_val / last_alpha_bar)
                last_alpha_bar = alpha_bar_val
                self.step_index_map.append(idx)
        kwargs["beta_schedule"] = np.array(spaced_betas)
        super().__init__(**kwargs)

    def reverse_mean_variance(self, denoiser, *args, **kwargs):
        return super().reverse_mean_variance(self._wrap_denoiser(denoiser), *args, **kwargs)

    def _rescale_step(self, step):
        return step   # scaling handled by the wrapped model

    def _wrap_denoiser(self, denoiser):
        if isinstance(denoiser, _TimestepRemappedModel):
            return denoiser
        return _TimestepRemappedModel(denoiser, self.step_index_map, self.rescale_steps,
                                      self.full_step_count)


class _TimestepRemappedModel:
    """Maps spaced timestep indices back onto the original schedule before
    calling the denoising network."""

    def __init__(self, denoiser, step_index_map, rescale_steps, full_step_count):
        self.denoiser = denoiser
        self.step_index_map = step_index_map
        self.rescale_steps = rescale_steps
        self.full_step_count = full_step_count

    def __call__(self, noisy_latent, step, condition=None, **kwargs):
        step_lookup = torch.tensor(self.step_index_map, device=step.device, dtype=step.dtype)
        mapped_step = step_lookup[step]
        if self.rescale_steps:
            mapped_step = mapped_step.float() * (1000.0 / self.full_step_count)
        return self.denoiser(noisy_latent, mapped_step, condition, **kwargs)

    def __getattr__(self, name):
        return getattr(self.denoiser, name)


# --------------------------------------------------------------------------
# Denoising network
# --------------------------------------------------------------------------
class SinusoidalStepEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion timestep."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, step):
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=step.device)
            * -(math.log(10000) / (half_dim - 1)))
        phase = step[:, None] * freqs[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class LatentDenoiser(nn.Module):
    """MLP denoiser that predicts the clean latent x_0.

    The target-property embedding is concatenated at the input of every hidden
    layer, and the timestep embedding is added as a residual at each layer.
    Classifier-free guidance is available at sampling time via
    ``forward_with_cfg``.

    Note: the submodule attribute names below (``property_encoder``,
    ``time_embed``, ``layers``, ``layer_norms``) are load-bearing for the
    existing checkpoint files' state_dict keys and must not be renamed
    without also remapping the checkpoints.
    """

    def __init__(self, latent_dim, hidden_dim, num_layers, cond_dim, cond_embed_dim):
        super().__init__()
        self.num_layers = num_layers

        # target-property (condition) encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(cond_dim, 128), nn.SiLU(),
            nn.Linear(128, cond_embed_dim), nn.SiLU(),
        )
        # timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalStepEmbedding(70),
            nn.Linear(70, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # backbone: property embedding concatenated at each hidden layer's input
        layers = [nn.Linear(latent_dim + cond_embed_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim + cond_embed_dim, hidden_dim)
                   for _ in range(num_layers - 2)]
        layers += [nn.Linear(hidden_dim, latent_dim)]
        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.activation = nn.GELU()

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, noisy_latent, step, condition):
        batch_size = noisy_latent.size(0)
        cond_embed = self.property_encoder(
            torch.nan_to_num(condition.view(batch_size, -1), nan=0.0))
        step_embed = self.time_embed(step)
        for layer_idx in range(self.num_layers - 1):
            noisy_latent = torch.cat((noisy_latent, cond_embed), dim=1)
            noisy_latent = self.layer_norms[layer_idx](self.layers[layer_idx](noisy_latent))
            noisy_latent = self.activation(noisy_latent) + step_embed   # timestep as a per-layer residual
        return self.layers[-1](noisy_latent)

    def forward_with_cfg(self, noisy_latent, step, condition, guidance_scale=1.0):
        """Classifier-free guidance combination at sampling time."""
        if guidance_scale == 1.0 or condition is None:
            return self.forward(noisy_latent, step, condition)
        conditional_pred = self.forward(noisy_latent, step, condition)
        unconditional_pred = self.forward(noisy_latent, step, torch.zeros_like(condition))
        return unconditional_pred + guidance_scale * (conditional_pred - unconditional_pred)


# --------------------------------------------------------------------------
# Condition and latent transforms
# --------------------------------------------------------------------------
class StiffnessCondition:
    """Target stiffness (N, 21) in GPa -> denoiser condition (N, 21) in [0, 1].

    Diagonal entries: log10(C_ii); off-diagonal entries: C_ij / sqrt(C_ii C_jj)
    (dimensionless, sign kept); then min-max scaling with the training ranges.
    Components that are zero in every class (C15, C25, C35, C46) map to 0.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)                 # dict: names, lo, hi
        self.names = list(d["names"])
        self.lo, self.hi = np.array(d["lo"]), np.array(d["hi"])
        self.diag = [self.names.index("C%d%d" % (i, i)) for i in range(1, 7)]
        self.off = [(c, self.names.index("C%s%s" % (n[1], n[1])),
                     self.names.index("C%s%s" % (n[2], n[2])))
                    for c, n in enumerate(self.names) if n[1] != n[2]]

    def transform(self, C21):
        C21 = np.atleast_2d(np.asarray(C21, dtype=float))
        if (C21[:, self.diag] <= 0).any():
            raise ValueError("the diagonal stiffness components C11 ... C66 must be > 0")
        T = C21.copy()
        for c, a, b in self.off:
            T[:, c] = C21[:, c] / np.sqrt(C21[:, a] * C21[:, b])
        T[:, self.diag] = np.log10(C21[:, self.diag])
        return (T - self.lo) / (self.hi - self.lo)


class LatentStandardization:
    """z_std = (z - mean) / std per latent dimension (statistics of the
    training latents); the diffusion model generates z_std."""

    def __init__(self, path, device="cpu"):
        with open(path, "rb") as f:
            d = pickle.load(f)                 # dict: mean, std (numpy arrays)
        self.mean = torch.tensor(d["mean"], dtype=torch.float32, device=device)
        self.std = torch.tensor(d["std"], dtype=torch.float32, device=device)

    def to_latent(self, z_std):
        return z_std * self.std + self.mean
