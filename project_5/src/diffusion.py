"""DDPM noise schedule and sampling routines."""
import torch


class GaussianDiffusion:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.n_steps = n_steps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    def _gather(self, source, timesteps, sample_shape):
        gathered = source.gather(0, timesteps)
        reshape = [-1] + [1] * (len(sample_shape) - 1)
        return gathered.view(*reshape)

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self._gather(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar = self._gather(
            self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels, device=None):
        device = device or self.device
        x = torch.randn(shape, device=device)

        for step in reversed(range(self.n_steps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = model(x, timesteps, labels)

            beta = self.betas[step]
            sqrt_recip_alpha = self.sqrt_recip_alphas[step]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[step]

            mean = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus_alpha_bar)
            if step > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[step]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        return x
