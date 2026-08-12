from .base import BaseGuidance

import torch

class AdamDPSGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(AdamDPSGuidance, self).__init__(args, **kwargs)
        self.b1, self.b2 = args.beta1, args.beta2
        self.m, self.v, self.s = None, None, 1
        self.one_minus_b1, self.one_minus_b2 = 1 - self.b1, 1 - self.b2
        self.device = args.device

    def reset(self):
        self.m = None
        self.v = None
        self.s = 1

    def adaptive_moment_estimate(self, g):
        if self.m is None and self.v is None:
            self.m = torch.zeros_like(g)
            self.v = torch.zeros_like(g)
        self.m.mul_(self.b1).add_(g, alpha=self.one_minus_b1)
        self.v.mul_(self.b2).add_(torch.square(g), alpha=self.one_minus_b2)

        m_hat = self.m / (1 - self.b1 ** self.s)
        v_hat = self.v / (1 - self.b2 ** self.s)
        self.s += 1

        return m_hat / (torch.sqrt(v_hat) + 1e-8)

    @torch.no_grad()
    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:

        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        with torch.enable_grad():
            x_need_grad = x.clone().detach().requires_grad_()
            epsilon = unet(x_need_grad, t)

        sqrt_one_minus_alpha = (1 - alpha_prod_t) ** 0.5
        sqrt_alpha = alpha_prod_t ** 0.5

        with torch.enable_grad():
            x0 = (x_need_grad - sqrt_one_minus_alpha * epsilon) / sqrt_alpha
            logp_norm = self.guider.get_guidance(x0, return_logp=True, check_grad=False, time=t, **kwargs)
            gradient = torch.autograd.grad(logp_norm.sum(), x_need_grad)[0]

        guidance = self.adaptive_moment_estimate(gradient)

        guidance.mul_(self.args.guidance_strength)

        # Add guidance to noise, fix scale
        alpha_t = alpha_prod_t / alpha_prod_t_prev
        x0 = torch.clamp(x0.detach(), -self.args.clip_sample_range, self.args.clip_sample_range)
        x_prev = self._predict_x_prev_from_zero(
            x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
        x_prev += guidance / alpha_t ** 0.5

        return x_prev