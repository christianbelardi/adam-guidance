from .base import BaseGuidance

import torch

class AdamMPGDGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(AdamMPGDGuidance, self).__init__(args, **kwargs)
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

        m_hat = self.m / (1 - self.b1 ** self.s) # TODO: could precompute this as a tensor and then index
        v_hat = self.v / (1 - self.b2 ** self.s)
        self.s += 1

        return m_hat / (torch.sqrt(v_hat) + 1e-8)

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
        """
            This function implements Algorithm 1 in MPGD paper.
        """

        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]
        t = ts[t]

        eps = unet(x, t)

        # predict inital x0 using xt and epsilon
        x0 = self._predict_x0(x, eps, alpha_prod_t, **kwargs)

        # update x0 using guidance according to MPGD paper
        guidance = self.guider.get_guidance(x0.clone().detach().requires_grad_(), **kwargs)
        guidance = self.adaptive_moment_estimate(guidance)
        x0 += self.args.guidance_strength * guidance

        # update xt using x0 and call DDIM sample prediction
        xt = (1 - alpha_prod_t) ** (0.5) * eps + alpha_prod_t ** (0.5) * x0

        x_prev = self._predict_x_prev_from_zero(
            xt, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs
        )
        
        return x_prev