from .base import BaseGuidance

import torch

class RedDiffGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(RedDiffGuidance, self).__init__(args, **kwargs)
        self.lr = args.guidance_strength
        self.b1 = args.beta1
        self.b2 = args.beta2
        self.lmbda = args.lmbda
        self.device = args.device

    def reset(self):
        self.mu = torch.zeros(
            self.args.per_sample_batch_size, 3,
            self.args.image_size, self.args.image_size,
            device=self.device
        ).requires_grad_(True)
        self.optim = torch.optim.Adam(
            [self.mu], lr=self.lr, betas=(self.b1, self.b2)
        )

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
        t = ts[t]

        sqrt_alpha_prod_t = alpha_prod_t.sqrt()
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t).sqrt()
        with torch.no_grad():
            eps = torch.randn_like(self.mu, device=self.device)
            xt = sqrt_alpha_prod_t * self.mu + sqrt_one_minus_alpha_prod_t * eps
            pred_eps = unet(xt, t).detach()

        self.optim.zero_grad()
        with torch.enable_grad():
            mu = self.mu.clone().detach().requires_grad_(True)
            loss = self.guider.get_guidance(mu, return_logp=True, check_grad=False, time=t, **kwargs)
            gradient = torch.autograd.grad(loss.sum(), mu)[0]

        gradient = -gradient + self.lmbda * (pred_eps - eps)
        self.mu.grad = gradient

        self.optim.step()

        return self.mu.clone().detach()