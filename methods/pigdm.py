from .base import BaseGuidance
from tasks.utils import rescale_grad

import torch

class PiGDMGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(PiGDMGuidance, self).__init__(args, **kwargs)

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

        sqrt_one_minus_alpha = (1 - alpha_prod_t) ** 0.5
        sqrt_alpha = alpha_prod_t ** 0.5

        with torch.enable_grad():
            x_need_grad = x.clone().detach().requires_grad_()
            epsilon = unet(x_need_grad, t)
            x0 = (x_need_grad - sqrt_one_minus_alpha * epsilon) / sqrt_alpha
            x0 = torch.clamp(x0, -self.args.clip_sample_range, self.args.clip_sample_range)
            loss = self.guider.get_guidance(x0, return_logp=True, check_grad=False, pseudoinverse=True, time=t, **kwargs)
            guidance = torch.autograd.grad(loss.sum(), x_need_grad)[0]
        guidance = rescale_grad(guidance, clip_scale=1.0, **kwargs)
        guidance.mul_(self.args.guidance_strength)

        alpha_t = alpha_prod_t / alpha_prod_t_prev
        x_prev = self._predict_x_prev_from_zero(
            x, x0.detach(), alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
        x_prev += guidance / alpha_t ** 0.5

        return x_prev