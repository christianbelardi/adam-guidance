from .base import BaseGuidance

import torch
from tasks.utils import rescale_grad


class LGDGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(LGDGuidance, self).__init__(args, **kwargs)

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

        epsilon = unet(x, t)
        x0 = self._predict_x0(x, epsilon, alpha_prod_t, **kwargs)

        # the method name is like "lgd_10"
        eps_btz = int(self.args.guidance_name.split("_")[-1])

        # (eps_btz, bsz, c, h, w)
        mc_eps = torch.stack([self.noise_fn(torch.zeros_like(x), (1-alpha_prod_t)**0.5, **kwargs) for _ in range(eps_btz)])

        # guidance function (we want gradient on xt)
        def func(xt): 
            predict_x0 = (xt - (1 - alpha_prod_t) ** (0.5) * unet(xt, t)) / (alpha_prod_t ** (0.5)) # (bsz, c, h, w)
            # add variance to the predict_x0
            predict_x0 = predict_x0[None] + mc_eps # (eps_btz, bsz, c, h, w)
            # flatten the predict_x0 for logp computation
            flatten_predict_x0 = predict_x0.reshape(-1, *x.shape[1:]) # (eps_btz * bsz, c, h, w)
            return flatten_predict_x0

        with torch.enable_grad():
            # flatten_log_prob: (eps_btz * bsz)
            x_need_grad = x.clone().detach().requires_grad_()
            x0_grad = (x_need_grad - (1 - alpha_prod_t) ** (0.5) * unet(x_need_grad, t)) / (alpha_prod_t ** (0.5))
            flat_x0 = x0_grad[None] + mc_eps

            # loop over MC samples; torch.vmap breaks for some guide networks (e.g. the ImageNet ViT classifier)
            log_probs = []
            for i in range(eps_btz):
                log_prob_i = self.guider.get_guidance(flat_x0[i], return_logp=True, check_grad=False, **kwargs)
                log_probs.append(log_prob_i)

            log_prob = torch.stack(log_probs, dim=0).contiguous()

            # compute logsumexp over eps_btz: log(1/n * sum(exp(log_prob))) = log(1/n) + logsumexp(log_prob)
            # since we only need to get the gradient of log_prob, we can ignore the constant term log(1/n)
            log_prob = torch.logsumexp(log_prob, dim=0)
            # divide by eps_btz to get the expectation
            guidance = torch.autograd.grad(log_prob.sum(), x_need_grad)[0] / eps_btz
            guidance = rescale_grad(guidance, clip_scale=self.args.clip_scale, **kwargs)

        guidance.mul_(self.args.guidance_strength)

        alpha_t = alpha_prod_t / alpha_prod_t_prev
        x0 = torch.clamp(x0.detach(), -self.args.clip_sample_range, self.args.clip_sample_range)
        x_prev = self._predict_x_prev_from_zero(
            x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
        x_prev += guidance / alpha_t ** 0.5

        return x_prev