import torch

from .utils import load_image_dataset, check_grad_fn, rescale_grad
from .networks import image_inverse_operator

class SuperResolution:

    def __init__(self, args, scale_factor=4):

        self.args = args
        self.scale_factor = scale_factor

        self.device = args.device

        self.dataset = load_image_dataset(args.dataset, args.num_samples, is_tuning=args.is_tuning)
        self.load_op()
    
    def load_op(self):
        self.operator = image_inverse_operator.get_operator('super_resolution', scale_factor=self.scale_factor, device=self.device)
        self.noise = image_inverse_operator.get_noise('gaussian', sigma=self.args.gaussian_observation_noise_sigma)
        self.noisy_images = self.noise(self.operator.forward(self.dataset.to(self.device))).cpu()
    
    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):
        if kwargs.get('pseudoinverse', False):
            return self.get_pseudoinverse_guidance(x_need_grad, func, post_process, return_logp, check_grad, **kwargs)
        
        start_idx = self.args.batch_id * self.args.per_sample_batch_size
        end_idx = min((self.args.batch_id + 1) * self.args.per_sample_batch_size, len(self.noisy_images))

        if check_grad:
            check_grad_fn(x_need_grad)
        
        x = post_process(func(x_need_grad))

        noisy_images = self.noisy_images[start_idx:end_idx, ...].to(self.args.device)

        # if mc is performed, noisy_images should have shape (mc_eps.shape[0], x0.shape[0], *x0.shape[1:])
        noisy_images = torch.cat([noisy_images for _ in range(x.shape[0] // noisy_images.shape[0])], dim=0)
        
        difference = noisy_images - self.operator.forward(x)

        # negative L2 norm (the "log_probs" here are actually negative losses)
        log_probs = -torch.norm(difference, p=2, dim=(1, 2, 3))

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)

    @torch.enable_grad()
    def get_pseudoinverse_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        start_idx = self.args.batch_id * self.args.per_sample_batch_size
        end_idx = min((self.args.batch_id + 1) * self.args.per_sample_batch_size, len(self.noisy_images))

        if check_grad:
            check_grad_fn(x_need_grad)

        x = post_process(func(x_need_grad))

        noisy_images = self.noisy_images[start_idx:end_idx, ...].to(self.args.device)
        noisy_images = torch.cat([noisy_images for _ in range(x.shape[0] // noisy_images.shape[0])], dim=0)

        difference = self.operator.transpose(noisy_images) - self.operator.transpose(self.operator.forward(x))
        loss = (difference.detach() * x).sum()

        if return_logp:
            return loss

        grad = torch.autograd.grad(loss.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)