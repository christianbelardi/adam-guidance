import torch
from functools import partial

from .image_label_guidance import ImageLabelGuidance
from .super_resolution import SuperResolution
from .gaussian_deblur import GaussianDeblur
from .inpainting import Inpainting

class BaseGuider:

    def __init__(self, args):
        self.args = args
        self.generator = torch.manual_seed(args.seed)

        self.load_processor()   # e.g., vae for latent diffusion
        self.load_guider()      # guidance network

    def load_processor(self):
        self.processor = lambda x: x

    @torch.enable_grad()
    def process(self, x):
        return self.processor(x)

    @torch.no_grad()
    def load_guider(self):

        self.get_guidance = None

        # for combined guidance
        device = self.args.device

        guiders = []

        for task, guide_network, target in zip(self.args.tasks, self.args.guide_networks, self.args.targets):

            if task == 'label_guidance':
                guider = ImageLabelGuidance(self.args, guide_network, target, device, time=False)
            elif task == 'label_guidance_time' or task == 'label_guidance_time_optimize_accuracy':
                guider = ImageLabelGuidance(self.args, guide_network, target, device, time=True)
            elif task == 'super_resolution':
                guider = SuperResolution(self.args, scale_factor=4)
            elif task == 'super_resolution_8':
                guider = SuperResolution(self.args, scale_factor=8)
            elif task == 'super_resolution_12':
                guider = SuperResolution(self.args, scale_factor=12)
            elif task == 'super_resolution_16':
                guider = SuperResolution(self.args, scale_factor=16)
            elif task == 'gaussian_deblur':
                guider = GaussianDeblur(self.args, intensity=3)
            elif task == 'gaussian_deblur_6':
                guider = GaussianDeblur(self.args, intensity=6)
            elif task == 'gaussian_deblur_9':
                guider = GaussianDeblur(self.args, intensity=9)
            elif task == 'gaussian_deblur_12':
                guider = GaussianDeblur(self.args, intensity=12)
            elif task == 'inpainting':
                guider = Inpainting(self.args)
            else:
                raise NotImplementedError

            guiders.append(guider)

        if len(guiders) == 1:
            self.get_guidance = partial(guider.get_guidance, post_process=self.process)
        else:
            self.get_guidance = partial(self._get_combined_guidance, guiders=guiders)

    def _get_combined_guidance(self, x, guiders, *args, **kwargs):
        values = []
        for guider in guiders:
            values.append(guider.get_guidance(x, post_process=self.process, *args, **kwargs))
        return sum(values)
