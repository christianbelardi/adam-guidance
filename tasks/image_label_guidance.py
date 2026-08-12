import torch
from .utils import check_grad_fn, rescale_grad, ban_requires_grad, load_image_dataset_labels

from .networks.resnet import ResNet18
from .networks.huggingface_classifier import HuggingfaceClassifier

from .networks.time_dependent_classifier import create_time_classifier

class ImageLabelGuidance:

    def __init__(self, args, guide_network, targets, device, time=False):
        self.args = args
        self.guide_network = guide_network

        if targets == 'no' or targets == -1 or targets == None:
            self.labels = torch.tensor(load_image_dataset_labels(
                self.args.dataset, num_samples=self.args.num_samples, is_tuning=self.args.is_tuning,
                target=self.args.targets
            )).reshape(-1, 1).to(device)
        else:
            if isinstance(targets, str) and '+' in targets:
                target_val = [int(t) for t in targets.split('+')]
            else:
                # Otherwise try to convert to int directly
                target_val = [int(targets)]

            self.labels = torch.tensor(load_image_dataset_labels(
                self.args.dataset, num_samples=self.args.num_samples, is_tuning=self.args.is_tuning,
                target=target_val
            )).reshape(-1, 1).to(device)


        self._load_model(time=time)

        self.classifier.to(device)
        self.classifier.eval()

        self.get_guidance = self.get_guidance_with_time if time else self.get_guidance_without_time

    def _load_model(self, time: bool = False):

        # load resnet or huggingface model. Register more networks here.
        if time:
            self.classifier = create_time_classifier(guide_network=self.guide_network)
        
        else: 
            if 'resnet' in self.guide_network:
                self.classifier = ResNet18(guide_network=self.guide_network)
            else:
                self.classifier = HuggingfaceClassifier(guide_network=self.guide_network)
        
            ban_requires_grad(self.classifier)

    @torch.enable_grad()
    def get_guidance_with_time(self, x_need_grad, time=0, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        start_idx = self.args.batch_id * self.args.per_sample_batch_size
        end_idx = min((self.args.batch_id + 1) * self.args.per_sample_batch_size, len(self.labels))

        if check_grad:
            check_grad_fn(x_need_grad)
        
        x_need_grad = func(x_need_grad)
        x = post_process(x_need_grad)

        log_probs = self.classifier(x, time, **kwargs)
        log_probs = log_probs.gather(dim=1, index=self.labels[start_idx:end_idx]).flatten()
        
        if return_logp:
            return log_probs
        
        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)

    @torch.enable_grad()
    def get_guidance_without_time(self, x_need_grad, func=lambda x:x, post_process=lambda x:x, return_logp=False, check_grad=True, **kwargs):

        start_idx = self.args.batch_id * self.args.per_sample_batch_size
        end_idx = min((self.args.batch_id + 1) * self.args.per_sample_batch_size, len(self.labels))

        if check_grad:
            check_grad_fn(x_need_grad)

        x = post_process(func(x_need_grad))

        # classifier returns log prob!
        log_probs = self.classifier(x)
        log_probs = log_probs.gather(dim=1, index=self.labels[start_idx:end_idx]).flatten()

        if return_logp:
            return log_probs

        grad = torch.autograd.grad(log_probs.sum(), x_need_grad)[0]

        return rescale_grad(grad, clip_scale=1.0, **kwargs)
        
    