import torch
from torchvision import transforms
from .utils.cmmd import compute_cmmd
from .utils.fid import calculate_fid
from .utils.inception_score import inception_score
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import Accuracy

from .base import BaseEvaluator
from tasks.utils import load_image_dataset, load_image_dataset_labels
from utils.env_utils import *
import logger

class ImageEvaluator(BaseEvaluator):

    def __init__(self, args):
        super(ImageEvaluator, self).__init__()

        self.args = args
    
    @torch.no_grad()
    def _compute_accuracy(self, images, return_preds=False, guide_networks=None):

        if guide_networks is None:
            guide_networks = self.args.guide_networks
        assert len(self.args.guide_networks) == 1, "modified function - only supports one guide network"

        labels = load_image_dataset_labels(
            self.args.dataset, num_samples=self.args.num_samples, is_tuning=self.args.is_tuning,
            target=self.args.targets
        )

        guide_network = guide_networks[0]

        model_path_or_model = COND_VALIDITY_PATH_MAPPING[guide_network]
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_path_or_model)
        model = AutoModelForImageClassification.from_pretrained(model_path_or_model)
        model.eval()
        self.classifier = model
        self.classifier.to(self.args.device)

        probs = self._get_prediction(images, batchsize=self.args.eval_batch_size, guide_network=guide_network, return_probs=True)

        if return_preds:
            return probs.argmax(dim=-1).numpy()

        acc = Accuracy(task="multiclass", num_classes=probs.size(1), top_k=1)
        top2_acc = Accuracy(task="multiclass", num_classes=probs.size(1), top_k=2)
        top3_acc = Accuracy(task="multiclass", num_classes=probs.size(1), top_k=3)
        top5_acc = Accuracy(task="multiclass", num_classes=probs.size(1), top_k=5)
        top10_acc = Accuracy(task="multiclass", num_classes=probs.size(1), top_k=10)

        accuracy = acc(probs, torch.tensor(labels))
        top2_accuracy = top2_acc(probs, torch.tensor(labels))
        top3_accuracy = top3_acc(probs, torch.tensor(labels))
        top5_accuracy = top5_acc(probs, torch.tensor(labels))
        top10_accuracy = top10_acc(probs, torch.tensor(labels))

        return accuracy.item(), top2_accuracy.item(), top3_accuracy.item(), top5_accuracy.item(), top10_accuracy.item()

    @torch.no_grad()
    def _compute_cmmd(self, images):
        sample_img = torch.stack([transforms.ToTensor()(img) for img in images], dim=0).to("cpu")
        target_img = load_image_dataset(
            self.args.dataset, num_samples=self.args.num_samples, is_tuning=self.args.is_tuning,
            target=self.args.targets, return_tensor=True, normalize=False
        ).to("cpu")
        return compute_cmmd(target_img, sample_img).item()

    @torch.no_grad()
    def _compute_lpips(self, images):
        # compute metric on cpu instead of self.args.device
        # TODO: batch and put on device
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to("cpu")
        
        target_img = load_image_dataset(
            self.args.dataset, num_samples=self.args.num_samples, is_tuning=self.args.is_tuning,
            target=self.args.targets, return_tensor=True, normalize=False
        ).to("cpu")

        sample_img = torch.stack([transforms.ToTensor()(img) for img in images], dim=0).to("cpu")
        lpips_score = lpips(sample_img, target_img)
        return lpips_score.item()

    @torch.no_grad()
    def _get_prediction(self, samples, batchsize=None, guide_network=None, return_probs=False):
        assert self.feature_extractor is not None
        assert self.classifier is not None

        if guide_network is None:
            guide_network = self.args.guide_network

        if batchsize is None:
            batchsize = len(samples)
        
        # iterate over the samples with batch size
        for i in range(0, len(samples), batchsize):
            inputs = self.feature_extractor(samples[i:i+batchsize], return_tensors="pt")
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            outputs = self.classifier(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

            if i == 0:
                all_probs = probs
            else:
                all_probs = torch.cat([all_probs, probs], dim=0)

        if return_probs:
            return all_probs.cpu()
        return all_probs.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def _compute_fid(self, samples, dataset, target):

        ref_images = load_image_dataset(
            dataset, num_samples=-1, is_tuning=self.args.is_tuning,
            target=target, return_tensor=False
        )

        fid = calculate_fid(ref_images, samples, self.args.eval_batch_size, self.args.device)

        return fid

    def _compute_inception_score(self, samples):
        scores = inception_score(samples, self.args.device, batch_size=self.args.eval_batch_size)
        return scores

    def evaluate(self, samples):
        metrics = {}
        logger.log(f"Evaluating {len(samples)} samples")

        if self.args.is_tuning: # eval only target metric
            assert len(self.args.tasks) == 1, "only one task is supported for tuning"
            task = self.args.tasks[0]
            if 'super_resolution' in task or 'gaussian_deblur' in task or 'inpainting' in task:
                lpips = self._compute_lpips(samples)
                metrics['neg_lpips'] = -lpips
            elif task == 'label_guidance' or task == 'label_guidance_time':
                metrics['cmmd'] = self._compute_cmmd(samples)
            elif task == 'label_guidance_time_optimize_accuracy':
                accuracy, _, _, _, _ = self._compute_accuracy(samples)
                metrics['accuracy'] = accuracy
            else:
                raise NotImplementedError(f"Task {task} not supported for tuning yet")

        else: # eval all relevant metrics
            inception_score = self._compute_inception_score(samples)
            metrics['inception_score'] = inception_score

            # we only allow combined guidance within the same dataset
            if self.args.dataset in ['imagenet', 'cifar10', 'cat']:
                fid = self._compute_fid(samples, self.args.dataset, self.args.targets)
                metrics['fid'] = fid
            
            if 'label_guidance' in self.args.tasks or 'label_guidance_time' in self.args.tasks or 'label_guidance_time_optimize_accuracy' in self.args.tasks:
                accuracy, top2_accuracy, top3_accuracy, top5_accuracy, top10_accuracy = self._compute_accuracy(samples)
                metrics['accuracy'] = accuracy
                metrics['top2_accuracy'] = top2_accuracy
                metrics['top3_accuracy'] = top3_accuracy
                metrics['top5_accuracy'] = top5_accuracy
                metrics['top10_accuracy'] = top10_accuracy

            if any(['super_resolution' in task or 'gaussian_deblur' in task or 'inpainting' in task for task in self.args.tasks]):
                lpips = self._compute_lpips(samples)
                metrics['neg_lpips'] = -lpips

        return metrics
