import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, Resize

def convert_age_logp(logp: torch.Tensor):
    newlogp = torch.concat([logp[:, :3].mean(dim=1, keepdim=True), logp[:, 5:].mean(dim=1, keepdim=True)], dim=1)
    return newlogp

class HuggingfaceClassifier(nn.Module):

    def __init__(self, guide_network=None):
        
        super(HuggingfaceClassifier, self).__init__()
        
        self.model = AutoModelForImageClassification.from_pretrained(guide_network)
        
        processor = AutoProcessor.from_pretrained(guide_network)
        
        self.transforms = Compose([
            Resize([processor.size['height'], processor.size['width']]),
            Normalize(mean=processor.image_mean, std=processor.image_std)
        ])

        self.model.eval()

    @torch.enable_grad()
    def forward(self, x):
        '''
            return tensor in the shape of (batch_size, )
        '''
        resized_x = self.transforms(x)
        logits = self.model(resized_x).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        return log_probs

        