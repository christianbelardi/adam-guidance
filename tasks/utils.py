import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from torchvision import transforms

from utils.env_utils import *

def ban_requires_grad(module):
    for param in module.parameters():
        param.requires_grad = False

def check_grad_fn(x_need_grad):
    assert x_need_grad.requires_grad, "x_need_grad should require grad"


def rescale_grad(
    grad: torch.Tensor, clip_scale, **kwargs
):  # [B, N, 3+5]
    node_mask = kwargs.get('node_mask', None)

    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:  # [B, N, 1]
        scale: torch.Tensor = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale  # [B]
        grad = grad * co_ef.view(-1, 1, 1)

    return grad


def load_image_dataset(dataset, num_samples=-1, is_tuning=False, target=-1, return_tensor=True, normalize=True):

    if dataset == 'cat':
        images = load_dataset("cats_vs_dogs")
        images = images.filter(lambda x: x == 0, input_columns='labels')

        images = images['train'].train_test_split(train_size=32, seed=42)
        if is_tuning:
            images = images['train']
        else:
            images = images['test']

        if num_samples > 0:
            images = images[:num_samples]
        images = [image.convert('RGB') for image in images['image']]
        if not return_tensor:
            images = [images.resize((256, 256)) for images in images]
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    elif dataset == 'cifar10':
        dataset = load_dataset('cifar10')

        if isinstance(target, int) and target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')

        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('img', 'images')
        dataset = concatenate_datasets([dataset['train'], dataset['test']])

        dataset = dataset.train_test_split(train_size=32, seed=42)
        if is_tuning:
            dataset = dataset['train']
        else:
            dataset = dataset['test']

        if num_samples > 0:
            dataset = dataset[:num_samples]

        images = [images.resize((32, 32)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif dataset == 'imagenet':
        dataset = load_from_disk(IMAGENET_PATH)

        if isinstance(target, int) and target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')

        dataset = dataset.remove_columns('label')
        dataset = dataset.rename_column('image', 'images')

        dataset = dataset.train_test_split(train_size=32, seed=42)
        if is_tuning:
            dataset = dataset['train']
        else:
            dataset = dataset['test']

        if num_samples > 0:
            dataset = dataset[:num_samples]

        images = [images.resize((256, 256)) for images in dataset['images']]
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    else:
        raise NotImplementedError

    if normalize:
        tf.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if return_tensor:
        image_tensors = [tf(img) for img in images]
        return torch.stack(image_tensors, dim=0)
    else:
        return images


def load_image_dataset_labels(dataset, num_samples=-1, is_tuning=False, target=-1):

    if dataset == 'cifar10':
        dataset = load_dataset('cifar10')

        if isinstance(target, list) and isinstance(target[0], int) and target[0] != -1:
            dataset = dataset.filter(lambda x: x in target, input_columns='label')
        elif isinstance(target, list) and isinstance(target[0], str) and target[0] != 'no':
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')

        dataset = dataset.remove_columns('img')
        dataset = dataset.rename_column('label', 'labels')
        dataset = concatenate_datasets([dataset['train'], dataset['test']])

        dataset = dataset.train_test_split(train_size=32, seed=42)
        if is_tuning:
            dataset = dataset['train']
        else:
            dataset = dataset['test']

        if num_samples > 0:
            dataset = dataset[:num_samples]

        labels = [label for label in dataset['labels']]

    elif dataset == 'imagenet':
        dataset = load_from_disk(IMAGENET_PATH)

        if isinstance(target, int) and target != -1:
            dataset = dataset.filter(lambda x: x in [int(tar) for tar in target], input_columns='label')

        dataset = dataset.remove_columns('image')
        dataset = dataset.rename_column('label', 'labels')

        dataset = dataset.train_test_split(train_size=32, seed=42)
        if is_tuning:
            dataset = dataset['train']
        else:
            dataset = dataset['test']

        if num_samples > 0:
            dataset = dataset[:num_samples]

        labels = [label for label in dataset['labels']]

    else:
        raise NotImplementedError

    return labels
