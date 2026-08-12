from dataclasses import dataclass, field
from typing import Literal
from logger.base import BaseLogger

@dataclass
class Arguments:

    # data related
    data_type: Literal['image'] = field(default='image')
    dataset: str = field(default='cifar10')
    task: str = field(default='label_guidance')

    # image related
    image_size: int = field(default=32)

    clip_scale: float = field(default=100)

    # model related
    model_name_or_path: str = field(default='google/ddpm-cifar10-32')

    # diffusion related
    train_steps: int = field(default=1000)
    inference_steps: int = field(default=50)
    eta: float = field(default=1.0)
    clip_x0: bool = field(default=True)
    clip_sample_range: float = field(default=1.0)

    # inference related:
    seed: int = field(default=42)
    device: str = field(default='cuda')
    logging_dir: str = field(default='logs')
    logger: BaseLogger = None       # Initialize upon instantiation
    per_sample_batch_size: int = field(default=128)
    num_samples: int = field(default=2048)
    is_tuning: bool = field(default=False) # enable tuning on different samples than testing
    batch_id: int = field(default=0)    # start from the zero

    # guidance related
    guidance_name: str = field(default='no')
    target: str = field(default=None)
    recur_steps: int = field(default=1)
    iter_steps: int = field(default=1)
    guidance_strength: float = field(default=1.0)

    # specific for tfg method
    rho: float = field(default=1.0)
    mu: float = field(default=1.0)
    sigma: float = field(default=0.01)
    eps_bsz: int = field(default=1)
    rho_schedule: str = field(default='increase')
    mu_schedule: str = field(default='increase')
    sigma_schedule: str = field(default='decrease')

    # specific for adam dps method
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.995)
    delta_x0_guidance_strength: float = field(default=1.0)

    # specific for reddiff method
    lmbda: float = field(default=0.25)

    # cond_fn related
    guide_network: str = field(default='aaraki/vit-base-patch16-224-in21k-finetuned-cifar10')

    # evaluation related
    eval_batch_size: int = field(default=128)

    # logging related
    logging_resolution: int = field(default=64)
    log_suffix: str = field(default='')
    max_show_images: int = field(default=64)
    check_done: bool = field(default=True)

    # wandb related
    wandb: bool = field(default=False)
    wandb_project: str = field(default=None)
    wandb_name: str = field(default=None)
    wandb_entity: str = field(default=None)

    # observation noise
    gaussian_observation_noise_sigma: float = field(default=0.00)
