import torch
import os
from transformers import HfArgumentParser

from .configs import Arguments

from evaluations.image import ImageEvaluator

from diffusion.ddim import ImageSampler

from methods.mpgd import MPGDGuidance
from methods.adam_mpgd import AdamMPGDGuidance
from methods.lgd import LGDGuidance
from methods.base import BaseGuidance
from methods.ugd import UGDGuidance
from methods.adam_dps import AdamDPSGuidance
from methods.dps import DPSGuidance
from methods.tfg import TFGGuidance
from methods.cg import ClassifierGuidance
from methods.adam_cg import AdamClassifierGuidance
from methods.reddiff import RedDiffGuidance
from methods.pigdm import PiGDMGuidance
from methods.adam_pigdm import AdamPiGDMGuidance


def get_logging_dir(arg_dict: dict):
    if arg_dict['guidance_name'] == 'tfg':
        # record rho, mu, sigma with scheduler
        suffix = f"rho={arg_dict['rho']}-{arg_dict['rho_schedule']}+mu={arg_dict['mu']}-{arg_dict['mu_schedule']}+sigma={arg_dict['sigma']}-{arg_dict['sigma_schedule']}"
    elif arg_dict['guidance_name'] == 'reddiff':
        suffix = f"guidance_strength={arg_dict['guidance_strength']}+beta1={arg_dict['beta1']}+beta2={arg_dict['beta2']}+lmbda={arg_dict['lmbda']}"
    elif 'adam' in arg_dict['guidance_name']:
        suffix = f"guidance_strength={arg_dict['guidance_strength']}+beta1={arg_dict['beta1']}+beta2={arg_dict['beta2']}"
    else:
        suffix = "guidance_strength=" + str(arg_dict['guidance_strength'])
        if 'ugd' in arg_dict['guidance_name']:
            suffix += f"+delta_x0_guidance_strength={arg_dict['delta_x0_guidance_strength']}"

    if arg_dict['gaussian_observation_noise_sigma'] != 0.05:
        suffix += f"+observation_noise={arg_dict['gaussian_observation_noise_sigma']}"

    if arg_dict['inference_steps'] != 100:
        suffix += f"+inference_steps={arg_dict['inference_steps']}"

    if arg_dict['is_tuning']:
        suffix += f"+tuning={arg_dict['is_tuning']}"

    return os.path.join(
        arg_dict['logging_dir'],
        f"guidance_name={arg_dict['guidance_name']}+recur_steps={arg_dict['recur_steps']}+iter_steps={arg_dict['iter_steps']}",
        "model=" + arg_dict['model_name_or_path'].replace("/", '_'),
        "guide_net=" + arg_dict['guide_network'].replace('/', '_'),
        "target=" + str(arg_dict['target']).replace(" ", "_"),
        suffix,
    )

def get_config(add_logger=True) -> Arguments:
    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    args.device = torch.device(args.device)

    if add_logger:
        from logger import setup_logger
        args.logging_dir = get_logging_dir(vars(args))
        print("logging to", args.logging_dir)
        setup_logger(args)

    # examine combined guidance

    args.tasks = args.task.split('+')
    args.guide_networks = args.guide_network.split('+')
    args.targets = args.target.split('+')

    assert len(args.tasks) == len(args.guide_networks) == len(args.targets)

    return args


def get_evaluator(args):
    if args.data_type == 'image':
        return ImageEvaluator(args)
    else:
        raise NotImplementedError

def get_guidance(args, network):
    noise_fn = getattr(network, 'noise_fn', None)
    if args.guidance_name == 'no':
        return BaseGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'adam_cg':
        return AdamClassifierGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'cg':
        return ClassifierGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'mpgd':
        return MPGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'ugd':
        return UGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'adam_dps':
        return AdamDPSGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'dps':
        return DPSGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'adam_mpgd':
        return AdamMPGDGuidance(args, noise_fn=noise_fn)
    elif 'lgd' in args.guidance_name:
        return LGDGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'tfg':
        return TFGGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'reddiff':
        return RedDiffGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'pigdm':
        return PiGDMGuidance(args, noise_fn=noise_fn)
    elif args.guidance_name == 'adam_pigdm':
        return AdamPiGDMGuidance(args, noise_fn=noise_fn)
    else:
        raise NotImplementedError

def get_network(args):
    if args.data_type == 'image':
        return ImageSampler(args)
    else:
        raise NotImplementedError
