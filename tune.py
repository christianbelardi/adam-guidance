import argparse
import os
import yaml
from tuning.bayesian_optimization import BayesianGlobalOptimization

def get_model_name_from_dataset(dataset):
    if dataset == 'cat':
        return 'google/ddpm-ema-cat-256'
    elif dataset == 'imagenet':
        return 'openai_imagenet.pt'
    elif dataset == 'cifar10':
        return 'openai_cifar10.pt'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_target_from_task(task):
    if 'super_resolution' in task or 'gaussian_deblur' in task or task == 'inpainting':
        return 'neg_lpips'
    elif task == 'label_guidance' or task == 'label_guidance_time':
        return 'cmmd'
    elif task == 'label_guidance_time_optimize_accuracy':
        return 'accuracy'
    else:
        raise ValueError(f"Unknown task: {task}")

def should_minimize(target_metric):
    return target_metric in ['cmmd']

def get_guide_network(dataset, task):
    if task == 'label_guidance':
        if dataset == 'cifar10':
            return 'resnet_cifar10.pt'
        elif dataset == 'imagenet':
            return 'google/vit-base-patch16-224'
    elif task == 'label_guidance_time' or task == 'label_guidance_time_optimize_accuracy':
        if dataset == 'cifar10':
            return 'timeclassifier_cifar10.pt'
        elif dataset == 'imagenet':
            return 'timeclassifier_imagenet.pt'
    return 'no'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters using Bayesian optimization.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--n_sobol",
        type=int,
        default=50,
        help="Number of Sobol points to sample.",
    )
    parser.add_argument(
        "--n_total",
        type=int,
        default=150,
        help="Total number of search iterations.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to be performed.",
    )
    parser.add_argument(
        "--guidance_name",
        type=str,
        required=True,
        help="Task to be performed.",
    )
    parser.add_argument(
        "--iter_steps",
        type=int,
        required=True,
        help="Task to be performed.",
    )
    parser.add_argument(
        "--override_inference_steps",
        type=int,
        default=None,
        help="Override the inference steps in the config file.",
    )
    parser.add_argument(
        "--gaussian_observation_noise_sigma",
        type=float,
        default=0.0,
        help="Standard deviation of the Gaussian noise added to the observations.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config['model_name_or_path'] = get_model_name_from_dataset(config['dataset'])
    config['task'] = args.task
    config['guide_network'] = get_guide_network(config['dataset'], args.task)
    config['target_metric'] = get_target_from_task(args.task)
    config['minimize'] = should_minimize(config['target_metric'])
    config['guidance_name'] = args.guidance_name
    config['iter_steps'] = args.iter_steps
    if args.override_inference_steps is not None:
        config['inference_steps'] = args.override_inference_steps
    config['gaussian_observation_noise_sigma'] = args.gaussian_observation_noise_sigma

    config['experiment_name'] = f"{config['guidance_name']}"
    if 'tfg' in config['guidance_name'] or 'ugd' in config['guidance_name']:
        config['experiment_name'] += f"_niter{config['iter_steps']}"
    if args.gaussian_observation_noise_sigma > 0:
        config['experiment_name'] += f"-sigma{args.gaussian_observation_noise_sigma}"
    if args.override_inference_steps is not None:
        config['experiment_name'] += f"-inference_steps{args.override_inference_steps}"
    config['logging_dir'] = os.path.join(config['logging_dir'], config['dataset'], config['task'], config['experiment_name'])

    optimizer = BayesianGlobalOptimization(config, args.n_sobol, args.n_total)
    optimizer.run()
