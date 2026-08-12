import argparse
import os
import sys
import pandas as pd
import yaml
import subprocess
from tune import get_model_name_from_dataset, get_target_from_task, should_minimize, get_guide_network


def find_best_hyperparams(experiment_dir, minimize, search_filename="search_50_150.csv"):
    results_dataframe = pd.read_csv(os.path.join(experiment_dir, search_filename))
    if minimize:
        best_row_index = results_dataframe['metric'].argmin()
    else:
        best_row_index = results_dataframe['metric'].argmax()
    best_row = results_dataframe.iloc[best_row_index]
    print(f"Best hyperparameters found: {best_row.to_dict()}")
    params = {}
    if 'tfg' in experiment_dir:
        params['rho'] = best_row['rho']
        params['mu'] = best_row['mu']
        params['sigma'] = best_row['sigma']
    else:
        params['guidance_strength'] = best_row['guidance_strength']
        if 'adam' in experiment_dir:
            params['beta1'] = best_row['beta1']
            params['beta2'] = best_row['beta2']
        if 'ugd' in experiment_dir:
            params['delta_x0_guidance_strength'] = best_row['delta_x0_guidance_strength']
        if 'reddiff' in experiment_dir:
            params['beta1'] = best_row['beta1']
            params['beta2'] = best_row['beta2']
            params['lmbda'] = best_row['lmbda']

    return params

def run_main_with_params(args, config, params, gpu_id=0):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(repo_root, "main.py"),
        "--data_type", config.get("data_type", "image"),
        "--image_size", str(config.get("image_size", 256)),
        "--dataset", config["dataset"],
        "--model_name_or_path", config["model_name_or_path"],
        "--task", config["task"],
        "--guide_network", config.get("guide_network", "no"),
        "--target", config.get("target", "no"),
        "--train_steps", str(config.get("train_steps", 1000)),
        "--inference_steps", str(config.get("inference_steps", 100)),
        "--eta", str(config.get("eta", 1.0)),
        "--clip_x0", str(config.get("clip_x0", True)),
        "--seed", str(args.seed),
        "--logging_dir", config['output_dir'],
        "--per_sample_batch_size", str(config.get("per_sample_batch_size", 16)),
        "--num_samples", str(args.num_samples),
        "--logging_resolution", str(config.get("logging_resolution", 512)),
        "--guidance_name", config["guidance_name"],
        "--eval_batch_size", str(config.get("eval_batch_size", 16)),
        "--wandb", str(config.get("wandb", False)),
        "--recur_steps", str(config.get("recur_steps", 1)),
        "--iter_steps", str(config.get("iter_steps", 1)),
        "--eps_bsz", str(config.get("eps_bsz", 1)),
    ]
    for param_name, param_value in params.items():
        cmd.extend([f"--{param_name}", str(param_value)])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run main.py with the best hyperparameters from tuning.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the tuning configuration file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to be performed.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="Number of samples to generate.",
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
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for test outputs (e.g. 'reproduced-results'). Best "
             "hyperparameters are still read from the config's logging_dir. "
             "If unset, outputs go alongside the search results (original behavior).",
    )
    parser.add_argument(
        "--search_root",
        type=str,
        default=None,
        help="Directory to read search results from instead of the config's "
             "logging_dir (e.g. 'paper-results/hyperparameter-search' to use "
             "the released paper searches).",
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

    # Best hyperparameters are always read from the search results (read-only).
    search_root = args.search_root if args.search_root is not None else config['logging_dir']
    if args.output_root is not None:
        # Redirect test outputs to a separate tree (e.g. reproduced-results/),
        # leaving the original results/ untouched.
        config['output_dir'] = os.path.join(args.output_root, 'test', config['dataset'], config['task'])
    else:
        config['output_dir'] = os.path.join(os.path.dirname(os.path.dirname(search_root)), 'test', config['dataset'], config['task'])
    config['logging_dir'] = os.path.join(search_root, config['dataset'], config['task'], config['experiment_name'])

    params = find_best_hyperparams(config['logging_dir'], config["minimize"])

    run_main_with_params(args, config, params, args.gpu_id)

if __name__ == "__main__":
    main()
