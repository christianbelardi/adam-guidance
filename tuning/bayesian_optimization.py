import os
import logger
import torch
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from copy import deepcopy
from pipeline import BasePipeline
from tqdm.auto import trange
from utils.configs import Arguments
from utils.utils import get_evaluator, get_guidance, get_network


def get_observation(args, target_metric):
    # draw new seed for each trial
    args.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # load network and guider
    network = get_network(args)
    guider = get_guidance(args, network)

    # load evaluator and set up pipeline
    evaluator = get_evaluator(args)
    pipeline = BasePipeline(args, network, guider, evaluator)

    # sample and evaluate
    samples = pipeline.sample(args.num_samples)
    torch.cuda.empty_cache()
    metrics = evaluator.evaluate(samples)
    torch.cuda.empty_cache()

    # return target metric
    return metrics[target_metric]

class BayesianGlobalOptimization:
    def __init__(self, config, n_sobol, n_total):
        self.config = config
        self.n_sobol = n_sobol
        self.n_total = n_total
        self.args = self.set_experiment_fixed_args()
        self.ax_client = AxClient(generation_strategy=GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=n_sobol,
                    min_trials_observed=n_sobol,
                    max_parallelism=1,
                    model_kwargs={"seed": 0},
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=1,
                ),
            ]
        ))
        self.ax_client.create_experiment(
            name=config["experiment_name"],
            parameters=self.set_parameters(),
            objectives={"metric": ObjectiveProperties(minimize=config["minimize"])},
        )

    def set_parameters(self):
        if self.config['guidance_name'] == 'tfg':
            return [
                {"name": "rho", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
                {"name": "mu", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
                {"name": "sigma", "type": "range", "bounds": [0.0, 1.0], "value_type": "float", "log_scale": False},
            ]
        elif self.config['guidance_name'] == 'adam_cg':
            return [
                {"name": "guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
                {"name": "beta1", "type": "range", "bounds": [0.0, 0.999], "value_type": "float", "log_scale": False},
                {"name": "beta2", "type": "range", "bounds": [0.0, 0.999], "value_type": "float", "log_scale": False},
            ]
        elif self.config['guidance_name'] == 'cg':
            return [
                {"name": "guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
            ]
        elif 'adam_dps' in self.config['guidance_name'] or 'adam_mpgd' in self.config['guidance_name'] or 'adam_pigdm' in self.config['guidance_name']:
            return [
                {"name": "guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
                {"name": "beta1", "type": "range", "bounds": [0.0, 0.999], "value_type": "float", "log_scale": False},
                {"name": "beta2", "type": "range", "bounds": [0.0, 0.999], "value_type": "float", "log_scale": False},
            ]
        elif self.config['guidance_name'] == 'dps' or 'lgd' in self.config['guidance_name'] or self.config['guidance_name'] == 'mpgd' or self.config['guidance_name'] == 'pigdm':
            return [
                {"name": "guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
            ]
        elif self.config['guidance_name'] == 'ugd':
            return [
                {"name": "guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
                {"name": "delta_x0_guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
            ]
        elif self.config['guidance_name'] == 'reddiff':
            return [
                {"name": "guidance_strength", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
                {"name": "beta1", "type": "range", "bounds": [0.0, 0.999], "value_type": "float", "log_scale": False},
                {"name": "beta2", "type": "range", "bounds": [0.0, 0.999], "value_type": "float", "log_scale": False},
                {"name": "lmbda", "type": "range", "bounds": [1e-3, 10_000], "value_type": "float", "log_scale": True},
            ]
        else:
            raise ValueError(f"Unknown method: {self.config['guidance_name']}")

    def set_experiment_fixed_args(self):
        args = Arguments()
        args.is_tuning = True

        args.data_type = self.config['data_type']
        args.image_size = self.config['image_size']
        args.dataset = self.config['dataset']
        args.model_name_or_path = self.config['model_name_or_path']
        args.task = self.config['task']
        args.guide_network = self.config['guide_network']
        args.target = self.config['target']

        args.train_steps = self.config['train_steps']
        args.inference_steps = self.config['inference_steps']
        args.eta = self.config['eta']
        args.clip_x0 = self.config['clip_x0']

        args.logging_dir = self.config['logging_dir']
        args.per_sample_batch_size = self.config['per_sample_batch_size']
        args.num_samples = self.config['num_samples']
        args.logging_resolution = self.config['logging_resolution']
        args.guidance_name = self.config['guidance_name']
        args.eval_batch_size = self.config['eval_batch_size']
        args.wandb = self.config['wandb']
        args.recur_steps = self.config['recur_steps']
        args.iter_steps = self.config['iter_steps']
        args.eps_bsz = self.config['eps_bsz']
        args.gaussian_observation_noise_sigma = self.config['gaussian_observation_noise_sigma']

        args.tasks = args.task.split('+')
        args.guide_networks = args.guide_network.split('+')
        args.targets = args.target.split('+')

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return args

    def get_candidate_args(self, params):
        args = deepcopy(self.args)
        if self.config['guidance_name'] == 'tfg':
            args.rho = params['rho']
            args.mu = params['mu']
            args.sigma = params['sigma'] # sigma is gamma in the tfg paper
        elif self.config['guidance_name'] == 'adam_cg':
            args.guidance_strength = params['guidance_strength']
            args.beta1 = params['beta1']
            args.beta2 = params['beta2']
        elif self.config['guidance_name'] == 'cg':
            args.guidance_strength = params['guidance_strength']
        elif 'adam_dps' in self.config['guidance_name'] or 'adam_mpgd' in self.config['guidance_name'] or 'adam_pigdm' in self.config['guidance_name']:
            args.guidance_strength = params['guidance_strength']
            args.beta1 = params['beta1']
            args.beta2 = params['beta2']
        elif self.config['guidance_name'] == 'dps' or 'lgd' in self.config['guidance_name'] or self.config['guidance_name'] == 'mpgd' or self.config['guidance_name'] == 'pigdm':
            args.guidance_strength = params['guidance_strength']
        elif self.config['guidance_name'] == 'ugd':
            args.guidance_strength = params['guidance_strength']
            args.delta_x0_guidance_strength = params['delta_x0_guidance_strength']
        elif self.config['guidance_name'] == 'reddiff':
            args.guidance_strength = params['guidance_strength']
            args.beta1 = params['beta1']
            args.beta2 = params['beta2']
            args.lmbda = params['lmbda']
        else:
            raise ValueError(f"Unknown method: {self.config['guidance_name']}")
        return args

    def run(self):
        os.makedirs(self.config['logging_dir'], exist_ok=True)
        if os.path.exists(self.config['logging_dir'] + f"/search_{self.n_sobol}_{self.n_total}.csv"):
            raise ValueError(f"Search file already exists: {self.config['logging_dir']}/search_{self.n_sobol}_{self.n_total}.csv")

        logger.setup_logger(self.args)
        print("logging to", self.args.logging_dir) # will overwrite each trial log intentionally

        for i in trange(self.n_total):
            parameters, trial_index = self.ax_client.get_next_trial()
            candidate_args = self.get_candidate_args(parameters)
            y = get_observation(candidate_args, self.config['target_metric'])
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"metric": (y, None)} # learn uncertainty
            )
            df = exp_to_df(self.ax_client.experiment)
            df.to_csv(self.config['logging_dir'] + f"/search_{self.n_sobol}_{self.n_total}.csv", index=False)

        print("best parameters: ", self.ax_client.get_best_parameters())
