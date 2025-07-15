import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

from transformers import is_bitsandbytes_available, is_torchvision_available

from trl.core import flatten_dict


@dataclass
class AlignPropConfig:
    r"""
    Configuration class for the [`AlignPropTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(sys.argv[0])[: -len(".py")]`):
            Name of this experiment (defaults to the file name without the extension).
        run_name (`str`, *optional*, defaults to `""`):
            Name of this run.
        log_with (`Optional[Literal["wandb", "tensorboard"]]`, *optional*, defaults to `None`):
            Log with either `"wandb"` or `"tensorboard"`. Check
            [tracking](https://huggingface.co/docs/accelerate/usage_guides/tracking) for more details.
        log_image_freq (`int`, *optional*, defaults to `1`):
            Frequency for logging images.
        tracker_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the tracker (e.g., `wandb_project`).
        accelerator_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator.
        project_kwargs (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Keyword arguments for the accelerator project config (e.g., `logging_dir`).
        tracker_project_name (`str`, *optional*, defaults to `"trl"`):
            Name of project to use for tracking.
        logdir (`str`, *optional*, defaults to `"logs"`):
            Top-level logging directory for checkpoint saving.
        num_epochs (`int`, *optional*, defaults to `100`):
            Number of epochs to train.
        save_freq (`int`, *optional*, defaults to `1`):
            Number of epochs between saving model checkpoints.
        num_checkpoint_limit (`int`, *optional*, defaults to `5`):
            Number of checkpoints to keep before overwriting old ones.
        mixed_precision (`str`, *optional*, defaults to `"fp16"`):
            Mixed precision training.
        allow_tf32 (`bool`, *optional*, defaults to `True`):
            Allow `tf32` on Ampere GPUs.
        resume_from (`str`, *optional*, defaults to `""`):
            Path to resume training from a checkpoint.
        sample_num_steps (`int`, *optional*, defaults to `50`):
            Number of sampler inference steps.
        sample_eta (`float`, *optional*, defaults to `1.0`):
            Eta parameter for the DDIM sampler.
        sample_guidance_scale (`float`, *optional*, defaults to `5.0`):
            Classifier-free guidance weight.
        train_use_8bit_adam (`bool`, *optional*, defaults to `False`):
            Whether to use the 8bit Adam optimizer from `bitsandbytes`.
        train_learning_rate (`float`, *optional*, defaults to `1e-3`):
            Learning rate.
        train_adam_beta1 (`float`, *optional*, defaults to `0.9`):
            Beta1 for Adam optimizer.
        train_adam_beta2 (`float`, *optional*, defaults to `0.999`):
            Beta2 for Adam optimizer.
        train_adam_weight_decay (`float`, *optional*, defaults to `1e-4`):
            Weight decay for Adam optimizer.
        train_adam_epsilon (`float`, *optional*, defaults to `1e-8`):
            Epsilon value for Adam optimizer.
        train_gradient_accumulation_steps (`int`, *optional*, defaults to `1`):
            Number of gradient accumulation steps.
        train_max_grad_norm (`float`, *optional*, defaults to `1.0`):
            Maximum gradient norm for gradient clipping.
        negative_prompts (`Optional[str]`, *optional*, defaults to `None`):
            Comma-separated list of prompts to use as negative examples.
        truncated_backprop_rand (`bool`, *optional*, defaults to `True`):
            If `True`, randomized truncation to different diffusion timesteps is used.
        truncated_backprop_timestep (`int`, *optional*, defaults to `49`):
            Absolute timestep to which the gradients are backpropagated. Used only if `truncated_backprop_rand=False`.
        truncated_rand_backprop_minmax (`Tuple[int, int]`, *optional*, defaults to `(0, 50)`):
            Range of diffusion timesteps for randomized truncated backpropagation.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the final model to the Hub.
    """

    exp_name: str = os.path.basename(sys.argv[0])[: -len(".py")]
    run_name: str = ""
    seed: int = 0
    target_val_metric: str = "reward_mean"  # Default metric to track
    target_val_mode: str = "max"  # "max" for maximizing, "min" for minimizing
    log_with: Optional[Literal["wandb", "tensorboard"]] = "wandb"
    log_image_freq: int = 1
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    accelerator_kwargs: Dict[str, Any] = field(default_factory=dict)
    project_kwargs: Dict[str, Any] = field(default_factory=dict)
    save_strategy: str = "best"
    tracker_project_name: str = "trl"
    logdir: str = "logs"
    num_epochs: int = 100
    train_samples_per_epoch: int = 128
    num_validation_samples: int = 32
    num_eval_samples: int = 64
    val_frac: float = 0.0
    save_freq: int = 5
    val_freq: int = 1
    eval_len: int = 6# number of images generated
    test_freq: int = 1
    num_checkpoint_limit: int = 5
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    resume_from: str = ""
    sample_num_steps: int = 50
    reward_fn: str = 'hps'
    grad_scale: float = 1
    obj_coeff: float = 1.0
    reg_coeff: float = 0.0
    aesthetic_target: float = 10.0 # TODO (ihounie): why do we need this? AFAIK it's just adding a bias to the reward function, which won't affect gradients
    sample_eta: float = 1.0
    sample_guidance_scale: float = 5.0
    train_prompt_fn: str = 'simple_animals'
    eval_prompt_fn: str = 'eval_simple_animals'
    backprop_strategy: str = 'gaussian'    # gaussian, uniform, fixed
    backprop_kwargs = {'gaussian': {'mean': int(0.8*sample_num_steps), 'std': 5}, 'uniform': {'min': 0, 'max': 50}, 'fixed': {'value': 49}}
    
    train_batch_size: int = 8
    eval_batch_size: int = 8
    train_use_8bit_adam: bool = False
    train_learning_rate: float = 1e-3
    train_adam_beta1: float = 0.9
    train_adam_beta2: float = 0.999
    train_adam_weight_decay: float = 1e-4
    train_adam_epsilon: float = 1e-8
    train_gradient_accumulation_steps: int = 8
    train_max_grad_norm: float = 1.0
    negative_prompts: Optional[str] = None
    
    push_to_hub: bool = False
    
    constrained: bool = True
    dual_learning_rate: float = 1.0
    primals_per_dual: int = 5
    dual_update: str = 'simultaneous'
    constraint_list: str = "" #list of tuples of form (constraint type, constraint threshold, initial dual variable). the first tuple pertains to the objective. the constraint type can be one of the following strings: kl, r_aesthetic, r_imagereward
    constraint_setting_batches: int = 10
    blip_prompt: str = "A photorealistic image of an animal fully in frame, set in its natural habitat, with sharp details, vibrant colors, and soft natural lighting."
    normalize_constraints: bool = True
    use_cached_scale: bool = False

    use_nupi: bool = False
    nupi_nu: float = 0.0
    nupi_kappa_p: float = 10
    nupi_kappa_i: float = 2
    dual_warmup_epochs: int = 10

    project_dir: str = "constrained_rewards"
    
    def to_dict(self):
        output_dict = {}
        for key, value in self.__dict__.items():
            output_dict[key] = value
        return flatten_dict(output_dict)

    def __post_init__(self):
        if self.log_with not in ["wandb", "tensorboard"]:
            warnings.warn(
                "Accelerator tracking only supports image logging if `log_with` is set to 'wandb' or 'tensorboard'."
            )

        if self.log_with == "wandb" and not is_torchvision_available():
            warnings.warn("Wandb image logging requires torchvision to be installed")

        if self.train_use_8bit_adam and not is_bitsandbytes_available():
            raise ImportError(
                "You need to install bitsandbytes to use 8bit Adam. "
                "You can install it with `pip install bitsandbytes`."
            )
