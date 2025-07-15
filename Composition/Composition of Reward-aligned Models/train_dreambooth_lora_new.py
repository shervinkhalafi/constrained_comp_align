import sys
import os

# Add the project root to sys.path
# Assumes the script is in implementations/Product_Composition_Correct/stablediff_rewards/
# and the 'implementations' directory is at the project root.
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Explicitly add the parent directory of AlignProp (project_root/implementations) to sys.path.
# This allows Python to find the 'AlignProp' package within the 'implementations' directory
# for imports like 'from AlignProp.rewards import ...'.
implementations_path = os.path.abspath(os.path.join(project_root, "implementations"))
if implementations_path not in sys.path:
    sys.path.insert(1, implementations_path) # Insert after project_root to maintain order preference

# Also add the AlignProp directory itself to sys.path,
# to assist with intra-package imports within AlignProp if needed.
align_prop_dir_path = os.path.abspath(os.path.join(project_root, "implementations", "AlignProp"))
if align_prop_dir_path not in sys.path:
    sys.path.insert(2, align_prop_dir_path) # Insert with a lower preference

import argparse
import copy
import gc
import logging
import math
import os
import warnings
from pathlib import Path
from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler, 
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from pipeline_stable_diffusion_extended import StableDiffusionPipelineExtended
from scheduling_ddim_extended import DDIMSchedulerExtended
import wandb
from my_utils import dual_step, validation, sample, primal_step_new, compute_KL, compute_KL_weighted
from peft import get_peft_model

# Attempt to import reward functions for validation
# These will be None if not found, and handled gracefully later.
from AlignProp.rewards import *

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA DreamBooth - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )
    tags = ["text-to-image", "diffusers", "lora", "diffusers-training"]
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(["stable-diffusion", "stable-diffusion-diffusers"])
    else:
        tags.extend(["if", "if-diffusers"])
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))




def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        + pipeline_args["prompt"]
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=False)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    guidance_scales = np.linspace(0.0, args.guidance_scale, num = args.num_validation_images)
    if args.validation_images is None:
        images = []
        for l in range(args.num_validation_images):
            
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, generator=generator, num_inference_steps = args.num_inference_steps_val, guidance_scale = guidance_scales[l]).images[0]
                images.append(image)
    else:
        images = []
        for image in args.validation_images:
            image = Image.open(image)
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, image=image, generator=generator, guidance_scale = args.guidance_scale).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: " + pipeline_args["prompt"] + f", guidance scale = {guidance_scales[i]}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--project_name", type=str, default='dreambooth_lora_test', required=False, help="A folder containing the training data of class images.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="The integration to report the results and logs to. Supported platforms are `\"tensorboard\"` (default), `\"wandb\"` and `\"comet_ml\"`. Use `\"all\"` to report to all integrations.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default="lora-dreambooth-model", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='nota-ai/bk-sdm-v2-tiny', required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")


    #Validation parameters
    parser.add_argument("--num_inference_steps_val", type=int, default=50, help="minimum diffusion timestep of finetuning interval")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--num_images_per_prompt", type=int, default=2, help="number of images per prompt")
    parser.add_argument("--validation_epochs", type=int, default=50, help="Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images`.")
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, required=False, help="guidance scale for validation")
    parser.add_argument("--validation_rewards", type=str, default="mps,aesthetic,hps,pickscore,imagereward", help="Comma-separated list of reward functions to evaluate after validation (e.g., 'mps,aesthetic').")

    #dual parameters
    parser.add_argument("--lr_dual", type=float, default=1.0, help="dual learning rate")
    parser.add_argument("--constrained", type=int, default=0, help="whether to run constrained training algo")
    parser.add_argument("--const_thresholds", type=str, default="0.0", help="constraint thresholds")
    parser.add_argument("--initial_dual_mults", type=str, default=None, required=False, help="initial dual multipliers")
    parser.add_argument("--primal_per_dual", type=int, default=1, help="number of primal steps per dual step")
    parser.add_argument("--KL_num_batches", type=int, default=1, help="number of batches sampled for KL divergence computation")
    parser.add_argument("--KL_batch_size", type=int, default=1, help="batch size for KL divergence computation")
    parser.add_argument("--initial_dual_steps", type=int, default=2, help="number of initial dual steps")
    #training parameters
    parser.add_argument("--prompts", type=str, default="photo of a dog", required=False, help="prompts for training")
    parser.add_argument("--rank", type=int, default=16, help="The dimension of the LoRA update matrices.")
    parser.add_argument("--num_inference_steps_train", type=int, default=50, help="number of inference steps for images sampled fortraining")
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--max_timestep", type=int, default=999, help="maximum diffusion timestep of finetuning interval")
    parser.add_argument("--min_timestep", type=int, default=1, help="minimum diffusion timestep of finetuning interval")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `\"latest\"` to automatically select the last available checkpoint.")

    #primal learning rate parameters
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type to use. Choose between [\"linear\", \"cosine\", \"cosine_with_restarts\", \"polynomial\", \"constant\", \"constant_with_warmup\"]")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate (after the potential warmup period) to use.")
    



    parser.add_argument("--classifier_class_names", type=str, default=None, required=False, help="minimum diffusion timestep of finetuning interval")
    parser.add_argument("--load_model_only", type=int, default=0, help="")
    parser.add_argument("--load_model_dir", type=str, default=None, help="directory of model to load if resuming from checkpoint")   
    
    
    
    
    
    
    #DO NOT TOUCH
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--instance_data_dir", type=str, default=None, required=True, help="A folder containing the training data of instance images.")
    parser.add_argument("--class_data_dir", type=str, default=None, required=False, help="A folder containing the training data of class images.")
    parser.add_argument("--instance_prompt", type=str, default=None, required=True, help="The prompt with identifier specifying the instance")
    parser.add_argument("--class_prompt", type=str, default=None, help="The prompt to specify images in the same class as provided instance images.")
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--num_class_images", type=int, default=100, help="Minimal class images for prior preservation loss. If there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.")
    parser.add_argument("--center_crop", default=False, action="store_true", help="Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    parser.add_argument("--prior_generation_precision", type=str, default=None, choices=["no", "fp32", "fp16", "bf16"], help="Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--pre_compute_text_embeddings", action="store_true", help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.")
    parser.add_argument("--tokenizer_max_length", type=int, default=None, required=False, help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.")
    parser.add_argument("--text_encoder_use_attention_mask", action="store_true", required=False, help="Whether to use attention mask for the text encoder")
    parser.add_argument("--validation_images", required=False, default=None, nargs="+", help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.")
    parser.add_argument("--class_labels_conditioning", required=False, default=None, help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.")
    parser.add_argument("--adapters", required=False, default=None, help="The path to the unet `adapters` to compose using minmax DKL product composition.")
    
    parser.add_argument("--gen_mode", required=False, default="cfg", help="The mode to generate images, available values are `cfg`, `unconditional`, `conditional`.")
    parser.add_argument("--constraints_kl_mode", required=False, default="cfg", help="The mode to compute KL divergence, available values are `unconditional`, `conditional`, `cfg`.")
    parser.add_argument("--init_mixture_weights", required=False, action="store_true", help="Whether to initialize the mixture weights")
    
    # Reward Normalization Arguments
    parser.add_argument("--normalize_rewards", action="store_true", help="Whether to normalize rewards based on initial statistics.")
    parser.add_argument("--reward_norm_batches", type=int, default=10, help="Number of batches to use for estimating reward statistics if normalize_rewards is True.")
    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(self.instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        self.instance_images_path = list(Path(self.instance_data_root).iterdir())
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example
    

class DreamBoothDatasetSampled(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        train_batch_size=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            # if the instance data root does not exist, we will create a new one
            self.instance_data_root.mkdir(parents=True, exist_ok=True)

        # self.instance_images_path = list(Path(instance_data_root).iterdir())
        # apend device/gpu id to the image name in order to avoid overwriting
        self.instance_images_path = [Path(instance_data_root) / f"image_{i}.png" for i in range(train_batch_size)]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    def sample_images(self, accelerator, pipe, num_inference_steps_val, guidance_scale, validation_prompt, adapters, mu, num_validation_images, return_type = "PIL", mode="cfg"):
        images = sample(accelerator, pipe, num_inference_steps_val, guidance_scale, validation_prompt, adapters, mu, num_validation_images, return_type = "PIL", mode=mode)
        # Delete all images in instance data root
        for img_path in self.instance_images_path:
            if img_path.is_file():
                os.remove(img_path)
        # Save images to instance data root
        for i, image in enumerate(images):
            image.save(self.instance_images_path[i])


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example



def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def load_lora_adapters(pipeline, adapter_paths_input):
    """
    Loads multiple LoRA adapters into the PEFT-wrapped model.
    If the same adapter path is provided multiple times, it's loaded
    with a unique PEFT adapter name each time (e.g., basename_0, basename_1).
    Args:
        pipeline: The diffusion pipeline.
        adapter_paths_input (list[str]): List of paths to adapter parent directories.
                                         It's assumed each path `p` contains `p/lora_adapters.pth`
                                         or is a directory that load_lora_weights can handle.
    Returns:
        list[str]: A list of unique PEFT adapter names corresponding to each input path.
    """
    lora_names = []
    for adapter_dir_path in adapter_paths_input:
        # Derive a base name from the adapter directory path. Using Path().name is robust.
        base_name = adapter_dir_path.split("/")[-2]
        #breakpoint()

        # Construct a unique adapter name for PEFT by appending the index
        unique_peft_adapter_name = f"{base_name}"

        
        lora_weights_source_path = os.path.join(adapter_dir_path, "lora_adapters.pth")

        # Log which path is being used (directory vs specific file) for clarity
        # This check is illustrative; actual loading depends on how `pipeline.load_lora_weights` handles the path.
        # Assuming the original `os.path.join(adapter_dir_path, "lora_adapters.pth")` was intended.
        
        print(f"Attempting to load LoRA from source '{lora_weights_source_path}' as PEFT adapter '{unique_peft_adapter_name}'")
        
        pipeline.load_lora_weights(
            lora_weights_source_path, # Path to the LoRA weights (file or directory)
            adapter_name=unique_peft_adapter_name
        )
        print(f"Successfully loaded PEFT adapter '{unique_peft_adapter_name}' from source '{lora_weights_source_path}'")
        lora_names.append(unique_peft_adapter_name)
    return lora_names


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    adapters = args.adapters.split(", ")
    print(f"Using {len(adapters)} pretrained adapters")
    for adapter in adapters:
        print(adapter)
    print("*"*100)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    parsed_validation_rewards = []
    if args.validation_rewards:
        parsed_validation_rewards = [r.strip() for r in args.validation_rewards.split(',')]
        if "mps" in parsed_validation_rewards and mps_loss_fn is None:
            raise ValueError("MPS reward requested for validation, but mps_loss_fn could not be imported. It will be skipped.")
        if "aesthetic" in parsed_validation_rewards and aesthetic_loss_fn is None:
            raise ValueError("Aesthetic reward requested for validation, but aesthetic_loss_fn could not be imported. It will be skipped.")
        if "hps" in parsed_validation_rewards and hps_loss_fn is None:
            raise ValueError("HPS reward requested for validation, but hps_loss_fn could not be imported. It will be skipped.")
        if "pickscore" in parsed_validation_rewards and pickscore_loss_fn is None:
            raise ValueError("PickScore reward requested for validation, but pickscore_loss_fn could not be imported. It will be skipped.")
        if "imagereward" in parsed_validation_rewards and imagereward_loss_fn is None:
            raise ValueError("ImageReward reward requested for validation, but imagereward_loss_fn could not be imported. It will be skipped.")


    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
                variant=args.variant,
                #use_lora=True,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    noise_scheduler = DDIMSchedulerExtended.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    
    #NEW#
    pipe = StableDiffusionPipelineExtended.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
        
    pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
    vae = pipe.vae
    
    pipe.unet.eval()
    text_encoder = pipe.text_encoder


    # Freeze vae, text_encoder, unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    
    pipe.to(accelerator.device, dtype=weight_dtype)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    

    adapter_list = load_lora_adapters(pipe, adapters)
    m = len(adapter_list)
    if m > 0:
        mu = [1 / m for _ in range(m)]
    else:
        if args.initial_dual_mults is not None:
            print(f"Initial dual multipliers: {args.initial_dual_mults}")
            mu = args.initial_dual_mults.split(", ")
            mu = [float(s.strip()) for s in mu]
        else: # args.initial_dual_mults is None
            print("Initial dual multipliers not provided. Defaulted to equal multipliers for all losses (original logic).")
        
        if m > 0: # Only print mu if it's not empty
            print(f"Mu: {mu}")
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionLoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.append(text_encoder_)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.append(text_encoder)


    if args.pre_compute_text_embeddings:

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            return prompt_embeds

        pre_computed_encoder_hidden_states = compute_text_embeddings(args.instance_prompt)
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        if args.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(args.validation_prompt)
        else:
            validation_prompt_encoder_hidden_states = None

        if args.class_prompt is not None:
            pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(args.class_prompt)
        else:
            pre_computed_class_prompt_encoder_hidden_states = None

        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states = None
        validation_prompt_encoder_hidden_states = None
        validation_prompt_negative_prompt_embeds = None
        pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    
    #NEW
    dataloaders = []
    iterators = []
    instance_dirs = args.instance_data_dir.split(', ')

    train_dataset_sampled = DreamBoothDatasetSampled(
        instance_data_root=instance_dirs[0],
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
        train_batch_size=args.train_batch_size,
    )

    train_dataloader_sampled = torch.utils.data.DataLoader(
        train_dataset_sampled,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    dataloaders += [accelerator.prepare(train_dataloader_sampled)]
    iterators += [cycle(dataloaders[0])]


    


    # Initialize validation reward functions
    validation_reward_fns_dict = {}
    if parsed_validation_rewards: # Initialize only on main process where eval will hap
        if "mps" in parsed_validation_rewards and mps_loss_fn is not None:
            validation_reward_fns_dict['mps'] = mps_loss_fn(
                device=accelerator.device,
                inference_dtype=weight_dtype
            )
            logger.info("Initialized MPS reward function for validation.")
        
        if "aesthetic" in parsed_validation_rewards and aesthetic_loss_fn is not None:
            validation_reward_fns_dict['aesthetic'] = aesthetic_loss_fn(
                device=accelerator.device,
                accelerator=accelerator, # aesthetic_loss_fn from AlignProp needs this
                torch_dtype=weight_dtype
            )
            logger.info("Initialized Aesthetic reward function for validation.")
        if "hps" in parsed_validation_rewards and hps_loss_fn is not None:
            validation_reward_fns_dict['hps'] = hps_loss_fn(
                device=accelerator.device,
                inference_dtype=weight_dtype
            )
            logger.info("Initialized HPS reward function for validation.")
        if "pickscore" in parsed_validation_rewards and pickscore_loss_fn is not None:
            validation_reward_fns_dict['pickscore'] = pickscore_loss_fn(
                device=accelerator.device,
                inference_dtype=weight_dtype
            )
            logger.info("Initialized PickScore reward function for validation.")
        if "imagereward" in parsed_validation_rewards and imagereward_loss_fn is not None:
            validation_reward_fns_dict['imagereward'] = imagereward_loss_fn(
                device=accelerator.device,
                inference_dtype=weight_dtype
            )
            logger.info("Initialized ImageReward reward function for validation.")


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_sampled) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # set requires grad to false for all parameters
    for param in pipe.unet.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False


    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        pipe.unet, pipe.text_encoder, optimizer, train_dataloader_sampled, lr_scheduler = accelerator.prepare(
            pipe.unet, pipe.text_encoder, optimizer, train_dataloader_sampled, lr_scheduler
        )
    else:
        unet, train_dataloader_sampled, = accelerator.prepare(
            pipe.unet, train_dataloader_sampled
        )
        
    pipe.unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_sampled) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers(project_name = args.project_name, config=tracker_config)

    # ---- Begin Reward Normalization Logic ----
    reward_baseline_stats = {} # To store mean and std for each reward

    def initialize_reward_functions(do_normalize=False, baseline_stats=None):
        # Initializes or re-initializes reward functions
        # baseline_stats should be a dict like {'mps': {'mean': m, 'std': s}, ...}
        if baseline_stats is None:
            baseline_stats = {}

        initialized_fns = {}
        if "mps" in parsed_validation_rewards and mps_loss_fn is not None:
            mean, std = (baseline_stats.get('mps', {}).get('mean', 0.0), baseline_stats.get('mps', {}).get('std', 1.0)) if do_normalize else (0.0, 1.0)
            initialized_fns['mps'] = mps_loss_fn(
                device=accelerator.device, inference_dtype=weight_dtype,
                do_normalize=do_normalize, baseline_mean=mean, baseline_std=std
            )
        if "aesthetic" in parsed_validation_rewards and aesthetic_loss_fn is not None:
            mean, std = (baseline_stats.get('aesthetic', {}).get('mean', 0.0), baseline_stats.get('aesthetic', {}).get('std', 1.0)) if do_normalize else (0.0, 1.0)
            initialized_fns['aesthetic'] = aesthetic_loss_fn(
                device=accelerator.device, accelerator=accelerator, torch_dtype=weight_dtype,
                do_normalize=do_normalize, baseline_mean=mean, baseline_std=std
            )
        if "hps" in parsed_validation_rewards and hps_loss_fn is not None:
            mean, std = (baseline_stats.get('hps', {}).get('mean', 0.0), baseline_stats.get('hps', {}).get('std', 1.0)) if do_normalize else (0.0, 1.0)
            initialized_fns['hps'] = hps_loss_fn(
                device=accelerator.device, inference_dtype=weight_dtype,
                do_normalize=do_normalize, baseline_mean=mean, baseline_std=std
            )
        if "pickscore" in parsed_validation_rewards and pickscore_loss_fn is not None:
            mean, std = (baseline_stats.get('pickscore', {}).get('mean', 0.0), baseline_stats.get('pickscore', {}).get('std', 1.0)) if do_normalize else (0.0, 1.0)
            initialized_fns['pickscore'] = pickscore_loss_fn(
                device=accelerator.device, inference_dtype=weight_dtype,
                do_normalize=do_normalize, baseline_mean=mean, baseline_std=std
            )
        if "imagereward" in parsed_validation_rewards and imagereward_loss_fn is not None:
            mean, std = (baseline_stats.get('imagereward', {}).get('mean', 0.0), baseline_stats.get('imagereward', {}).get('std', 1.0)) if do_normalize else (0.0, 1.0)
            initialized_fns['imagereward'] = imagereward_loss_fn(
                device=accelerator.device, inference_dtype=weight_dtype,
                do_normalize=do_normalize, baseline_mean=mean, baseline_std=std
            )
        return initialized_fns

    # Initial initialization (unnormalized)
    validation_reward_fns_dict = initialize_reward_functions(do_normalize=False)

    if args.normalize_rewards and accelerator.is_main_process:
        logger.info(f"Estimating reward statistics using {args.reward_norm_batches} batches...")
        # Temporary dict to store sum and sum_sq for each reward type
        reward_sums = {name: 0.0 for name in parsed_validation_rewards if name in validation_reward_fns_dict}
        reward_sum_sqs = {name: 0.0 for name in parsed_validation_rewards if name in validation_reward_fns_dict}
        reward_counts = {name: 0 for name in parsed_validation_rewards if name in validation_reward_fns_dict}

        # Store original unet training state and set to eval
        original_unet_training_state = pipe.unet.training
        pipe.unet.eval()

        # --- Disable LoRA adapters for stat collection ---
        if adapter_list: # Check if adapters were loaded
            logger.info("Disabling LoRA adapters for reward stat estimation.")
            try:
                pipe.unet.disable_adapters()
            except Exception as e:
                logger.warning(f"Could not disable adapters on UNet, proceeding without explicit disabling: {e}")
        # --- End Disable LoRA adapters ---
        
        with torch.no_grad():
            for _ in tqdm(range(args.reward_norm_batches), desc="Estimating Reward Stats"):
                # Generate a batch of images
                # Use validation_prompt or a generic prompt for stat collection
                # For simplicity, using args.validation_prompt here.
                # Ensure prompt list matches batch size.
                current_prompts = [args.validation_prompt or "a photo"] * args.sample_batch_size

                # Sample images (similar to how it's done in validation but simplified)
                # We need a version of `sample` or direct pipeline call that returns tensor images
                # For simplicity, let's assume we adapt the sampling from the training loop directly.
                # This part might need careful adaptation based on `sample` function's true capability
                # or by directly using the pipeline.
                
                # Simplified image generation for stat collection:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                prompt_embeds = pipe._encode_prompt(
                    current_prompts,
                    device=accelerator.device,
                    num_images_per_prompt=1, # ensure 1 image per prompt for batch consistency
                    do_classifier_free_guidance=True, # Assuming CFG for consistency
                    negative_prompt=None, # Or provide a generic negative prompt
                    negative_prompt_embeds=None # Or provide generic negative embeddings
                )
                
                latents = torch.randn(
                    (args.sample_batch_size, pipe.unet.config.in_channels, args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor),
                    generator=generator,
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                pipe.scheduler.set_timesteps(args.num_inference_steps_val, device=accelerator.device)
                timesteps = pipe.scheduler.timesteps

                for t in pipe.progress_bar(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if True else latents # Assuming CFG
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                    if True: # CFG
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

                # Decode latents to images
                images_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1) # Normalize to [0,1]

                for reward_name, reward_fn in validation_reward_fns_dict.items():
                    if reward_name not in parsed_validation_rewards: # Should not happen if dict is built from parsed_rewards
                        continue
                    try:
                        # Reward functions are expected to return (loss_mean, rewards_tensor)
                        # We need the raw rewards_tensor for stat calculation.
                        if reward_name == 'aesthetic':
                            _, rewards_tensor = reward_fn(images_tensor) # Pass only images for aesthetic
                        else:
                            _, rewards_tensor = reward_fn(images_tensor, current_prompts) # Pass images and prompts for others
                        
                        if rewards_tensor is not None and rewards_tensor.numel() > 0:
                            reward_sums[reward_name] += torch.sum(rewards_tensor).item()
                            reward_sum_sqs[reward_name] += torch.sum(rewards_tensor**2).item()
                            reward_counts[reward_name] += rewards_tensor.numel()
                        else:
                            logger.warning(f"Reward tensor for {reward_name} was None or empty during stat collection.")
                    except Exception as e:
                        logger.error(f"Error calculating rewards for {reward_name} during stat collection: {e}")
                        # Potentially skip this reward or handle error
                        pass
        
        # Restore original unet training state
        pipe.unet.train(original_unet_training_state)

        # --- Re-enable LoRA adapters if they were disabled ---
        if adapter_list: # Check if adapters were loaded and potentially disabled
            logger.info("Re-enabling LoRA adapters after reward stat estimation.")
            try:
                pipe.unet.enable_adapters()
            except Exception as e:
                logger.warning(f"Could not re-enable adapters on UNet: {e}")
        # --- End Re-enable LoRA adapters ---

        # Calculate mean and std for each reward type
        for reward_name in reward_sums.keys():
            if reward_counts[reward_name] > 1: # Need at least 2 samples for std
                mean = reward_sums[reward_name] / reward_counts[reward_name]
                # Variance = E[X^2] - (E[X])^2
                variance = (reward_sum_sqs[reward_name] / reward_counts[reward_name]) - (mean**2)
                std = math.sqrt(variance) if variance > 0 else 0.0 # Avoid sqrt of negative due to precision
                reward_baseline_stats[reward_name] = {'mean': mean, 'std': std if std > 1e-6 else 1.0} # Avoid std too close to zero
                logger.info(f"Reward '{reward_name}': Mean={mean:.4f}, Std={std:.4f} (Count: {reward_counts[reward_name]})")
                # Log to accelerator trackers
                if accelerator.trackers:
                     accelerator.log({
                         f"reward_norm/{reward_name}_mean": mean,
                         f"reward_norm/{reward_name}_std": std,
                         f"reward_norm/{reward_name}_count": reward_counts[reward_name],
                     }, step=0) # Log at step 0 or a pre-training step
            else:
                logger.warning(f"Not enough samples to calculate valid stats for {reward_name} (Count: {reward_counts[reward_name]}). Using default (0,1).")
                reward_baseline_stats[reward_name] = {'mean': 0.0, 'std': 1.0}


        # Re-initialize reward functions with normalization enabled
        logger.info("Re-initializing reward functions with normalization.")
        validation_reward_fns_dict = initialize_reward_functions(do_normalize=True, baseline_stats=reward_baseline_stats)
    elif args.normalize_rewards and not accelerator.is_main_process:
        # For non-main processes, still ensure reward_fns_dict is updated if main process did normalization
        # This assumes the reward functions themselves don't need to be on multiple processes for init,
        # or that they are lightweight enough.
        # A more robust solution might involve broadcasting stats from main process.
        # For now, re-initialize with defaults (0,1) if not main, or wait for a broadcast mechanism if complex.
        # Simplified: just re-init, assuming the main process's log is what matters for user.
        logger.info("Non-main process: Assuming reward functions will be (re)initialized based on main process decisions if normalization is active.")
        # Potentially, this could lead to non-main processes having unnormalized functions if stats aren't broadcast.
        # However, validation usually runs on main process.
        validation_reward_fns_dict = initialize_reward_functions(do_normalize=True, baseline_stats=None) # Uses defaults if no stats


    # ---- End Reward Normalization Logic ----

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_sampled)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader_sampled)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    prompts = args.prompts
    
    for global_step in range(args.max_train_steps):

        train_dataset_sampled.sample_images(accelerator, pipe, args.num_inference_steps_train, args.guidance_scale, prompts, adapter_list, mu, args.train_batch_size, return_type = "PIL")
        true_samples = next(iterators[0])

        KLs = []
        logs = {}
        mode = args.constraints_kl_mode
        for adapter in adapter_list:
            KL_per_step, KL = compute_KL_weighted(args, accelerator, pipe, prompts, weight_dtype, adapter_list, adapter, mu, unwrap_model, true_samples)
            KLs.append(KL[mode])
            # add KL to logs
            for mode, value in KL.items():
                logs[f"KL/{mode}/{adapter}"] = value.item()
        print("mu: ", mu)
        print("KLs computed")
        print("KLs: ")
        print(KLs)
        # dual lr decay
        if global_step >= 0:
            args.lr_dual = args.lr_dual * 0.9
        mu = dual_step(args, KLs, mu, 0)
        # add mu to logs
        for i in range(len(mu)):
            logs[f"mu/{adapter_list[i]}"] = mu[i]
        print("mu updated")
        print("mu: ", mu)
        print("Epoch: ", global_step)

            
        if (global_step%(args.validation_epochs) == 0) or (global_step == args.max_train_steps - 1):
            print("Validation")
            #Do a Validation Pass
            # set the adapters to the mixture w the current mu
            pipe.set_adapters(adapter_list, adapter_weights=mu)
            n_c, pil_images, tensor_images, probs = validation(args,
                                            accelerator,
                                            pipe,
                                            unet,
                                            text_encoder,
                                            weight_dtype,
                                            unwrap_model,
                                            args.validation_prompt)
            #Add relevant info to logs
            class_names = args.classifier_class_names.split(', ')
            for j in range(len(class_names)):
                logs[class_names[j]] = n_c[j]          
                logs["validation"] = [wandb.Image(img, caption=f"{i}: " + class_names[torch.argmax(probs[i, :])]) for i, img in enumerate(pil_images)]
            # images = sample(accelerator, pipe, args.num_inference_steps_val, args.guidance_scale, [args.instance_prompt], mu, args.num_validation_images, return_type = "PIL")
            # logs["validation"] = [wandb.Image(image, caption=args.instance_prompt) for image in images]

            # Evaluate additional validation rewards if specified
            print("*"*100)
            print("Validation Finished, computing rewards")
            print("*"*100)
            print("parsed_validation_rewards: ", parsed_validation_rewards)
            print("pil_images: ", pil_images)
            if accelerator.is_main_process and parsed_validation_rewards and pil_images:
                logger.info("Evaluating additional validation rewards...")
                print(f"Validation Finished, computing rewards {parsed_validation_rewards}")

                try:
                    
                    if tensor_images is None or tensor_images.nelement() == 0:
                        logger.error("Tensor images are not available for validation reward calculation. Skipping.")
                    else:
                        # Prepare prompts (list of strings)
                        # The validation() function uses args.validation_prompt
                        validation_prompts_list = [args.validation_prompt] * len(pil_images) # Use len(pil_images) as reference for num_images

                        for reward_name in parsed_validation_rewards:
                            if reward_name in validation_reward_fns_dict:
                                current_reward_fn = validation_reward_fns_dict[reward_name]
                                reward_values = None
                                try:
                                    if reward_name == 'mps':
                                        # mps_loss_fn typically returns (loss_mean, rewards_tensor)
                                        _, reward_values = current_reward_fn(tensor_images, validation_prompts_list)
                                    elif reward_name == 'aesthetic':
                                        # aesthetic_loss_fn typically returns (loss_mean, rewards_tensor)
                                        _, reward_values = current_reward_fn(tensor_images) # Aesthetic might not need prompt
                                    # Add other reward cases here if implemented
                                    # elif reward_name == 'hps':
                                    #     _, reward_values = current_reward_fn(tensor_images, validation_prompts_list)
                                    elif reward_name == 'hps':
                                        _, reward_values = current_reward_fn(tensor_images, validation_prompts_list)
                                    elif reward_name == 'pickscore':
                                        _, reward_values = current_reward_fn(tensor_images, validation_prompts_list)
                                    elif reward_name == 'imagereward':
                                        _, reward_values = current_reward_fn(tensor_images, validation_prompts_list)
                                    else:
                                        logger.warning(f"Reward function for '{reward_name}' not fully implemented for validation score calculation.")
                                        continue
                                    
                                    if reward_values is not None:
                                        mean_reward = torch.mean(reward_values).item()
                                        accelerator.log({f"val_reward/{reward_name.capitalize()}": mean_reward}, step=global_step + 1)
                                        logger.info(f"  Validation Reward ({reward_name.capitalize()}): {mean_reward:.4f}")
                                    else:
                                        logger.warning(f"Could not obtain reward values for {reward_name} during validation.")

                                except Exception as e:
                                    logger.error(f"Error calculating validation reward for {reward_name}: {e}")
                            else:
                                logger.warning(f"Requested validation reward '{reward_name}' was not initialized or available. Skipping.")
                except Exception as e:
                    logger.error(f"Error during preparation for additional validation rewards: {e}")
    
             
                
        accelerator.log(logs, step = global_step+1)
        print("logs: ", logs)
              
             

    # Save the lora layers
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
