import ipdb
st = ipdb.set_trace
import builtins
import time
import os
builtins.st = ipdb.set_trace
from dataclasses import dataclass, field
import prompts as prompts_file
import numpy as np
from transformers import HfArgumentParser

from config.alignprop_config import AlignPropConfig
from alignprop_trainer import AlignPropTrainer
from sd_pipeline import DiffusionPipeline


from diffusers import UNet2DConditionModel



@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})




def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts = [image_pair_data["images"], image_pair_data["prompts"]]
    for i, image in enumerate(images[:4]):
        prompt = prompts[i]
        result[f"{prompt}"] = image.unsqueeze(0).float()
    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":


    
    parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # Update backprop_kwargs gaussian mean to be 80% of sample_num_steps
    training_args.backprop_kwargs['gaussian']['mean'] = int(0.8 * training_args.sample_num_steps)
    
    os.makedirs(f"checkpoints/{training_args.project_dir}", exist_ok=True)
    
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"checkpoints/{training_args.project_dir}",
    }

    train_prompt_fn = getattr(prompts_file, training_args.train_prompt_fn)#(val_frac=training_args.val_frac)
    val_prompt_fn = None
    eval_prompt_fn = getattr(prompts_file, training_args.eval_prompt_fn)
    
    pipeline = DiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=True,
    )
    # pretrain model to calculate kl
    unet_copy = UNet2DConditionModel.from_pretrained(
        script_args.pretrained_model,
        subfolder = "unet",
        revision = script_args.pretrained_revision,
    )
    # freeze unet copy
    unet_copy.requires_grad_(False)


    
    trainer = AlignPropTrainer(
        unet_copy,
        training_args,
        train_prompt_fn,
        eval_prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
        validation_prompt_function = val_prompt_fn,
    )

    
    trainer.train()
    # save lora adapters
    trainer.save_training_state(f"checkpoints/{training_args.project_dir}")

# Example usage of the new image attribute rewards:
#
# To use the new image attribute rewards, add them to your constraint list with the direction included in the name:
#
# 1. Brightness: Measures the mean luminance of the image
#    Examples: 
#    - ("brightness_high", 0, weight) to maximize brightness
#    - ("brightness_low", 0, weight) to minimize brightness
#
# 2. Global Contrast: Measures the standard deviation of luminance
#    Examples: 
#    - ("global_contrast_high", 0, weight) to maximize contrast
#    - ("global_contrast_low", 0, weight) to minimize contrast
#
# 3. Local Contrast: Measures unsharp mask energy (difference from blurred version)
#    Examples: 
#    - ("local_contrast_high", 0, weight) to maximize local contrast
#    - ("local_contrast_low", 0, weight) to minimize local contrast
#    Optional parameters: contrast_kernel_size (default 5), contrast_sigma (default 1.0)
#
# 4. Saturation: Measures color saturation in the HSV sense
#    Examples: 
#    - ("saturation_high", 0, weight) to maximize saturation
#    - ("saturation_low", 0, weight) to minimize saturation
#
# 5. Colorfulness: Measures the Hasler-Süsstrunk colorfulness metric
#    Examples: 
#    - ("colorfulness_high", 0, weight) to maximize colorfulness
#    - ("colorfulness_low", 0, weight) to minimize colorfulness
#
# If direction is not specified (e.g., just "brightness"), "high" is used by default.
# The direction should be specified directly in the constraint name by adding "_high" or "_low" suffix.
# 
# Example constraint list:
# [
#   ("aesthetic", 0, 1.0),                # Objective: maximize aesthetic quality
#   ("brightness_high", 0.7, 0.5),        # Constraint: brightness should be at least 0.7
#   ("saturation_low", 0.3, 0.3),         # Constraint: saturation should be at most 0.3
#   ("colorfulness_high", 0.6, 0.4)       # Constraint: colorfulness should be at least 0.6
# ]