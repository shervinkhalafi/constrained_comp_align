import os
import random
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple
from warnings import warn
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
import ast


from accelerate.utils import ProjectConfiguration, set_seed
from transformers import is_wandb_available


from sd_pipeline import DiffusionPipeline
from config.alignprop_config import AlignPropConfig
from trl.trainer import BaseTrainer
from peft import get_peft_model_state_dict
import json
import numpy as np
if is_wandb_available():
    import wandb

from rewards import imagereward_loss_fn, aesthetic_loss_fn, hps_loss_fn, clip_x_loss_fn, pickscore_loss_fn, blip2_itm_loss_fn, mps_loss_fn, brightness_loss_fn, global_contrast_loss_fn, local_contrast_loss_fn, saturation_loss_fn, colorfulness_loss_fn
from dual_optimizers import NuPIController

logger = get_logger(__name__)


class AlignPropTrainer(BaseTrainer):
    """
    The AlignPropTrainer uses Deep Diffusion Policy Optimization to optimise diffusion models.
    Note, this trainer is heavily inspired by the work here: https://github.com/mihirp1998/AlignProp/
    As of now only Stable Diffusion based pipelines are supported

    Attributes:
        config (`AlignPropConfig`):
            Configuration object for AlignPropTrainer. Check the documentation of `PPOConfig` for more details.
        reward_function (`Callable[[torch.Tensor, Tuple[str], Tuple[Any]], torch.Tensor]`):
            Reward function to be used
        prompt_function (`Callable[[], Tuple[str, Any]]`):
            Function to generate prompts to guide model
        sd_pipeline (`DiffusionPipeline`):
            Stable Diffusion pipeline to be used for training.
        image_samples_hook (`Optional[Callable[[Any, Any, Any], Any]]`):
            Hook to be called to log images
    """

    _tag_names = ["trl", "alignprop"]

    def __init__(
        self,
        unet_copy,
        config: AlignPropConfig,
        train_prompt_function: Callable[[], Tuple[str, Any]],
        eval_prompt_function: Callable[[], Tuple[str, Any]],
        sd_pipeline: DiffusionPipeline,
        validation_prompt_function: Optional[Callable[[Any, Any, Any], Any]] = None,
        image_samples_hook: Optional[Callable[[Any, Any, Any], Any]] = None
    ):
        if image_samples_hook is None:
            warn("No image_samples_hook provided; no images will be logged")

        self.unet_copy = unet_copy

        self.train_prompt_fn = train_prompt_function
        self.val_prompt_fn = validation_prompt_function
        self.eval_prompt_fn = eval_prompt_function
        
        self.config = config
        self.image_samples_callback = image_samples_hook
        
        
        if config.constrained:
            list_of_tuples = ast.literal_eval(''.join(config.constraint_list))
        else:
            list_of_tuples = [(config.reward_fn, 0, 0), ("kl", 0, config.reg_coeff)]

        # add 0.25 to sharpness
        for i in range(len(list_of_tuples)):
            if list_of_tuples[i][0] == 'local_contrast_low':
                list_of_tuples[i] = ('local_contrast_low', list_of_tuples[i][1] + 0.25, list_of_tuples[i][2])
            
        
        #m is the number of constraints
        m = len(list_of_tuples) - 1

        self.dual_var = np.zeros(m)
            
        self.constraint_threshold = np.zeros(m)
        if self.config.constrained:
            # TODO (ihounie): Why do we need lr to be a tensor if it is a single scalar?
            self.lr_dual = config.dual_learning_rate*torch.ones(1, requires_grad = False)
        else:
            self.lr_dual = torch.zeros(1, requires_grad = False)
        
        if config.use_nupi:
            if self.config.dual_warmup_epochs > 0:
                kappa_p = config.nupi_kappa_p / self.config.dual_warmup_epochs
                kappa_i = config.nupi_kappa_i / self.config.dual_warmup_epochs
            else:
                kappa_p = config.nupi_kappa_p
                kappa_i = config.nupi_kappa_i
            self.nupi = NuPIController(theta_0 = self.dual_var, xi_0 = self.constraint_threshold,
                                        nu = config.nupi_nu, kappa_p = kappa_p, kappa_i = kappa_i)
        
        self.const_list = [list_of_tuples[0][0]]
        for i in range(m):
            self.constraint_threshold[i] = list_of_tuples[i + 1][1]
            self.const_list.append(list_of_tuples[i + 1][0])
            self.dual_var[i] = list_of_tuples[i + 1][2]
        
        # Initialize the dictionary to store loss functions by constraint name
        self.loss_functions = {}

        accelerator_project_config = ProjectConfiguration(**self.config.project_kwargs)

        if self.config.resume_from:
            self.config.resume_from = os.path.normpath(os.path.expanduser(self.config.resume_from))
            if not os.path.isfile(self.config.resume_from):
                #TODO (ihounie): I have broken this by messing with checkpointing.
                raise NotImplementedError
        self.accelerator = Accelerator(
            log_with=self.config.log_with,
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
            # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
            # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
            # the total number of optimizer steps to accumulate across.
            gradient_accumulation_steps=self.config.train_gradient_accumulation_steps,
            **self.config.accelerator_kwargs,
        )

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        config_dict = config.to_dict()
        config_dict['checkpoints_dir'] = self.config.project_kwargs['project_dir']

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=dict(alignprop_trainer_config=config_dict)
                if not is_using_tensorboard
                else config.to_dict(),
                init_kwargs=self.config.tracker_kwargs,
            )

        logger.info(f"\n{config}")

        set_seed(self.config.seed, device_specific=True)

        self.sd_pipeline = sd_pipeline

        self.sd_pipeline.set_progress_bar_config(
            position=1,
            disable=not self.accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        self.sd_pipeline.vae.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.text_encoder.to(self.accelerator.device, dtype=inference_dtype)
        self.sd_pipeline.unet.to(self.accelerator.device, dtype=inference_dtype)
        
        #newcode
        self.unet_copy.to(self.accelerator.device, dtype=inference_dtype)

        trainable_layers = self.sd_pipeline.get_trainable_layers()

        self.accelerator.register_save_state_pre_hook(self._save_model_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer = self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
        # more memory
        self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast

        if hasattr(self.sd_pipeline, "use_lora") and self.sd_pipeline.use_lora:
            unet, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
            self.trainable_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
        else:
            self.trainable_layers, self.optimizer = self.accelerator.prepare(trainable_layers, self.optimizer)
        
        # 1. Initialize unnormalized losses
        self._init_reward_functions(inference_dtype, do_normalize=False)

        # 2. Estimate baseline constraints with these unnormalized losses
        if self.config.normalize_constraints:
            self._estimate_scaling_constraints()
            # 3. Re-initialize the same constraints with do_normalize=True, 
            #    passing the newly collected baseline stats
            self._init_reward_functions(inference_dtype, do_normalize=True)
        
        if config.resume_from:
            logger.info(f"Resuming from {config.resume_from}")
            self.accelerator.load_state(config.resume_from)
            self.first_epoch = int(config.resume_from.split("_")[-1]) + 1
        else:
            self.first_epoch = 0

        # TODO (ihounie): fix batch generation to handle when num_steps is not a multiple of batch size
        self.num_steps_per_epoch = int(self.config.train_samples_per_epoch / (self.config.train_gradient_accumulation_steps*self.config.train_batch_size*self.accelerator.num_processes))
        self.num_validation_batches = int(self.config.num_validation_samples / (self.config.eval_batch_size*self.accelerator.num_processes))
        self.num_eval_batches = int((self.config.num_eval_samples / (self.config.eval_batch_size*self.accelerator.num_processes)))




    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.
        """
        info = defaultdict(list)
        print(f"Epoch: {epoch}, Global Step: {global_step}")

        self.sd_pipeline.unet.train()
        self.optimizer.zero_grad()
        for s in range(self.config.train_gradient_accumulation_steps):
            print(f"Step: {s}/{self.config.train_gradient_accumulation_steps}")
            with self.accelerator.accumulate(self.sd_pipeline.unet), self.autocast(), torch.enable_grad():
                prompt_image_pairs, kl_reg = self._generate_samples(
                    batch_size=self.config.train_batch_size,
                    prompt_fn = self.train_prompt_fn,
                    return_kl_div=True,
                    with_grad=True
                )
                
                lagrangian = torch.zeros(1).to(self.accelerator.device)
                loss_tensor = torch.zeros((len(self.const_list))).to(self.accelerator.device)
                losses = np.zeros((len(self.const_list), self.config.train_batch_size))
                all_rewards = {}
                
                for i in range(len(self.const_list)):
                    
                    if self.const_list[i] == 'kl':
                        l = kl_reg
                    elif self.const_list[i] == 'hps':
                        l, rewards = self.hps_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == 'aesthetic':
                        l, rewards = self.aesthetic_loss_fn(prompt_image_pairs["images"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "imagereward":
                        l, rewards = self.image_reward_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif "clip" in self.const_list[i]:
                        x = self.const_list[i].split("_", 1)[1] if "_" in self.const_list[i] else None
                        l, rewards = self.clip_x_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"], x=x)
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "pickscore":
                        l, rewards = self.pickscore_loss_fn(
                            prompt_image_pairs["images"],
                            prompt_image_pairs["prompts"]
                        )
                        all_rewards["pickscore"] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "blip_itm":
                        # The prompts argument will be ignored if config_prompt was set
                        l, rewards = self.blip_itm_loss_fn(
                            prompt_image_pairs["images"],
                            prompt_image_pairs["prompts"]  # this is ignored if config_prompt is used
                        )
                        all_rewards["blip_itm"] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "mps":
                        l, rewards = self.mps_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                        all_rewards["mps"] = rewards.detach().cpu().numpy()
                    elif "brightness" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["brightness"] = rewards.detach().cpu().numpy()
                    elif "global_contrast" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["global_contrast"] = rewards.detach().cpu().numpy()
                    elif "local_contrast" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["local_contrast"] = rewards.detach().cpu().numpy()
                    elif "saturation" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["saturation"] = rewards.detach().cpu().numpy()
                    elif "colorfulness" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["colorfulness"] = rewards.detach().cpu().numpy()
                    assert(l.requires_grad)
                    loss_tensor[i] = l.mean()
                    losses[i] = l.detach().cpu().numpy()

                if self.config.dual_update == 'alternate':
                    # Dual update step
                    if self.config.constrained == True and (global_step*self.config.train_gradient_accumulation_steps+s) % self.config.primals_per_dual == 0:
                        if self.config.use_nupi:
                            #-c-r<0
                            #-c<r
                            self.dual_var = self.nupi.update(losses[1:].mean(axis=1) - self.constraint_threshold)
                            if self.config.dual_warmup_epochs > 0 and epoch <= self.config.dual_warmup_epochs-1:
                                self.nupi.kappa_p = self.config.nupi_kappa_p * (epoch+1) / self.config.dual_warmup_epochs
                                self.nupi.kappa_i = self.config.nupi_kappa_i * (epoch+1) / self.config.dual_warmup_epochs
                        else:
                            for i in range(len(self.dual_var)):
                                self.dual_var[i] = torch.nn.functional.relu(self.dual_var[i] + self.lr_dual[i]*(losses[i + 1].mean() - self.constraint_threshold[i]))

                for i in range(len(self.const_list)):
                    l = loss_tensor[i]
                    if i == 0:
                        lagrangian += self.config.obj_coeff*l.mean()
                    else:
                        lagrangian += (self.dual_var[i - 1].item())*l.mean()
                
                self.accelerator.backward(lagrangian)
                if self.config.dual_update == 'simultaneous':
                    # Dual update step
                    if self.config.constrained == True and (global_step*self.config.train_gradient_accumulation_steps+s) % self.config.primals_per_dual == 0:
                        if self.config.use_nupi:
                            self.dual_var = self.nupi.update(losses[1:].mean(axis=1) - self.constraint_threshold)
                            if self.config.dual_warmup_epochs > 0 and epoch <= self.config.dual_warmup_epochs-1:
                                self.nupi.kappa_p = self.config.nupi_kappa_p * (epoch+1) / self.config.dual_warmup_epochs
                                self.nupi.kappa_i = self.config.nupi_kappa_i * (epoch+1) / self.config.dual_warmup_epochs
                        else:
                            for i in range(len(self.dual_var)):
                                self.dual_var[i] = torch.nn.functional.relu(self.dual_var[i] + self.lr_dual[i]*(losses[i + 1].mean() - self.constraint_threshold[i]))
                
                for i in range(len(self.const_list)):
                    if i == 0:
                        info["Objective Loss (" + str(self.const_list[i]) + ")"].extend(losses[i])
                    else:
                        info["Constraint Loss (" + str(self.const_list[i]) + ")"].extend(losses[i])
                        info["Constraint Slack (" + str(self.const_list[i]) + ")"].extend(losses[i] - self.constraint_threshold[i - 1])
                        # TODO (ihounie): We do not need to log the dual variable at every betch, it is enough to log it at the end of the grad acc steps
                        info["Dual Variable (" + str(self.const_list[i]) + ")"].append(self.dual_var[i - 1])
                    if self.const_list[i] in ['hps', 'aesthetic']:
                        info["Reward (" + str(self.const_list[i]) + ")"].extend(all_rewards[self.const_list[i]])
            
        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # log training-related stuff
            info = {"train/" + k: np.mean(v) for k, v in info.items()}
            info.update({"epoch": epoch})
            self.accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)
        else:
            raise ValueError(
                "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
        )
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.trainable_layers.parameters()
                if not isinstance(self.trainable_layers, list)
                else self.trainable_layers,
                self.config.train_max_grad_norm,
            )
            # Add the optimizer step here
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Logs generated images
        if self.image_samples_callback is not None and (global_step % self.config.log_image_freq == 0 or epoch == self.config.num_epochs-1) and self.accelerator.is_main_process:
            print("Logging images")
            # Fix the random seed for reproducibility
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            with torch.no_grad():
                prompt_image_pairs_eval = self._generate_samples(
                        batch_size=self.config.eval_len, with_grad=False, prompt_fn=self.eval_prompt_fn, return_kl_div=False
                    )

            self.image_samples_callback(prompt_image_pairs_eval, global_step, self.accelerator.trackers[0])
            seed = random.randint(0, 100)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed) 
                   
        # Determine if validation should be performed
        if self.val_prompt_fn and (global_step % self.config.val_freq == 0) and self.config.save_strategy == "best":
            metrics = self.validate(global_step, epoch)
            val_metric_value = metrics["val/"+self.config.target_val_metric]

            if (self.config.target_val_mode == "max" and val_metric_value > self.best_val_metric) or \
               (self.config.target_val_mode == "min" and val_metric_value < self.best_val_metric):
                self.best_val_metric = val_metric_value
                print("Saving best LoRA checkpoint")
                self.save_training_state("best_lora_adapters", "best")
        
        if epoch % self.config.test_freq == 0 or epoch == self.config.num_epochs-1:
                self.test(global_step, epoch)

        # If no val_freq is provided, fall back to save_freq for checkpointing
        elif self.val_prompt_fn == None and (epoch % self.config.save_freq == 0) or epoch == self.config.num_epochs-1:
            print("Saving latest LoRA checkpoint")
            self.save_training_state("latest_lora_adapters", "latest")
        print("Step Done")
        return global_step

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.config.train_learning_rate,
            betas=(self.config.train_adam_beta1, self.config.train_adam_beta2),
            weight_decay=self.config.train_adam_weight_decay,
            eps=self.config.train_adam_epsilon,
        )

    def _save_model_hook(self, models, weights, output_dir):
        self.sd_pipeline.save_checkpoint(models, weights, output_dir)
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def _load_model_hook(self, models, input_dir):
        self.sd_pipeline.load_checkpoint(models, input_dir)
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    def _generate_samples(self, batch_size, with_grad=True, prompt_fn=None, return_kl_div=False):
        """
        Generate samples from the model

        Args:
            batch_size (int): Batch size to use for sampling
            with_grad (bool): Whether the generated RGBs should have gradients attached to it.
            prompt_fn: Function that returns prompts to guide the image generation.

        Returns:
            prompt_image_pairs (Dict[Any])
        """
        prompt_image_pairs = {}

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        prompts, prompt_metadata = zip(*[prompt_fn() for _ in range(batch_size)])

        prompt_ids = self.sd_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)

        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

        if with_grad:
            sd_output = self.sd_pipeline.rgb_with_grad(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                backprop_strategy=self.config.backprop_strategy,
                backprop_kwargs=self.config.backprop_kwargs[self.config.backprop_strategy],
                output_type = "pt",
                return_kl_div=return_kl_div,  
                unet_copy = self.unet_copy,             
            )
        else:
            sd_output = self.sd_pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                output_type="pt",
                return_kl_div=return_kl_div,  
                unet_copy = self.unet_copy,       
            )
        if return_kl_div:
            sd_output, kl_regularizer = sd_output

        images = sd_output.images

        prompt_image_pairs["images"] = images
        prompt_image_pairs["prompts"] = prompts
        prompt_image_pairs["prompt_metadata"] = prompt_metadata

        if return_kl_div:
            return prompt_image_pairs, kl_regularizer
        else:
            return prompt_image_pairs

    def evaluate(self, prefix: str, prompt_fn, num_batches: int, step: int, epoch: int):
        """
        Perform evaluation using the given prompts and log metrics with the specified prefix.

        Args:
            prefix (str): The prefix for logging metrics (e.g., "val" or "eval").
            
        """
        # Initialize accumulators for metrics
        info = defaultdict(list)
        self.sd_pipeline.unet.eval()
        for batch_idx in range(num_batches):
            # Generate samples using the prompt function
            with torch.no_grad():
                prompt_image_pairs, kl_reg = self._generate_samples(
                    batch_size=self.config.eval_batch_size,  
                    prompt_fn=prompt_fn,
                    return_kl_div=True,
                    with_grad=False
                )

                lagrangian = torch.zeros(1).to(self.accelerator.device)
                losses = np.zeros((len(self.const_list), self.config.eval_batch_size))
                all_rewards = {}
                
                for i in range(len(self.const_list)):
                    
                    if self.const_list[i] == 'kl':
                        l = kl_reg
                        
                    elif self.const_list[i] == 'hps':
                        l, rewards = self.hps_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                        
                    elif self.const_list[i] == 'aesthetic':
                        l, rewards = self.aesthetic_loss_fn(prompt_image_pairs["images"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "imagereward":
                        l, rewards = self.image_reward_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif "clip" in self.const_list[i]:
                        x = self.const_list[i].split("_", 1)[1] if "_" in self.const_list[i] else None
                        l, rewards = self.clip_x_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"], x=x)
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "pickscore":
                        l, rewards = self.pickscore_loss_fn(
                            prompt_image_pairs["images"],
                            prompt_image_pairs["prompts"]
                        )
                        all_rewards[self.const_list[i] ] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "blip_itm":
                        # The prompts argument will be ignored if config_prompt was set
                        l, rewards = self.blip_itm_loss_fn(
                            prompt_image_pairs["images"],
                            prompt_image_pairs["prompts"]  # this is ignored if config_prompt is used
                        )
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif self.const_list[i] == "mps":
                        l, rewards = self.mps_loss_fn(prompt_image_pairs["images"], prompt_image_pairs["prompts"])
                        all_rewards[self.const_list[i]] = rewards.detach().cpu().numpy()
                    elif "brightness" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["brightness"] = rewards.detach().cpu().numpy()
                    elif "global_contrast" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["global_contrast"] = rewards.detach().cpu().numpy()
                    elif "local_contrast" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["local_contrast"] = rewards.detach().cpu().numpy()
                    elif "saturation" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["saturation"] = rewards.detach().cpu().numpy()
                    elif "colorfulness" in self.const_list[i]:
                        # Use the exact constraint name as the key to access the function
                        l, rewards = self.loss_functions[self.const_list[i]](prompt_image_pairs["images"])
                        all_rewards["colorfulness"] = rewards.detach().cpu().numpy()
                    
                    if i == 0:
                        lagrangian += self.config.obj_coeff*l.mean()
                    else:
                        lagrangian += (self.dual_var[i - 1])*l.mean()
                        
                    losses[i] = l.detach().cpu().numpy()

                for i in range(len(self.const_list)):
                    if i == 0:
                        info["Objective Loss (" + str(self.const_list[i]) + ")"].extend(losses[i])
                    else:
                        info["Constraint Loss (" + str(self.const_list[i]) + ")"].extend(losses[i])
                        info["Constraint Slack (" + str(self.const_list[i]) + ")"].extend(losses[i]- self.constraint_threshold[i - 1].item())
                    if self.const_list[i] in ['hps', 'aesthetic']:
                        info["Reward (" + str(self.const_list[i]) + ")"].extend(all_rewards[self.const_list[i]])


        # After all batches, compute mean and std of accumulated metrics
        metrics = {f"{prefix}/" + k: np.mean(v) for k, v in info.items()}
        metrics.update({"epoch": epoch})
           
        # Log metrics
        self.accelerator.log(metrics, step=step)

        # Return the mean reward for validation purposes
        return metrics

    def validate(self, step:int, epoch: int):
        """
        Perform validation using the validation prompts.
        """
        return self.evaluate("val", self.val_prompt_fn, self.num_validation_batches, step, epoch)

    def test(self, step: int, epoch: int):
        """
        Perform evaluation using the eval prompts.
        """
        return self.evaluate("test", self.eval_prompt_fn,  self.num_eval_batches, step, epoch)

    def train(self, epochs: Optional[int] = None):
        """
        Train the model for a given number of epochs
        """
        global_step = 0
        self.config.num_epochs+=1
        if epochs is None:
            epochs = self.config.num_epochs
        #self.test(global_step, self.first_epoch)
        global_step +=1
        self.first_epoch += 1
        for epoch in range(self.first_epoch, epochs):
            global_step = self.step(epoch, global_step)
        try:
            self.save_training_state("latest_lora_adapters", "latest")
        except:
            print("Could not upload latest model to wandb")
        try:
            self.upload_to_wandb("latest_lora_adapters", "best")
        except:
            print("Could not upload best model to wandb")


    def _save_pretrained(self, save_directory):
        # TODO(ihounie): This method is not used anywhere.
        # And is not implemented neither in the DiffusionPipeline in sd_pipeline.py 
        # nor in the DDPOStableDiffusionPipeline parent class fro trl (https://github.com/huggingface/trl/blob/main/trl/models/modeling_sd_base.py)
        self.sd_pipeline.save_pretrained(save_directory)
        self.create_model_card()
    
    def upload_to_wandb(self, directory: str, description: str):
        """
        Upload the training state to Weights & Biases.
        """
        artifact = wandb.Artifact(name=f"training_state_{description}", type="model")
        for file in os.listdir(directory):
            artifact.add_file(os.path.join(directory, file))
        wandb.log_artifact(artifact)
    
    def save_training_state(self, filename: str, description: str = ""):
        """
        Save the LoRA adapter weights, configuration, and optimizer state.
        """
        # Create a directory for saving
        os.makedirs(filename, exist_ok=True)

        # Save LoRA adapter weights
        lora_state_dict = get_peft_model_state_dict(self.sd_pipeline.unet)
        # append unet prefix
        lora_state_dict = {f"unet.{k}": v for k, v in lora_state_dict.items()}
        torch.save(lora_state_dict, os.path.join(filename, "lora_adapters.pth"))

        # Save configuration
        config_path = os.path.join(filename, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f)

        # TODO (ihounie) : do we need this?
        # Save optimizer state
        #optimizer_path = os.path.join(filename, "optimizer.pth")
        #torch.save(self.optimizer.state_dict(), optimizer_path)
    def _estimate_scaling_constraints(self):
        """
        Computes reward mean/std across multiple batches for each constraint,
        so we can normalize the *reward* (instead of the loss).
        """

        if not self.config.normalize_constraints: # if normalize_constraints is False, skip this.
            logger.info("Skipping baseline reward stats estimation as normalize_constraints is False.")
            return

        m = len(self.const_list) - 1
        if m <= 0:
            # No constraints to normalize other than potentially the objective, which is not handled here.
            return

        self.baseline_means = np.zeros(m, dtype=np.float64)
        self.baseline_stds = np.ones(m, dtype=np.float64) # Default std to 1.0

        if self.config.use_cached_scale:
            logger.info("Using cached reward scales.")
            try:
                # Path to the JSON file, relative to the workspace root
                json_path = "implementations/AlignProp/assets/reward_scaling.json"
                # Attempt to read the file using the read_file tool to ensure it's accessible
                # This is a placeholder to indicate we'd ideally verify file existence/contents
                # For now, we'll proceed with direct loading.
                
                # Correctly construct the absolute path if needed or ensure relative path is correct
                # For this example, assuming direct access within the trainer's execution environment
                # The user provided @reward_scaling.json, implying it's accessible.
                # We'll use a fixed relative path as specified.
                
                # In a real scenario, you might need to make this path configurable or more robust
                # For now, hardcoding based on the provided path.
                
                # Construct the full path relative to the script or a known base directory if necessary.
                # Here, assuming 'implementations/AlignProp/assets/reward_scaling.json' is accessible
                # from where the script runs, or it's an absolute path if specified differently.
                
                # The user mentioned @reward_scaling.json, which suggests a specific location.
                # Let's assume it's 'implementations/AlignProp/assets/reward_scaling.json'
                # based on the previous file interaction.
                
                # Correct approach for file path:
                # It should be relative to the workspace root or an absolute path.
                # The user attached implementations/AlignProp/assets/reward_scaling.json
                
                # Determine the absolute path to the workspace
                ASSETS_PATH = Path(__file__).parent
                full_json_path = os.path.join(workspace_path, json_path)


                if not os.path.exists(full_json_path):
                    logger.error(f"Cached reward scaling file not found at {full_json_path}. Falling back to online estimation.")
                    # Fallback to online estimation or handle error
                    # For now, let's try to proceed with online estimation if file not found
                else:
                    with open(full_json_path, 'r') as f:
                        cached_scales = json.load(f)

                    loaded_count = 0
                    for i in range(m):
                        constraint_name = self.const_list[i + 1]
                        # Normalize key for comparison (e.g. "image reward" vs "imagereward")
                        normalized_constraint_name = constraint_name.replace(" ", "").lower()
                        
                        found_match = False
                        for key_json, data_json in cached_scales.items():
                            normalized_key_json = key_json.replace(" ", "").lower()
                            if normalized_key_json == normalized_constraint_name:
                                if "final" in data_json:
                                    self.baseline_means[i] = data_json["final"]
                                    self.baseline_stds[i] = 1.0  # As per requirement
                                    logger.info(f"Loaded cached scale for {constraint_name}: mean={self.baseline_means[i]}, std=1.0")
                                    loaded_count += 1
                                    found_match = True
                                    break
                                else:
                                    logger.warning(f"'final' key not found for {key_json} in cached scales. Skipping {constraint_name}.")
                                    found_match = True # Mark as found to avoid falling back for this specific constraint
                                    break
                        if not found_match:
                            logger.warning(f"No cached scale found for constraint: {constraint_name}. It will use default mean=0, std=1 unless online estimation runs.")
                    
                    if loaded_count > 0 and loaded_count == m: # if all constraints were loaded
                        logger.info("Successfully loaded all cached reward scales.")
                        # Log the loaded baseline means and stds
                        for i in range(m):
                            self.accelerator.log(
                                {
                                    f"reward_mean_{self.const_list[i + 1]}": self.baseline_means[i],
                                    f"reward_std_{self.const_list[i + 1]}":  self.baseline_stds[i],
                                },
                                step=0 # Log at step 0 or an appropriate global step
                            )
                        return # Skip online estimation if all scales are loaded

                    elif loaded_count > 0 and loaded_count < m:
                        logger.warning("Partially loaded cached scales. Constraints without cached values will use defaults or require online estimation if not all were covered and fallback is enabled.")
                        # Decide if to proceed with partial online estimation or use defaults.
                        # For now, if any are missing, we might still want to run online for the rest, or error out.
                        # The current logic will proceed to online estimation if this block doesn't `return`.
                        # This means if not all are loaded, it will re-estimate ALL.
                        # This might need refinement based on desired behavior for partial matches.
                        logger.info("Proceeding to online estimation for constraints not covered by cache or if errors occurred.")


            except FileNotFoundError:
                logger.error(f"Cached reward scaling file not found at {json_path}. Proceeding with online estimation.")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {json_path}. Proceeding with online estimation.")
            except Exception as e:
                logger.error(f"An error occurred while loading cached scales: {e}. Proceeding with online estimation.")
        
        # Proceed with online estimation if not using cached scales, or if loading failed/was partial and we decide to re-estimate all.
        if getattr(self.config, "constraint_setting_batches", 0) <= 0:
            logger.info("constraint_setting_batches is 0 or less, skipping online baseline estimation.")
            # If we reached here and didn't load from cache, means and stds remain 0s and 1s.
            return


        # We'll collect sums of rewards (sum_of_x, sum_of_x2) locally,
        # then reduce across GPUs if necessary.
        sum_of_x_local = np.zeros(m, dtype=np.float64)
        sum_of_x2_local = np.zeros(m, dtype=np.float64)
        sum_count_local = np.zeros(m, dtype=np.float64)


        self.sd_pipeline.unet.eval()
        with torch.no_grad(), self.autocast():
            for _ in range(self.config.constraint_setting_batches):
                sample_batch = self._generate_samples(
                    batch_size=self.config.train_batch_size,
                    prompt_fn=self.train_prompt_fn,
                    return_kl_div=False,
                    with_grad=False
                )

                # We'll gather the *reward* for each constraint, not the loss
                for i in range(m):
                    constraint_name = self.const_list[i + 1]

                    if constraint_name == "kl":
                        continue
                    elif constraint_name == "hps":
                        _, rewards = self.hps_loss_fn(sample_batch["images"], sample_batch["prompts"])
                    elif constraint_name == "aesthetic":
                        _, rewards = self.aesthetic_loss_fn(sample_batch["images"])
                    elif constraint_name == "imagereward":
                        _, rewards = self.image_reward_loss_fn(sample_batch["images"], sample_batch["prompts"])
                    elif constraint_name == "pickscore":
                        _, rewards = self.pickscore_loss_fn(sample_batch["images"], sample_batch["prompts"])
                    elif "clip" in constraint_name:
                        x = constraint_name.split("_", 1)[1] if "_" in constraint_name else None
                        _, rewards = self.clip_x_loss_fn(sample_batch["images"], sample_batch["prompts"], x=x)
                    elif constraint_name == "blip_itm":
                        _, rewards = self.blip_itm_loss_fn(sample_batch["images"], sample_batch["prompts"])
                    elif constraint_name == "mps":
                        _, rewards = self.mps_loss_fn(sample_batch["images"], sample_batch["prompts"])
                    elif "brightness" in constraint_name:
                        # Use the exact constraint name as the key to access the function
                        _, rewards = self.loss_functions[constraint_name](sample_batch["images"])
                    elif "global_contrast" in constraint_name:
                        # Use the exact constraint name as the key to access the function
                        _, rewards = self.loss_functions[constraint_name](sample_batch["images"])
                    elif "local_contrast" in constraint_name:
                        # Use the exact constraint name as the key to access the function
                        _, rewards = self.loss_functions[constraint_name](sample_batch["images"])
                    elif "saturation" in constraint_name:
                        # Use the exact constraint name as the key to access the function
                        _, rewards = self.loss_functions[constraint_name](sample_batch["images"])
                    elif "colorfulness" in constraint_name:
                        # Use the exact constraint name as the key to access the function
                        _, rewards = self.loss_functions[constraint_name](sample_batch["images"])
                    else:
                        continue

                    rewards_np = rewards.detach().cpu().numpy()
                    sum_of_x_local[i]  += rewards_np.sum()
                    sum_of_x2_local[i] += (rewards_np**2).sum()
                    sum_count_local[i] += rewards_np.shape[0]

        # If multi-GPU, reduce or broadcast here. For simplicity, assume single GPU:
        # sum_of_x_local, sum_of_x2_local, sum_count_local are already global in single-GPU. 
        # For multi-GPU, do a self.accelerator.reduce(...) or broadcast. 
        # Summarized here for single-GPU usage:

        for i in range(m):
            if sum_count_local[i] == 0:
                continue
            mean_i = sum_of_x_local[i] / sum_count_local[i]
            e_x2   = sum_of_x2_local[i] / sum_count_local[i]
            var_i  = max(e_x2 - mean_i**2, 0.0)
            std_i  = np.sqrt(var_i)

            self.baseline_means[i] = mean_i
            self.baseline_stds[i]  = std_i

            self.accelerator.log(
                {
                    f"reward_mean_{self.const_list[i + 1]}": mean_i,
                    f"reward_std_{self.const_list[i + 1]}":  std_i,
                },
                step=0
            )
            logger.info(f"[Global Stats for {self.const_list[i + 1]}] => reward mean={mean_i:.4f}, std={std_i:.4f}")

        logger.info("Finished collecting baseline reward stats (means/stds).")

    def _init_reward_functions(self, inference_dtype, do_normalize=False):
        """
        Initializes each reward/loss function in self.const_list.
        If do_normalize=False, we pass baseline_mean=0, baseline_std=1, 
        so the reward is unnormalized for baseline gathering.
        If do_normalize=True, we use self.baseline_means[i], self.baseline_stds[i] 
        to properly normalize each constraint's reward/loss.
        """
        for i, name in enumerate(self.const_list):
            if i == 0:
                continue  # skip objective index 0

            # index in baseline arrays = i-1, since self.const_list[0] = objective
            arr_idx = i - 1

            # fallback to (0,1) if we haven't set them or if sum_count was zero
            mean_i = 0.0
            std_i = 1.0
            if (do_normalize 
                and hasattr(self, "baseline_means") 
                and hasattr(self, "baseline_stds") 
                and len(self.baseline_means) > arr_idx
            ):
                mean_i = self.baseline_means[arr_idx]
                std_i = self.baseline_stds[arr_idx]

            if name == "hps":
                self.hps_loss_fn = hps_loss_fn(
                    inference_dtype=inference_dtype,
                    device=self.accelerator.device,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif name == "aesthetic":
                self.aesthetic_loss_fn = aesthetic_loss_fn(
                    aesthetic_target=self.config.aesthetic_target,
                    grad_scale=self.config.grad_scale,
                    device=self.accelerator.device,
                    accelerator=self.accelerator,
                    torch_dtype=inference_dtype,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif name == "imagereward":
                self.image_reward_loss_fn = imagereward_loss_fn(
                    device=self.accelerator.device,
                    inference_dtype=inference_dtype,
                    grad_scale=self.config.grad_scale,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif "clip" in name:
                self.clip_x_loss_fn = clip_x_loss_fn(
                    device=self.accelerator.device,
                    clip_model_name="openai/clip-vit-large-patch14",
                    grad_scale=self.config.grad_scale,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif name == "pickscore":
                self.pickscore_loss_fn = pickscore_loss_fn(
                    device=self.accelerator.device,
                    pickscorer_id="yuvalkirstain/PickScore_v1",
                    inference_dtype=inference_dtype,
                    grad_scale=self.config.grad_scale,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif name == "blip_itm":
                self.blip_itm_loss_fn = blip2_itm_loss_fn(
                    device=self.accelerator.device,
                    model_name="Salesforce/blip2-itm-vit-g",
                    grad_scale=self.config.grad_scale,
                    config_prompt=self.config.blip_prompt,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif name == "mps":
                self.mps_loss_fn = mps_loss_fn(
                    device=self.accelerator.device,
                    mps_id="RE-N-Y/mpsv1",
                    inference_dtype=inference_dtype,
                    grad_scale=self.config.grad_scale,
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif "brightness" in name:
                # Use the exact constraint name as the key to access the function
                self.loss_functions[name] = brightness_loss_fn(
                    device=self.accelerator.device,
                    grad_scale=self.config.grad_scale,
                    direction="high" if name.endswith("_high") else "low" if name.endswith("_low") else "high",
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif "global_contrast" in name:
                # Use the exact constraint name as the key to access the function
                self.loss_functions[name] = global_contrast_loss_fn(
                    device=self.accelerator.device,
                    grad_scale=self.config.grad_scale,
                    direction="high" if name.endswith("_high") else "low" if name.endswith("_low") else "high",
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif "local_contrast" in name:
                # Use the exact constraint name as the key to access the function
                self.loss_functions[name] = local_contrast_loss_fn(
                    device=self.accelerator.device,
                    kernel_size=getattr(self.config, "contrast_kernel_size", 5),
                    sigma=getattr(self.config, "contrast_sigma", 1.0),
                    grad_scale=self.config.grad_scale,
                    direction="high" if name.endswith("_high") else "low" if name.endswith("_low") else "high",
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif "saturation" in name:
                # Use the exact constraint name as the key to access the function
                self.loss_functions[name] = saturation_loss_fn(
                    device=self.accelerator.device,
                    grad_scale=self.config.grad_scale,
                    direction="high" if name.endswith("_high") else "low" if name.endswith("_low") else "high",
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
            elif "colorfulness" in name:
                # Use the exact constraint name as the key to access the function
                self.loss_functions[name] = colorfulness_loss_fn(
                    device=self.accelerator.device,
                    grad_scale=self.config.grad_scale,
                    direction="high" if name.endswith("_high") else "low" if name.endswith("_low") else "high",
                    do_normalize=do_normalize,
                    baseline_mean=mean_i,
                    baseline_std=std_i
                )
