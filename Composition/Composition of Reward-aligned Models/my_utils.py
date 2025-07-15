import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
from scheduling_ddim_extended import DDIMSchedulerExtended
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image
from tqdm.auto import tqdm
import shutil
from collections import defaultdict
from torchvision import transforms

def primal_step_new(args,
                batch,
                pipe,
                mu,
                adapters,
                weight_dtype,
                accelerator,
                encode_prompt,
                unwrap_model,
                text_encoder,
                progress_bar,
                noise_scheduler,
                global_step,
                optimizer,
                lr_scheduler,
                params_to_optimize,
                logger,
                ):
    
    with accelerator.accumulate(pipe.unet):

        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
        
        model_input = pipe.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * pipe.vae.config.scaling_factor
        
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        
        timesteps = torch.randint(args.min_timestep, args.max_timestep, (bsz,), device=model_input.device)

        timesteps = timesteps.long()
        
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        if args.pre_compute_text_embeddings:
            encoder_hidden_states = batch["input_ids"]
        else:
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )
        
        if unwrap_model(pipe.unet).config.in_channels == channels * 2:
            noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

        if args.class_labels_conditioning == "timesteps":
            class_labels = timesteps
        else:
            class_labels = None

        
        pipe.set_adapters("product")
        pipe.unet.train()
        # Predict the noise residual
        model_pred = pipe.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states,
            class_labels=class_labels,
            return_dict=False,
        )[0]
        
        
        
        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
                    # Predict noise residuals
        noise_preds = []
        for i in range(len(adapters)):
            # set adapter
            pipe.set_adapters(adapters[i])
            pipe.unet.eval()
            adapter_pred = pipe.unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                class_labels=class_labels,
                return_dict=False,
            )[0]
            if adapter_pred.shape[1] == 6:
                adapter_pred, _ = torch.chunk(adapter_pred, 2, dim=1)
            noise_preds.append(adapter_pred)
        final_noise_pred = torch.zeros_like(noise_preds[0])
        for i in range(len(adapters)):
            final_noise_pred += mu[i].to(noise_preds[i].dtype) * noise_preds[i]
            

              # Cast weight to match noise pred dtype
        
        loss = F.mse_loss(model_pred.float(), final_noise_pred.float(), reduction="mean")
        
   
        accelerator.backward(loss)
       
            
        
        if accelerator.sync_gradients:
            
            #log stuff
            logs = {}
    
            logs["loss"] = loss.detach().item()
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            
            
            progress_bar.set_postfix(**logs)

            #end of log
            
            accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
        
        
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
    
            

    # Checks if the accelerator has performed an optimization step behind the scenes
    if accelerator.sync_gradients:
        progress_bar.update(1)

        if accelerator.is_main_process:
            if global_step % args.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #accelerator.save_state(save_path)
                #logger.info(f"Saved state to {save_path}")
        
        return logs, loss


                        
def dual_step(args, KLs, mu, b):

    lr_dual = args.lr_dual
    l = torch.tensor(KLs, requires_grad = False)
    mu = torch.tensor(mu, requires_grad = False)
    b = 0
    grad = lr_dual*(l - b)
    
    if torch.linalg.norm(grad).detach() != 0:   
        # GD step
        mu = (mu + grad).detach()
        # make sure mu is in the simplex
        mu = project_onto_simplex(mu)

    return mu.tolist()
            
def validation(args,
              accelerator,
              pipe,
              unet,
              text_encoder,
              weight_dtype,
              unwrap_model,
              validation_prompts,
              ):
    
    class_names = args.classifier_class_names.split(', ')
    n_c = torch.zeros(len(class_names))
    for i in range(10000):

        n = torch.zeros(len(class_names), device = accelerator.device)

        validation_prompts = args.validation_prompt.split(", ")
        # create pipeline
        
        for k in range(len(validation_prompts)):
            pipe.unet.eval()
                

            pil_images = log_validation_extended(
                pipe,
                args,
                accelerator,
                {"prompt": validation_prompts[k]},
                weight_dtype,
                output_type='pil',
            )

            
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


            inputs = clip_processor(text=class_names, images=pil_images, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

            for j in range(probs.size()[0]):
                n[torch.argmax(probs[j, :])] += 1
                
        n_c += accelerator.reduce(n, reduction = 'sum').to('cpu')

        if torch.sum(n_c) >= args.num_validation_images:
            # Convert PIL images to tensors for reward calculation
            # Ensure this transform matches the one used/expected by reward models
            image_transforms_for_reward = transforms.Compose(
                [
                    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(args.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            try:
                # Assuming pil_images is a list of PIL Images from the last validation prompt batch
                tensor_images = torch.stack([image_transforms_for_reward(img) for img in pil_images]).to(accelerator.device, dtype=weight_dtype)
            except Exception as e:
                # Log error and return None for tensor_images if conversion fails
                # This might happen if pil_images is empty or contains non-image data
                print(f"Error converting PIL images to tensors in validation: {e}")
                tensor_images = None # Or an empty tensor: torch.tensor([]).to(accelerator.device, dtype=weight_dtype)

            return n_c, pil_images, tensor_images, probs

def project_onto_simplex(x: torch.Tensor) -> torch.Tensor:
    """
    Projects a 1D tensor x onto the simplex:
       { u : u[i] >= 0, sum_i u[i] = 1 }.
    """
    # Sort x in descending order
    sorted_x, sorted_indices = torch.sort(x, descending=True)

    # Accumulate the sorted values
    cssv = torch.cumsum(sorted_x, dim=0)

    # Identify the smallest k such that sorted_x[k] - (cssv[k] - 1) / (k+1) > 0
    # (Note: we shift by one because of 0-based indexing)
    rhos = sorted_x - (1.0 / torch.arange(1, x.numel() + 1, device=x.device)) * (cssv - 1)
    rho = torch.nonzero(rhos > 0, as_tuple=True)[0][-1]

    # The threshold to subtract
    theta = (cssv[rho] - 1.0) / (rho + 1.0)

    # Re-project and clamp to zero
    return torch.clamp(x - theta, min=0)

def log_validation_extended(
    pipe,
    args,
    accelerator,
    pipeline_args,
    torch_dtype,
    output_type='pil',
    ):
    
    
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    

    if args.validation_images is None:
        images = []
        num_validation_batches = args.num_validation_images // args.num_images_per_prompt
        if num_validation_batches == 0:
            num_validation_batches = 1
        for _ in range(num_validation_batches):
            with torch.cuda.amp.autocast():
                image_batch = pipe(**pipeline_args, output_type = output_type, guidance_scale = args.guidance_scale, generator=generator, num_inference_steps = args.num_inference_steps_val, num_images_per_prompt = args.num_images_per_prompt).images
                images += image_batch
    else:
        images = []
        for image in args.validation_images:
            image = Image.open(image)
            with torch.cuda.amp.autocast():
                image = pipe(**pipeline_args, image=image, guidance_scale = args.guidance_scale, generator=generator).images[0]
            images.append(image)

    torch.cuda.empty_cache()

    return images

def compute_KL(args, accelerator, pipe, prompt, weight_dtype, adapters, unwrap_model, samples):
    #Computes kl divergence between score_1 and score_2, using trajectories from score_prod_true
    
    pipe = pipe.to(accelerator.device)

    batch_size = 1
    device = accelerator.device
    errors = torch.zeros(args.num_inference_steps_val, requires_grad = False).to(device)
    noise_scheduler = pipe.scheduler


    pixel_values = samples["pixel_values"].to(dtype=weight_dtype)
    # Convert images to latent space
    model_input = pipe.vae.encode(pixel_values).latent_dist.mean#.sample()
    model_input = model_input * pipe.vae.config.scaling_factor


    bsz, channels, height, width = model_input.shape

    batch_samples = model_input


    batch_size = bsz

    prompt_embed = pipe._encode_prompt(
                prompt,
                device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )

    batch_samples_original = batch_samples.clone()

    errors = { "unconditional": torch.zeros(args.num_inference_steps_val, requires_grad = False),
              "conditional": torch.zeros(args.num_inference_steps_val, requires_grad = False),
              "cfg": torch.zeros(args.num_inference_steps_val, requires_grad = False)}
    #sample correct forward trajectories
    for i,t in tqdm(enumerate(reversed(pipe.scheduler.timesteps))):
        with torch.no_grad():
            #sample forward noise
            noise = torch.randn_like(batch_samples_original)
            batch_samples = noise_scheduler.add_noise(batch_samples_original, noise, t)
            
            
            if unwrap_model(pipe.unet).config.in_channels == channels * 2:
                batch_samples = torch.cat([batch_samples, batch_samples], dim=1)

            #get the pre-trained model prediction:
            latent_model_input = torch.cat([batch_samples] * 2)         
            # latent_model_input = pipe.scheduler.scale_model_input(batch_samples, t)

            preds = defaultdict(list)
            for adapter in adapters:
                # Predict noise residuals
                pipe.set_adapters(adapter)
                pipe.unet.eval()
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embed
                ).sample

                noise_pred = noise_pred.detach()

                # Perform guidance
                noise_pred_uncond = noise_pred.chunk(2)[0]
                noise_pred_text = noise_pred.chunk(2)[1]
                preds["unconditional"].append(noise_pred_uncond)
                preds["conditional"].append(noise_pred_text)
                preds["cfg"].append(noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond))

            for mode in preds:
                loss = F.mse_loss(preds[mode][0].float(), preds[mode][1].float(), reduction="mean").detach()
                errors[mode][i] += loss.item()
    # free memory
    del preds
    del noise_pred
    del noise_pred_uncond
    del noise_pred_text
    del latent_model_input
    del batch_samples
    del batch_samples_original
    torch.cuda.empty_cache()

    return errors, {"unconditional": torch.sum(errors["unconditional"]), "conditional": torch.sum(errors["conditional"]), "cfg": torch.sum(errors["cfg"])}

def compute_KL_weighted(args, accelerator, pipe, prompt, weight_dtype, adapter_list, adapter,  w, unwrap_model, samples):
    #Computes kl divergence between score_1 and score_2, using trajectories from score_prod_true
    
    pipe = pipe.to(accelerator.device)

    batch_size = 1
    device = accelerator.device
    errors = torch.zeros(args.num_inference_steps_val, requires_grad = False).to(device)
    noise_scheduler = pipe.scheduler


    pixel_values = samples["pixel_values"].to(dtype=weight_dtype)
    # Convert images to latent space
    model_input = pipe.vae.encode(pixel_values).latent_dist.mean#.sample()
    model_input = model_input * pipe.vae.config.scaling_factor


    bsz, channels, height, width = model_input.shape

    batch_samples = model_input


    batch_size = bsz

    prompt_embed = pipe._encode_prompt(
                prompt,
                device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )

    batch_samples_original = batch_samples.clone()

    errors = { "unconditional": torch.zeros(args.num_inference_steps_val, requires_grad = False),
              "conditional": torch.zeros(args.num_inference_steps_val, requires_grad = False),
              "cfg": torch.zeros(args.num_inference_steps_val, requires_grad = False)}
    #sample correct forward trajectories
    for i,t in tqdm(enumerate(reversed(pipe.scheduler.timesteps))):
        with torch.no_grad():
            #sample forward noise
            noise = torch.randn_like(batch_samples_original)
            batch_samples = noise_scheduler.add_noise(batch_samples_original, noise, t)
            
            
            if unwrap_model(pipe.unet).config.in_channels == channels * 2:
                batch_samples = torch.cat([batch_samples, batch_samples], dim=1)

            #get the pre-trained model prediction:
            latent_model_input = torch.cat([batch_samples] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            preds = defaultdict(list)
            if isinstance(w, torch.Tensor):
                w = w.tolist()
            pipe.set_adapters(adapter_list, adapter_weights=w)

            noise_pred_mixture = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embed
            ).sample.detach()
            noise_pred_uncond_mixture = noise_pred_mixture.chunk(2)[0]
            noise_pred_text_mixture = noise_pred_mixture.chunk(2)[1]
            preds["unconditional"].append(noise_pred_uncond_mixture)
            preds["conditional"].append(noise_pred_text_mixture)
            preds["cfg"].append(noise_pred_uncond_mixture + args.guidance_scale * (noise_pred_text_mixture - noise_pred_uncond_mixture))

            pipe.set_adapters([adapter], adapter_weights=[1.0])

            pipe.unet.eval()
            noise_pred_single = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embed
            ).sample.detach()
            noise_pred_uncond_single = noise_pred_single.chunk(2)[0]
            noise_pred_text_single = noise_pred_single.chunk(2)[1]
            preds["unconditional"].append(noise_pred_uncond_single)
            preds["conditional"].append(noise_pred_text_single)
            preds["cfg"].append(noise_pred_uncond_single + args.guidance_scale * (noise_pred_text_single - noise_pred_uncond_single))

            # Now preds[mode][0] is from the single 'adapter', preds[mode][1] is from the mixture
            for mode in preds:
                loss = F.mse_loss(preds[mode][0].float(), preds[mode][1].float(), reduction="mean").detach()
                errors[mode][i] += loss.item()


    return errors, {"unconditional": torch.sum(errors["unconditional"]), "conditional": torch.sum(errors["conditional"]), "cfg": torch.sum(errors["cfg"])}

def sample(accelerator, pipe, num_inference_steps, guidance_scale, prompt, adapters, w, batch_size, return_type = "PIL", mode="conditional"):
    #returns a list of PIL images
    device = accelerator.device
    if w is None:
        w = torch.ones(len(adapters), requires_grad = False)/len(adapters)
    with torch.no_grad():
        # Initialize the pipeline
        pipe = pipe.to(device)

        print(f"\nGenerating image with weights: {w}")
        
        # Initialize latents
        latents = torch.randn(
            (batch_size, pipe.unet.config.in_channels, 64, 64),
            device=device,
            dtype=pipe.unet.dtype  # Match UNet dtype
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        # Set timesteps
        pipe.scheduler.set_timesteps(num_inference_steps)
        
        timesteps = pipe.scheduler.timesteps

        # Encode the prompts
        prompt_embeds = pipe._encode_prompt(
                prompt,
                device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )

        # Denoising loop
        for t in tqdm(timesteps):
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise residuals
            noise_preds = []
            for i in range(len(adapters)):
                # set adapter
                pipe.set_adapters([adapters[i]], adapter_weights=[1.0])
                pipe.unet.eval()
                noise_preds.append(pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds
                ).sample)

            # Perform guidance
            noise_preds_uncond = []
            noise_preds_text = []
            for i in range(len(adapters)):
                noise_preds_uncond.append(noise_preds[i].chunk(2)[0])
                noise_preds_text.append(noise_preds[i].chunk(2)[1])
                if mode == "unconditional":
                    noise_preds[i] = noise_preds_uncond[i]
                elif mode == "conditional":
                    noise_preds[i] = noise_preds_text[i]
                elif mode == "cfg":
                    noise_preds[i] = noise_preds_uncond[i] + guidance_scale * (noise_preds_text[i] - noise_preds_uncond[i])
                else:
                    raise ValueError(f"Unknown mode {mode}")
            final_noise_pred = torch.zeros_like(noise_preds[0])
            for i in range(len(adapters)):
                final_noise_pred += w[i] * noise_preds[i]  # Cast weight to match noise pred dtype
            
            # Scheduler step
            latents = pipe.scheduler.step(final_noise_pred, t, latents).prev_sample

        # Decode latents to image
        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.float().detach().cpu().permute(0, 2, 3, 1).numpy()
        images_pil = pipe.numpy_to_pil(images)

        if return_type == "PIL":
            return images_pil
        elif return_type == "numpy":
            return images
        elif return_type == "latents":
            return latents*pipe.vae.config.scaling_factor