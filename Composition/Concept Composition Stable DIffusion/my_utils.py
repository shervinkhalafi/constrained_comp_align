import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from diffusers import DDIMScheduler
from tqdm.auto import tqdm
import numpy as np
import torch.nn.functional as F


def sample_loop(prompts, num_inference_steps=10, guidance_scale=7.5, generator_seed=None):

    # Set generator seed if provided
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=torch_device).manual_seed(generator_seed)

    with torch.no_grad():
        # Get text embeddings
        text_embeddings = []
        for prompt in prompts:
            text_embeddings.append(get_text_embedding(prompt, tokenizer, device))
        text_embeddings = torch.cat(text_embeddings)
        
        # Get unconditional embeddings for classifier free guidance
        uncond_embeddings = get_text_embedding([""] * len(prompts), tokenizer, device)
        
        # Prepare scheduler
        scheduler.set_timesteps(num_inference_steps)
        
        # Prepare latents
        latents = torch.randn(
            (len(prompts), unet.config.in_channels, 64, 64),
            device=torch_device,
            dtype=torch.float32,
            generator=generator
        )
        latents = latents * scheduler.init_noise_sigma
        
        # Sampling loop
        progress_bar = tqdm(total=len(scheduler.timesteps))
        for t in scheduler.timesteps:
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Get model prediction
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings])
            ).sample


            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            progress_bar.update(1)
        progress_bar.close()
        
        # Decode latents to image
        # Decode latents to image using VAE
        latents = latents / vae.config.scaling_factor
        image = vae.decode(latents, return_dict=False)[0]
        
        # Normalize image to [0,1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to uint8 format
        image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        
    return image


def sample_loop_product(prompts, tokenizer, scheduler, vae, unet, text_encoder, device,num_inference_steps=10, guidance_scale=15.0, return_type = "image", mu = None, generator_seed=None, num_samples=1):
    
    if mu is None:
        mu = torch.ones(len(prompts), device=device)/len(prompts)

    # Set generator seed if provided
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)

    with torch.no_grad():
        # Pre-compute all text embeddings once
        text_embeddings = []
        for prompt in prompts:
            text_embeddings.append(get_text_embedding(prompt, tokenizer, text_encoder, device))
        text_embeddings = torch.stack(text_embeddings)  # [num_prompts, seq_len, hidden_size]
        
        # Get unconditional embeddings for classifier free guidance
        uncond_embeddings = get_text_embedding([""], tokenizer, text_encoder, device)
        
        # Prepare scheduler
        scheduler.set_timesteps(num_inference_steps)
        
        # Get in_channels from the module if using DDP, otherwise directly from unet
        in_channels = unet.module.config.in_channels if hasattr(unet, 'module') else unet.config.in_channels
        
        # Prepare latents
        latents = torch.randn(
            (num_samples, in_channels, 64, 64),
            device=device,
            dtype=torch.float32,
            generator=generator
        )
        latents = latents * scheduler.init_noise_sigma
        
        # Pre-compute repeated embeddings for efficiency
        uncond_embeddings_repeated = uncond_embeddings.repeat(num_samples, 1, 1)
        
        # Sampling loop
        progress_bar = tqdm(total=len(scheduler.timesteps))
        for t in scheduler.timesteps:
            # Expand latents for classifier free guidance
            latent_model_input = latents
            
            # Get model prediction for unconditional
            noise_pred_uncond = unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_embeddings_repeated
            ).sample

            # Process all prompts in parallel
            latent_model_input = torch.cat([latents] * 2)
            noise_preds_text = []
            noise_preds_uncond = []

            for i in range(len(prompts)):
                # Repeat text embeddings for each sample
                text_emb = text_embeddings[i].repeat(num_samples, 1, 1)
                uncond_emb = uncond_embeddings_repeated
                
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([uncond_emb, text_emb])
                ).sample
                noise_preds_text.append(noise_pred.chunk(2)[1])
                noise_preds_uncond.append(noise_pred.chunk(2)[0])

            # Perform guidance
            noise_pred = noise_pred_uncond

            for i in range(len(prompts)):
                noise_pred += mu[i] * guidance_scale * (noise_preds_text[i] - noise_preds_uncond[i])

            # Compute previous noisy sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            progress_bar.update(1)
        progress_bar.close()
        
        # Decode latents to image
        # Get scaling factor from the module if using DDP, otherwise directly from vae
        scaling_factor = vae.module.config.scaling_factor if hasattr(vae, 'module') else vae.config.scaling_factor
        latents = latents / scaling_factor
        
        # Get the VAE model (either the module or the model itself)
        vae_model = vae.module if hasattr(vae, 'module') else vae
        image = vae_model.decode(latents, return_dict=False)[0]
        
        # Normalize image to [0,1] range
        image = (image / 2 + 0.5).clamp(0, 1)
        
        # Convert to uint8 format
        image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8)
    if return_type == "image":
        return image
    elif return_type == "latents":
        return latents
    elif return_type == "all":
        return image, latents
    

def get_text_embedding(prompt, tokenizer, text_encoder, device):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    embeddings = text_encoder(text_input.input_ids.to(device))[0]
    return embeddings  # [seq_len, hidden_size]


def compute_KL(init_latents, prompts, scheduler, tokenizer, text_encoder, unet, vae, num_inference_steps=10, guidance_scale=7.5, mu = None):
    
    device = init_latents.device
    num_samples = init_latents.shape[0]  # Get number of samples from input latents

    with torch.no_grad():
        # Pre-compute all text embeddings once
        text_embeddings = []
        for prompt in prompts:
            text_embeddings.append(get_text_embedding(prompt, tokenizer, text_encoder, device))
        text_embeddings = torch.stack(text_embeddings)  # [num_prompts, seq_len, hidden_size]
        
        # Get unconditional embeddings for classifier free guidance
        uncond_embeddings = get_text_embedding([""], tokenizer, text_encoder, device)
        
        # Prepare scheduler
        scheduler.set_timesteps(num_inference_steps)
        
        latents = init_latents
        noise = torch.randn_like(latents)

        kls = torch.zeros(len(prompts), device=device)

        # Pre-compute repeated embeddings for efficiency
        uncond_embeddings_repeated = uncond_embeddings.repeat(num_samples, 1, 1)

        # Sampling loop
        progress_bar = tqdm(total=len(scheduler.timesteps))
        for t in scheduler.timesteps:
            # Compute previous noisy sample
            latents = scheduler.add_noise(init_latents, noise, t)
            latent_model_input = latents

            # Get model prediction for unconditional
            noise_pred_uncond = unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_embeddings_repeated
            ).sample

            # Process all prompts in parallel
            latent_model_input = torch.cat([latents] * 2)
            noise_preds_text = []
            noise_preds_uncond = []

            for i in range(len(prompts)):
                # Repeat text embeddings for each sample
                text_emb = text_embeddings[i].repeat(num_samples, 1, 1)
                uncond_emb = uncond_embeddings_repeated
                
                noise_pred_cond = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([uncond_emb, text_emb])
                ).sample
                noise_preds_text.append(noise_pred_cond.chunk(2)[1])
                noise_preds_uncond.append(noise_pred_cond.chunk(2)[0])

            # Perform guidance
            # noise_pred = 0.0*noise_pred_uncond.clone()
            noise_pred = 1.0*noise_pred_uncond.clone()

            for i in range(len(prompts)):
                noise_pred += mu[i] * guidance_scale * (noise_preds_text[i] - noise_preds_uncond[i])

            # Compute KL divergence for all prompts at once
            for i in range(len(prompts)):
                # kls[i] += F.mse_loss((noise_preds_text[i] - noise_preds_uncond[i]), noise_pred)
                kls[i] += F.mse_loss((noise_preds_text[i]), noise_pred)
            
            progress_bar.update(1)
        progress_bar.close()

    return kls


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

def dual_step(lr_dual, KLs, mu, b = None):

    l = torch.tensor(KLs, requires_grad = False).to(mu.device)
    if b is None:
        b = torch.zeros_like(mu)
    b = b.to(mu.device)
    grad = lr_dual*(l - b)

    mu = (mu + grad).detach()

    mu = project_onto_simplex(mu)

    return mu
