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
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import torch
from accelerate import Accelerator
import os
import argparse
import sys
import random
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
from my_utils import *

#arguments
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_logging", type=int, default=0, help="Whether to use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="super_diffusion", help="Wandb project name")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance_scale_sampling", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--guidance_scale_kl", type=float, default=15.0, help="Guidance scale")
    parser.add_argument("--lr_dual", type=float, default=0.001, help="Learning rate for dual")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--run_validation", type=int, default=1, help="Run validation")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images")
    parser.add_argument("--running_average_window", type=int, default=5, help="Running average window for computing final mu")
    if input_args is not None:
        print("Parsing provided arguments:", input_args)
        args = parser.parse_args(input_args)
    else:
        print("Parsing command line arguments:", sys.argv[1:])
        args = parser.parse_args()

    # Add debug print
    print("\nParsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):

    

    accelerator = Accelerator()
    device = accelerator.device

    #loading models
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
    )

    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()
    text_encoder.gradient_checkpointing_enable()

    # Use a more efficient scheduler
    scheduler = DDIMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="scheduler",
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        steps_offset=1,
    )

    vae, unet, text_encoder = accelerator.prepare(vae, unet, text_encoder)

    OBJ_VALUES=["a mountain landscape", "a flamingo", "a dragonfly", "dandelion", "a sunflower", "a rocket", "moon", "a snail", "an eagle", "zebra", "chess pawn", "a pineapple", "a spider web", "a waffle cone", "a cat", "a chair", "a donut", "otter", "pebbles on a beach", "teddy bear"]
    BG_VALUES=["silhouette of a dog", "a candy cane",  "a helicopter", "fireworks", "a lemon", "a cactus", "cookie", "a cinnamon roll", "an airplane", "barcode", "bottle cap", "a beehive", "a bicycle wheel", "a volcano", "a dog", "an avocado", "a map", "duck", "a turtle", "panda"]
    
    # Randomly sample one prompt from each list
    obj_prompt_1 = OBJ_VALUES[torch.randint(0, len(OBJ_VALUES), (1,)).item()]
    obj_prompt_2 = OBJ_VALUES[torch.randint(0, len(OBJ_VALUES), (1,)).item()]

    while(obj_prompt_1 == obj_prompt_2):
        obj_prompt_2 = OBJ_VALUES[torch.randint(0, len(OBJ_VALUES), (1,)).item()]

    bg_prompt = BG_VALUES[torch.randint(0, len(BG_VALUES), (1,)).item()]

    obj_prompt_1 = "moon"
    obj_prompt_2 = "teddy bear"
    bg_prompt = "a beehive"

    set_seed(args.seed)

    
    prompts = [obj_prompt_1, obj_prompt_2, bg_prompt]

    init_mu = torch.ones(len(prompts), device=device)/len(prompts)
    mu = init_mu.clone()

    print(f"Running job with OBJ1={obj_prompt_1}, OBJ2={obj_prompt_2}, BG={bg_prompt}")

    # Initialize wandb before training loop
    if args.wandb_logging and accelerator.is_main_process:
        run_name = f"train_{obj_prompt_1}, {obj_prompt_2}, {bg_prompt}"
        wandb.init(project=args.wandb_project, name=run_name)
        # Log configuration parameters
        wandb.config.update(vars(args))

    mu_history = torch.zeros(args.num_epochs, len(prompts), device=device)

    for epoch in range(args.num_epochs):

        with torch.no_grad():
            step_random = random.randint(1, 10000)
            device_seed = args.seed + accelerator.process_index + step_random
            images, lats = sample_loop_product(prompts,
                                             mu=mu,
                                             tokenizer=tokenizer,
                                             scheduler=scheduler,
                                             vae=vae,
                                             unet=unet,
                                             text_encoder=text_encoder,
                                             device=device,
                                             return_type="all",
                                             num_inference_steps=args.num_inference_steps,
                                             guidance_scale=args.guidance_scale_sampling,
                                             num_samples=args.batch_size,
                                             generator_seed=device_seed)
                                             
            kls = compute_KL(lats,
                           prompts,
                           scheduler,
                           tokenizer,
                           text_encoder,
                           unet,
                           vae,
                           num_inference_steps=args.num_inference_steps,
                           mu=mu,
                           guidance_scale=args.guidance_scale_kl)
            
            if accelerator.is_main_process and args.wandb_logging:
                image_np = images[0].cpu().numpy()

                wandb.log({
                    f"image": wandb.Image(image_np),
                }, step=epoch)

                for i in range(len(prompts)):
                    wandb.log({
                        f"kl_{i}": kls[i].item(),
                        f"mu_{i}": mu[i].item(),
                    }, step=epoch)

            mu = dual_step(args.lr_dual, kls, mu)

            # Gather clip scores across devices

            mu = accelerator.gather(mu)

            mu = torch.reshape(mu, (accelerator.num_processes, len(prompts)))

            mu = mu.mean(0)

            mu_history[epoch] = mu


    # Take average of last n mu values from mu_history

    final_mu = torch.stack([mu_history[i] for i in range(max(0, len(mu_history)-args.running_average_window), len(mu_history))]).mean(0)
        
    # Close the wandb run
    if args.wandb_logging and accelerator.is_main_process:
        wandb.finish()


    if args.run_validation:
        #validation
        weights = [
            init_mu, 
            final_mu, 
        ]

        if args.wandb_logging and accelerator.is_main_process:
            run_name = f"val_{obj_prompt_1}, {obj_prompt_2}, {bg_prompt}"
            wandb.init(project=args.wandb_project, name=run_name)
            # Log configuration parameters
            wandb.config.update({"final_mu": final_mu.tolist()})
            wandb.config.update(vars(args))



        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g").to(device)
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")


        for step in range(args.num_images):

  
            #joint prompting baseline
            prompts_joint = [obj_prompt_1 + ", " + obj_prompt_2 + "," + bg_prompt]
            mu_joint = torch.ones(1, device=device)/1
            step_random = random.randint(1, 10000)
            device_seed = args.seed + accelerator.process_index + step_random
            images = sample_loop_product(prompts_joint,
                                             mu=mu_joint,
                                             tokenizer=tokenizer,
                                             scheduler=scheduler,
                                             vae=vae,
                                             unet=unet,
                                             text_encoder=text_encoder,
                                             device=device,
                                             return_type="image",
                                             num_inference_steps=args.num_inference_steps,
                                             guidance_scale=args.guidance_scale_sampling,
                                             num_samples=args.batch_size,
                                             generator_seed=device_seed)
            images_np = images.cpu().numpy()
                
            # Convert to PIL for CLIP
            pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images_np]
            
            # Get CLIP scores
            inputs = clip_processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(device)
            outputs = clip_model(**inputs)
            clip_scores = outputs.logits_per_image.detach()

            

            

            # Get BLIP scores
            imgs = torch.tensor(images_np)
            imgs = torch.permute(imgs, (0, 3, 1, 2))
            img_true = (imgs * 255).clamp(1, 255).to(torch.int32)
            texts = [obj_prompt_1, obj_prompt_2, bg_prompt]


            inputs = processor(images=img_true, text=texts, return_tensors="pt", padding=True).to(device)
            itc_out = model(**inputs, use_image_text_matching_head=False)
            logits_per_image = itc_out.logits_per_image
            blip_scores = logits_per_image.detach()

           


            min_clip = torch.min(clip_scores, dim=1, keepdim=True)[0]
            min_blip = torch.min(blip_scores, dim=1, keepdim=True)[0]

            min_clip = min_clip.mean(0)
            min_blip = min_blip.mean(0)

            accelerator.wait_for_everyone()
            min_clip = accelerator.gather(min_clip)
            min_blip = accelerator.gather(min_blip)


            min_clip = min_clip.mean(0)
            min_blip = min_blip.mean(0)

            #average on device
            blip_scores = blip_scores.mean(0)

            # Gather BLIP scores across devices
            accelerator.wait_for_everyone()
            blip_scores = accelerator.gather(blip_scores)
            blip_scores = torch.reshape(blip_scores, (accelerator.num_processes, len(prompts)))
            blip_scores = blip_scores.mean(0)

            #average on device
            clip_scores = clip_scores.mean(0)
            # Gather clip scores across devices
            accelerator.wait_for_everyone()
            clip_scores = accelerator.gather(clip_scores)
            clip_scores = torch.reshape(clip_scores, (accelerator.num_processes, len(prompts)))
            clip_scores = clip_scores.mean(0)
            

            if args.wandb_logging and accelerator.is_main_process:
                wandb.log({
                    f"min_clip_score_joint": min_clip.item(),
                    f"min_blip_score_joint": min_blip.item(),
                }, step=step)


            if args.wandb_logging and accelerator.is_main_process:
                for j in range(len(prompts)):
                    wandb.log({
                        f"clip_score_prompt_{j}_joint": clip_scores[j].item(),
                        f"blip_score_prompt_{j}_joint": blip_scores[j].item(),
                    }, step=step)


            
                # Log everything to wandb
                wandb.log({
                    f"image_joint": wandb.Image(images_np[0]),
                }, step=step)



            

            for i, mu in enumerate(weights):
                # Generate image with device-specific seed
                # Set device-specific random seed for reproducibility
                # Sample random number for this step
                step_random = random.randint(1, 10000)
                device_seed = args.seed + accelerator.process_index + step_random
                images = sample_loop_product(prompts,
                                             mu=mu,
                                             tokenizer=tokenizer,
                                             scheduler=scheduler,
                                             vae=vae,
                                             unet=unet,
                                             text_encoder=text_encoder,
                                             device=device,
                                             return_type="image",
                                             num_inference_steps=args.num_inference_steps,
                                             guidance_scale=args.guidance_scale_sampling,
                                             num_samples=args.batch_size,
                                             generator_seed=device_seed)
                
                images_np = images.cpu().numpy()
                
                # Convert to PIL for CLIP
                pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images_np]
                
                # Get CLIP scores
                inputs = clip_processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(device)
                outputs = clip_model(**inputs)
                clip_scores = outputs.logits_per_image.detach()


                # Get BLIP scores
                imgs = torch.tensor(images_np)
                imgs = torch.permute(imgs, (0, 3, 1, 2))
                img_true = (imgs * 255).clamp(1, 255).to(torch.int32)
                texts = [obj_prompt_1, obj_prompt_2, bg_prompt]


                inputs = processor(images=img_true, text=texts, return_tensors="pt", padding=True).to(device)
                itc_out = model(**inputs, use_image_text_matching_head=False)
                logits_per_image = itc_out.logits_per_image
                blip_scores = logits_per_image.detach()





                min_clip = torch.min(clip_scores, dim=1, keepdim=True)[0]
                min_blip = torch.min(blip_scores, dim=1, keepdim=True)[0]

                min_clip = min_clip.mean(0)
                min_blip = min_blip.mean(0)

                accelerator.wait_for_everyone()
                min_clip = accelerator.gather(min_clip)
                min_blip = accelerator.gather(min_blip)


                min_clip = min_clip.mean(0)
                min_blip = min_blip.mean(0)

                #average on device
                blip_scores = blip_scores.mean(0)
                # Gather BLIP scores across devices
                accelerator.wait_for_everyone()
                blip_scores = accelerator.gather(blip_scores)
                blip_scores = torch.reshape(blip_scores, (accelerator.num_processes, len(prompts)))
                blip_scores = blip_scores.mean(0)

                #average on device
                clip_scores = clip_scores.mean(0)
                # Gather clip scores across devices
                accelerator.wait_for_everyone()
                clip_scores = accelerator.gather(clip_scores)
                clip_scores = torch.reshape(clip_scores, (accelerator.num_processes, len(prompts)))
                clip_scores = clip_scores.mean(0)
                

                if args.wandb_logging and accelerator.is_main_process:
                    wandb.log({
                        f"min_clip_score_weights_{i}": min_clip.item(),
                        f"min_blip_score_weights_{i}": min_blip.item(),
                    }, step=step)


                
                if args.wandb_logging and accelerator.is_main_process:
                    for j in range(len(prompts)):
                        wandb.log({
                            f"clip_score_prompt_{j}_weights_{i}": clip_scores[j].item(),
                            f"blip_score_prompt_{j}_weights_{i}": blip_scores[j].item(),
                        }, step=step)
                
                    # Log everything to wandb
                    wandb.log({
                        f"image_weights_{i}": wandb.Image(images_np[0]),
                    }, step=step)

            


        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)
