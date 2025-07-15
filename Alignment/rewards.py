"""
differentiable_losses.py

Provides differentiable reward-loss functions, with optional z-score normalization:

  - ImageReward
  - PickScorer
  - MPS
  - BLIP2/BLIP ITM
  - CLIP (clip_x_loss_fn)
  - HPS
  - AestheticScorer

Each function returns a closure `loss_fn(im_pix_un, prompts) -> (loss, rewards)`.
For the optional normalization, pass:
    do_normalize=True, baseline_mean=..., baseline_std=...
so that rewards = (rewards - mean) / (std + 1e-8).
"""

import os
import torch
import torchvision
from imscore.imreward.model import ImageReward 
from imscore.pickscore.model import PickScorer
from imscore.mps.model import MPS
from imscore.preference.model import CLIPScore
from aesthetic_scorer import AestheticScorerDiff
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModel, Blip2ForImageTextRetrieval, BlipForImageTextRetrieval
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

###############################################################################
# 1) ImageReward
###############################################################################
def imagereward_loss_fn(
    device: str = "cuda",
    model_name_or_path: str = "RE-N-Y/ImageReward",
    inference_dtype: torch.dtype = torch.float16,
    grad_scale: float = 1.0,
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    Creates a closure that loads ImageReward in specified precision.
    Optionally normalizes rewards via (r - mean)/std before computing loss.
    """

    model = ImageReward.from_pretrained(model_name_or_path).to(device, dtype=inference_dtype)
    model.eval()
    model.requires_grad_(False)

    def loss_fn(im_pix_un: torch.Tensor, prompts):
        # [B,3,H,W] in [-1,1] -> [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1)
        im_pix = im_pix.to(device=device, dtype=inference_dtype)

        # raw reward
        rewards = model.score(im_pix, prompts).squeeze()

        # optionally normalize
        if do_normalize:
            rewards = (rewards - baseline_mean) / (baseline_std + 1e-8)

        loss = -rewards * grad_scale
        return loss, rewards

    return loss_fn


###############################################################################
# 2) PickScorer
###############################################################################
def pickscore_loss_fn(
    device: str = "cuda",
    pickscorer_id: str = "yuvalkirstain/PickScore_v1",
    inference_dtype: torch.dtype = torch.float16,
    grad_scale: float = 1.0,
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    Returns a differentiable closure for the PickScorer model with optional z-score normalization.
    """

    model = PickScorer(pickscorer_id).to(device, dtype=inference_dtype)
    model.eval()
    model.requires_grad_(False)

    def loss_fn(im_pix_un: torch.Tensor, prompts):
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device, dtype=inference_dtype)
        rewards = model.score(im_pix, prompts).squeeze()

        if do_normalize:
            rewards = (rewards - baseline_mean) / (baseline_std + 1e-8)

        loss = -rewards * grad_scale
        return loss, rewards

    return loss_fn


###############################################################################
# 3) MPS
###############################################################################
def mps_loss_fn(
    device: str = "cuda",
    mps_id: str = "RE-N-Y/mpsv1",
    inference_dtype: torch.dtype = torch.float16,
    grad_scale: float = 0.1,
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    MPS (imscore) with optional reward normalization.
    """

    model = MPS.from_pretrained(mps_id).to(device, dtype=inference_dtype)
    model.eval()
    model.requires_grad_(False)

    def loss_fn(im_pix_un: torch.Tensor, prompts):
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device, dtype=inference_dtype)

        rewards = model.score(im_pix, prompts).squeeze()

        if do_normalize:
            rewards = (rewards - baseline_mean) / (baseline_std + 1e-8)

        loss = -rewards * grad_scale
        return loss, rewards

    return loss_fn


###############################################################################
# 4) BLIP2/BLIP ITM
###############################################################################
def blip2_itm_loss_fn(
    device="cuda",
    model_name="Salesforce/blip-itm-base-coco",
    grad_scale=1.0,
    config_prompt=None,
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    BLIP2/BLIP ITM w/ optional normalization.
    """

    processor = AutoProcessor.from_pretrained(model_name)
    if "blip2" in model_name:
        model = Blip2ForImageTextRetrieval.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    else:
        model = BlipForImageTextRetrieval.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
    model.eval()
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)

        text_prompts = [f"A photorealistic image of a {prompt} fully in frame, set in its natural habitat, with sharp details, vibrant colors, and soft natural lighting." for prompt in prompts]

        inputs = processor(
            images=im_pix,
            text=text_prompts,
            return_tensors="pt",
            padding=True
        ).to(device, torch.float16)

        if "blip2" in model_name:
            itm_out = model(**inputs, use_image_text_matching_head=True)
            logits = itm_out.logits_per_image
        else:
            itm_out = model(**inputs, use_itm_head=True)
            logits = itm_out[0]

        probabilities = torch.softmax(logits, dim=1)
        rewards = probabilities[:, 1]  # matching probability

        if do_normalize:
            rewards = (rewards - baseline_mean) / (baseline_std + 1e-8)

        loss = -rewards * grad_scale
        return loss, rewards

    return loss_fn


###############################################################################
# 5) CLIP (clip_x_loss_fn) using imscore CLIPScore
###############################################################################

def clip_x_loss_fn(
    device: str = "cuda",
    clip_model_name: str = "openai/clip-vit-large-patch14",
    grad_scale: float = 1.0,
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    CLIP score alignment with optional normalization.
    If `x` is provided at call time, it prepends the prompt with "x: prompt".
    Uses imscore's CLIPScore model internally.
    
    Args:
        device (str): e.g. "cuda" or "cpu".
        clip_model_name (str): e.g. "openai/clip-vit-large-patch14".
        grad_scale (float): factor to scale the gradient.
        do_normalize (bool): if True, apply z-score to the reward with baseline_mean/std.
        baseline_mean (float): mean used for normalization.
        baseline_std (float): std used for normalization.
    """

    # Use imscore's CLIPScore model
    model = CLIPScore(clip_model_name).to(device)
    model.eval()
    model.requires_grad_(False)

    def loss_fn(im_pix_un, prompts, x=None):
        """
        Args:
            im_pix_un (torch.Tensor): [B,3,H,W], in [-1,1].
            prompts (List[str]): length B text prompts.
            x (str or None): optional style or domain prefix.
        
        Returns:
            (loss, rewards):
              loss: shape [B], negative reward => maximize CLIPScore.
              rewards: shape [B].
        """
        # scale images from [-1,1] to [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1)

        # prepend x if provided
        if x is not None:
            styled_prompts = [f"{x}: {p}" for p in prompts]
        else:
            styled_prompts = prompts

        # get CLIPScore from imscore
        rewards = model.score(im_pix.to(device), styled_prompts)

        # optionally normalize rewards
        if do_normalize:
            rewards = (rewards - baseline_mean) / (baseline_std + 1e-8)

        # negative => maximizing CLIPScore
        loss = -rewards * grad_scale
        return loss, rewards

    return loss_fn



###############################################################################
# 6) HPS
###############################################################################
def hps_loss_fn(inference_dtype=None, device=None,
                do_normalize: bool = False,
                baseline_mean: float = 0.0,
                baseline_std: float = 1.0):
    """
    HPS with optional normalization.
    """

    model_name = "ViT-H-14"
    model, _, _ = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    

    tokenizer = get_tokenizer(model_name)

    # Download checkpoint if needed
    import requests
    from tqdm import tqdm

    link = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
    os.makedirs(os.path.expanduser('~/.cache/hpsv2'), exist_ok=True)
    checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"

    if not os.path.exists(checkpoint_path):
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(checkpoint_path, 'wb') as file, tqdm(
            desc="Downloading HPS_v2_compressed.pt",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

    # We call hpsv2.score just to force model download
    hpsv2.score([], "")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device, dtype=inference_dtype)
    model.eval()
    model.requires_grad_(False)

    target_size = 224
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    def loss_fn(im_pix, prompts):
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1)
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)

        caption = tokenizer(prompts).to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)

        if do_normalize:
            scores = (scores - baseline_mean) / (baseline_std + 1e-8)

        loss = - scores
        return loss, scores

    return loss_fn


###############################################################################
# 7) AestheticScorer
###############################################################################
def aesthetic_loss_fn(aesthetic_target=None,
                      grad_scale=0,
                      device=None,
                      accelerator=None,
                      torch_dtype=None,
                      do_normalize: bool = False,
                      baseline_mean: float = 0.0,
                      baseline_std: float = 1.0):
    """
    AestheticScorer with optional normalization.
    If aesthetic_target is None, we do a default max reward => loss = -rewards.
    Otherwise L1 distance => loss = |rewards - aesthetic_target|.
    """

    target_size = 224
    normalize_transform = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.eval()
    scorer.requires_grad_(False)

    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = torchvision.transforms.Resize(target_size)(im_pix)
        im_pix = normalize_transform(im_pix).to(im_pix_un.dtype)

        rewards = scorer(im_pix)  # shape [B]

        if do_normalize:
            rewards = (rewards - baseline_mean) / (baseline_std + 1e-8)

        if aesthetic_target is None:
            loss = -rewards
        else:
            # e.g. L1 distance
            loss = - rewards

        return loss * grad_scale, rewards

    return loss_fn

###############################################################################
# 8) Brightness / Exposure
###############################################################################
def brightness_loss_fn(
    device: str = "cuda",
    grad_scale: float = 1.0,
    target_brightness: float = None,
    direction: str = "high",
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    Differentiable brightness/exposure reward.
    Calculates Y = 0.2126 R + 0.7152 G + 0.0722 B and takes the mean.
    
    Args:
        device: Device to run on.
        grad_scale: Factor to scale gradients.
        target_brightness: If provided, minimizes distance to target; otherwise follows direction.
        direction: Either "high" (maximize brightness) or "low" (minimize brightness).
        do_normalize: Whether to apply z-score normalization to rewards.
        baseline_mean: Mean for normalization.
        baseline_std: Standard deviation for normalization.
    """
    
    if direction not in ["high", "low"]:
        raise ValueError(f"direction must be 'high' or 'low', got {direction}")
    
    def loss_fn(im_pix_un: torch.Tensor, *args):
        # Convert from [-1,1] to [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device)
        
        # Calculate luminance Y = 0.2126 R + 0.7152 G + 0.0722 B
        # im_pix has shape [B,3,H,W]
        r, g, b = im_pix[:,0], im_pix[:,1], im_pix[:,2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Mean brightness per image
        brightness = y.mean(dim=(1,2))
        
        if do_normalize:
            brightness = (brightness - baseline_mean) / (baseline_std + 1e-8)
        
        if target_brightness is not None:
            # Loss is distance to target
            loss = torch.abs(brightness - target_brightness) * grad_scale
        else:
            # Set sign based on direction
            sign = -1 if direction == "high" else 1
            loss = sign * brightness * grad_scale
            
        return loss, brightness
    
    return loss_fn

###############################################################################
# 9) Global RMS Contrast
###############################################################################
def global_contrast_loss_fn(
    device: str = "cuda",
    grad_scale: float = 1.0,
    target_contrast: float = None,
    direction: str = "high",
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    Differentiable global RMS contrast reward.
    Calculates the standard deviation of luminance Y.
    
    Args:
        device: Device to run on.
        grad_scale: Factor to scale gradients.
        target_contrast: If provided, minimizes distance to target; otherwise follows direction.
        direction: Either "high" (maximize contrast) or "low" (minimize contrast).
        do_normalize: Whether to apply z-score normalization to rewards.
        baseline_mean: Mean for normalization.
        baseline_std: Standard deviation for normalization.
    """
    
    if direction not in ["high", "low"]:
        raise ValueError(f"direction must be 'high' or 'low', got {direction}")
    
    def loss_fn(im_pix_un: torch.Tensor, *args):
        # Convert from [-1,1] to [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device)
        
        # Calculate luminance Y
        r, g, b = im_pix[:,0], im_pix[:,1], im_pix[:,2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Calculate contrast as standard deviation of Y
        contrast = torch.std(y, dim=(1,2))
        
        if do_normalize:
            contrast = (contrast - baseline_mean) / (baseline_std + 1e-8)
        
        if target_contrast is not None:
            # Loss is distance to target
            loss = torch.abs(contrast - target_contrast) * grad_scale
        else:
            # Set sign based on direction
            sign = -1 if direction == "high" else 1
            loss = sign * contrast * grad_scale
            
        return loss, contrast
    
    return loss_fn

###############################################################################
# 10) Local Contrast
###############################################################################
def local_contrast_loss_fn(
    device: str = "cuda",
    kernel_size: int = 5,
    sigma: float = 1.0,
    grad_scale: float = 1.0,
    target_contrast: float = None,
    direction: str = "high",
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    Differentiable local contrast reward.
    Applies Gaussian blur to luminance Y and calculates |Y-Y_blur|.mean().
    
    Args:
        device: Device to run on.
        kernel_size: Size of Gaussian kernel for blurring.
        sigma: Standard deviation for Gaussian kernel.
        grad_scale: Factor to scale gradients.
        target_contrast: If provided, minimizes distance to target; otherwise follows direction.
        direction: Either "high" (maximize local contrast) or "low" (minimize local contrast).
        do_normalize: Whether to apply z-score normalization to rewards.
        baseline_mean: Mean for normalization.
        baseline_std: Standard deviation for normalization.
    """
    
    if direction not in ["high", "low"]:
        raise ValueError(f"direction must be 'high' or 'low', got {direction}")
    
    def loss_fn(im_pix_un: torch.Tensor, *args):
        # Convert from [-1,1] to [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device)
        
        # Calculate luminance Y
        r, g, b = im_pix[:,0], im_pix[:,1], im_pix[:,2]
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Apply Gaussian blur
        padding = kernel_size // 2
        gaussian_blur = torchvision.transforms.GaussianBlur(
            kernel_size=kernel_size, 
            sigma=sigma
        )
        y_blur = gaussian_blur(y.unsqueeze(1)).squeeze(1)  # Add/remove channel dim
        
        # Calculate local contrast
        local_contrast = (y - y_blur).abs().mean(dim=(1,2))
        
        if do_normalize:
            local_contrast = (local_contrast - baseline_mean) / (baseline_std + 1e-8)
        
        if target_contrast is not None:
            # Loss is distance to target
            loss = torch.abs(local_contrast - target_contrast) * grad_scale
        else:
            # Set sign based on direction
            sign = -1 if direction == "high" else 1
            loss = sign * local_contrast * grad_scale
            
        return loss, local_contrast
    
    return loss_fn

###############################################################################
# 11) Saturation
###############################################################################
def saturation_loss_fn(
    device: str = "cuda",
    grad_scale: float = 1.0,
    target_saturation: float = None,
    direction: str = "high",
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0,
    epsilon: float = 1e-8
):
    """
    Differentiable saturation reward.
    Calculates S = (max(R,G,B) - min(R,G,B)) / max(R,G,B,epsilon).
    
    Args:
        device: Device to run on.
        grad_scale: Factor to scale gradients.
        target_saturation: If provided, minimizes distance to target; otherwise follows direction.
        direction: Either "high" (maximize saturation) or "low" (minimize saturation).
        do_normalize: Whether to apply z-score normalization to rewards.
        baseline_mean: Mean for normalization.
        baseline_std: Standard deviation for normalization.
        epsilon: Small value to avoid division by zero.
    """
    
    if direction not in ["high", "low"]:
        raise ValueError(f"direction must be 'high' or 'low', got {direction}")
    
    def loss_fn(im_pix_un: torch.Tensor, *args):
        # Convert from [-1,1] to [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device)
        
        # Calculate max and min of RGB channels
        r, g, b = im_pix[:,0], im_pix[:,1], im_pix[:,2]
        max_rgb = torch.maximum(torch.maximum(r, g), b)
        min_rgb = torch.minimum(torch.minimum(r, g), b)
        
        # Calculate saturation: (max-min)/max
        saturation = (max_rgb - min_rgb) / (max_rgb + epsilon)
        
        # Mean saturation per image
        mean_saturation = saturation.mean(dim=(1,2))
        
        if do_normalize:
            mean_saturation = (mean_saturation - baseline_mean) / (baseline_std + 1e-8)
        
        if target_saturation is not None:
            # Loss is distance to target
            loss = torch.abs(mean_saturation - target_saturation) * grad_scale
        else:
            # Set sign based on direction
            sign = -1 if direction == "high" else 1
            loss = sign * mean_saturation * grad_scale
            
        return loss, mean_saturation
    
    return loss_fn

###############################################################################
# 12) Colorfulness / Chroma Energy
###############################################################################
def colorfulness_loss_fn(
    device: str = "cuda",
    grad_scale: float = 1.0,
    target_colorfulness: float = None,
    direction: str = "high",
    do_normalize: bool = False,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0
):
    """
    Differentiable colorfulness/chroma energy reward using Hasler-Süsstrunk algorithm.
    Calculates √(σ_rg² + σ_yb²) + 0.3 √(μ_rg² + μ_yb²) with rg=R-G, yb=0.5·(R+G)-B.
    
    Args:
        device: Device to run on.
        grad_scale: Factor to scale gradients.
        target_colorfulness: If provided, minimizes distance to target; otherwise follows direction.
        direction: Either "high" (maximize colorfulness) or "low" (minimize colorfulness).
        do_normalize: Whether to apply z-score normalization to rewards.
        baseline_mean: Mean for normalization.
        baseline_std: Standard deviation for normalization.
    """
    
    if direction not in ["high", "low"]:
        raise ValueError(f"direction must be 'high' or 'low', got {direction}")
    
    def loss_fn(im_pix_un: torch.Tensor, *args):
        # Convert from [-1,1] to [0,1]
        im_pix = ((im_pix_un * 0.5) + 0.5).clamp(0, 1).to(device)
        
        # Calculate opponent color channels
        r, g, b = im_pix[:,0], im_pix[:,1], im_pix[:,2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        
        # Calculate statistics per image
        rg_mean = rg.mean(dim=(1,2))
        yb_mean = yb.mean(dim=(1,2))
        rg_std = torch.std(rg, dim=(1,2))
        yb_std = torch.std(yb, dim=(1,2))
        
        # Calculate Hasler-Süsstrunk colorfulness metric
        colorfulness = torch.sqrt(rg_std**2 + yb_std**2) + 0.3 * torch.sqrt(rg_mean**2 + yb_mean**2)
        
        if do_normalize:
            colorfulness = (colorfulness - baseline_mean) / (baseline_std + 1e-8)
        
        if target_colorfulness is not None:
            # Loss is distance to target
            loss = torch.abs(colorfulness - target_colorfulness) * grad_scale
        else:
            # Set sign based on direction
            sign = -1 if direction == "high" else 1
            loss = sign * colorfulness * grad_scale
            
        return loss, colorfulness
    
    return loss_fn
