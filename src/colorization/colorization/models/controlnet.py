# models/controlnet.py
import torch
import torch.nn as nn
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTextModel
import torch.nn.functional as F

class ColorizationControlNet:
    def __init__(self, config):
        self.config = config
        
        # Initialize ControlNet for grayscale conditioning
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart",
            torch_dtype=torch.float16,
            use_safetensors=True
        )

        # Initialize SDXL components
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="vae",
            torch_dtype=torch.float16
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder",
            torch_dtype=torch.float16
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="unet",
            torch_dtype=torch.float16
        )

        # Create pipeline
        self.pipeline = StableDiffusionXLControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=None,  # Will be automatically loaded
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=None,  # Will be set during training
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(self.device)
        
    def prepare_latents(self, batch):
        """
        Chuẩn bị latent vectors từ ảnh đầu vào
        """
        with torch.no_grad():
            latents = self.vae.encode(batch["jpg"].to(self.device)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def prepare_controlnet_condition(self, batch):
        """
        Chuẩn bị điều kiện cho ControlNet từ ảnh grayscale
        """
        condition = batch["hint"].to(self.device)
        return condition
    
    def encode_prompt(self, batch):
        """
        Mã hóa prompt text thành embeddings
        """
        prompt_embeds = self.text_encoder(
            batch["txt"],
            return_dict=False
        )[0]
        return prompt_embeds
    
    def forward(self, batch):
        """
        Forward pass của model
        """
        # Prepare inputs
        latents = self.prepare_latents(batch)
        condition = self.prepare_controlnet_condition(batch)
        prompt_embeds = self.encode_prompt(batch)
        
        # Get ControlNet conditioning
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents,
            timesteps=None,  # Will be set by scheduler during training
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=condition,
            return_dict=False
        )
        
        # Run SDXL UNet with ControlNet conditioning
        noise_pred = self.unet(
            latents,
            timesteps=None,  # Will be set by scheduler during training
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        ).sample
        
        return noise_pred
    
    def train_step(self, batch, optimizer):
        """
        Một bước training
        """
        optimizer.zero_grad()
        
        # Forward pass
        noise_pred = self.forward(batch)
        
        # Calculate loss (MSE between predicted noise and target noise)
        target_noise = torch.randn_like(noise_pred)
        loss = F.mse_loss(noise_pred, target_noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def generate(self, condition_image, prompt, num_inference_steps=30):
        """
        Tạo ảnh màu từ ảnh grayscale
        """
        images = self.pipeline(
            prompt=prompt,
            image=condition_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
        ).images[0]
        
        return images
    
    def save_pretrained(self, save_path):
        """
        Lưu model
        """
        self.pipeline.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, load_path, config):
        """
        Load model đã trained
        """
        instance = cls(config)
        instance.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            load_path,
            torch_dtype=torch.float16
        )
        return instance

# Utility functions
def init_controlnet(config):
    """
    Helper function để khởi tạo ControlNet model
    """
    return ColorizationControlNet(config)