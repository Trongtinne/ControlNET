class ModelConfig:
    def __init__(self):
        # SDXL Config
        self.pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        self.scheduler_name = "UniPCMultistepScheduler"
        
        # LoRA Config
        self.lora_r = 16  # LoRA rank
        self.lora_alpha = 32  # LoRA scaling
        self.lora_dropout = 0.1
        
        # ControlNet Config
        self.controlnet_conditioning_scale = 0.8
        
        # Training Config
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.num_epochs = 100
        self.gradient_accumulation_steps = 4
        self.mixed_precision = "fp16"
        self.save_steps = 500
        self.logging_steps = 100