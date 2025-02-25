# scripts/train.py
import sys
import os
import torch
from torch.utils.data import random_split, DataLoader
from accelerate import Accelerator
import wandb
from tqdm.auto import tqdm
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorization.datasets.colorization import ColorizationDataset
from colorization.models.lora import ColorizationModel
from colorization.trainers.trainer import ColorizationTrainer
from colorization.config.model_config import ModelConfig
from colorization.config.data_config import DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_wandb(config):
    """Khởi tạo Weights & Biases logging"""
    wandb.init(
        project="image-colorization",
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.num_epochs,
            "model": config.pretrained_model_name,
        }
    )

def check_gpu():
    """Kiểm tra và in thông tin GPU"""
    if torch.cuda.is_available():
        logger.info("\nGPU Information:")
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        return True
    else:
        logger.warning("No GPU available, using CPU!")
        return False

def check_sample_batch(dataset, num_samples=2):
    """Kiểm tra một số mẫu từ dataset"""
    logger.info("\nChecking sample batches:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        logger.info(f"\nSample {i}:")
        logger.info(f"Image shapes - Source: {sample['hint'].shape}, Target: {sample['jpg'].shape}")
        logger.info(f"Value ranges - Source: [{sample['hint'].min():.2f}, {sample['hint'].max():.2f}]")
        logger.info(f"Prompt: {sample['txt']}")
        
        if torch.isnan(sample['hint']).any() or torch.isnan(sample['jpg']).any():
            logger.error("WARNING: NaN values detected in tensors!")

def setup_output_dir():
    """Tạo thư mục để lưu checkpoints và logs"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    return output_dir

def main():
    # output directory
    output_dir = setup_output_dir()
    logger.info(f"Output directory: {output_dir}")

    # Load configs
    model_config = ModelConfig()
    data_config = DataConfig()
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Accelerator
    accelerator = Accelerator(
        mixed_precision=model_config.mixed_precision,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps
    )
    
    # Setup dataset
    dataset = ColorizationDataset(
        data_root=data_config.data_root,
        image_size=data_config.image_size
    )
    
    # Checking samples
    check_sample_batch(dataset)
    
    # Split dataset
    train_size = int(len(dataset) * data_config.train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"\nDataset split:")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=has_gpu
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=has_gpu
    )
    
    # Initialize model
    model = ColorizationModel(model_config)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=model_config.num_epochs
    )
    
    # Initialize wandb
    setup_wandb(model_config)
    
    # Prepare everything with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Initialize trainer
    trainer = ColorizationTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_config,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        output_dir=output_dir
    )
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Cleanup
        wandb.finish()
        
        # Save final model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            os.path.join(output_dir, 'final_model')
        )
        logger.info("Training completed. Model saved.")

if __name__ == "__main__":
    main()