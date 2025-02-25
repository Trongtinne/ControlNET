# trainers/trainer.py
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

class ColorizationTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Prepare everything with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            
            with tqdm(total=len(self.train_loader)) as pbar:
                for step, batch in enumerate(self.train_loader):
                    with self.accelerator.accumulate(self.model):
                        # Forward pass
                        outputs = self.model(batch)
                        loss = outputs.loss
                        
                        # Backward pass
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        train_loss += loss.detach().item()
                        
                    if step % self.config.logging_steps == 0:
                        self.accelerator.print(
                            f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                        )
                    
                    if step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{epoch}-{step}")
                        
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
            
            # Validation
            self.validate()
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(batch)
                val_loss += outputs.loss.item()
        
        val_loss /= len(self.val_loader)
        self.accelerator.print(f"Validation Loss: {val_loss:.4f}")
        
    def save_checkpoint(self, name):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), f"checkpoints/{name}.pt")