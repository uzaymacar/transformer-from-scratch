from typing import Callable
import random
from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from datasets import DatasetDict
from model import Transformer
from utils import get_log_probabilites
from config import TrainingArgs

class TransformerTrainer:
    def __init__(self, 
        args: TrainingArgs, 
        model: Transformer, 
        dataset_dict: DatasetDict, 
        sample_prompt_list: list[str], 
        sampling_function: Callable[[Transformer, str], str]
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.device = model.config.device
        self.sample_prompt_list = sample_prompt_list
        self.sampling_function = sampling_function

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True
        )
        self.test_loader = DataLoader(
            dataset_dict["test"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.
        `batch` is a dictionary with the single key 'tokens'.
        """
        tokens = batch["tokens"].to(self.device)
        logits = self.model(tokens)
        loss = -get_log_probabilites(logits, tokens).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({ "Training Loss": loss }, step=self.step)
        
        return loss

    @torch.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            tokens = batch["tokens"].to(self.device)
            logits = self.model(tokens)[:, :-1] # NOTE: Last predicted token is redundant, only need n-1 tokens to predict n token sequence
            predicted_tokens = logits.argmax(dim=-1)
            total_correct += (predicted_tokens == tokens[:, 1:]).sum().item() # NOTE: Don't need the first token which would map to itself, we're shifting the sequence
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
            
        accuracy = total_correct / total_samples
        wandb.log({ "Accuracy": accuracy }, step=self.step)
            
        return accuracy

    def train(self):
        """
        Trains the model for `self.args.num_epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        # Initialize wandb and set the run name to the current date and time
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        wandb.run.name = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        wandb.run.save()
        
        accuracy = np.nan
        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.num_epochs)
        completions_list = []

        for epoch in range(self.args.num_epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                if accuracy is not np.nan:
                    progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss:.3f} | Accuracy: {accuracy:.3f}")
                else:
                    progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss:.3f}")
                
                if self.step % self.args.text_sample_frequency == 0:
                    text_completions = [self.sampling_function(self.model, prompt) for prompt in self.sample_prompt_list]
                    completions_list.append([epoch, self.step, *text_completions])
                    print("EXAMPLE COMPLETION | ", random.choice(text_completions))
                
                if self.step % self.args.table_log_frequency == 0:
                    wandb.log({
                        "Completions": wandb.Table(
                            data=completions_list, 
                            columns=["Epoch", "Step", *[f"Prompt {i + 1}" for i in range(len(self.sample_prompt_list))]]
                        ),
                    })
                
                # Check if we've reached the maximum steps for this epoch
                if i >= self.args.max_steps_per_epoch:
                    break
                    
            accuracy = self.evaluate()

        wandb.finish()

