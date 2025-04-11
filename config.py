from dataclasses import dataclass
import torch

@dataclass
class TransformerConfig:
    embedding_dimension: int = 768 # E
    enable_debug: bool = True
    layer_normalization_epsilon: float = 1e-5
    vocabulary_size: int = 50257 # V
    weight_initialization_range: float = 0.02
    context_length: int = 1024 # T
    attention_head_dimension: int = 64 # H
    mlp_dimension: int = 3072 # M
    num_attention_heads: int = 12 # N
    num_transformer_layers: int = 12 # L
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
@dataclass
class TrainingArgs:
    learning_rate: float = 3e-4
    batch_size: int = 8
    num_epochs: int = 100
    max_steps_per_epoch: int = 200
    weight_decay: float = 1e-2
    num_workers: int = 0 # TODO: Adding more than 1 fails as it creates 4 runs on wandb - fix this
    num_processes: int = 4 # TODO: This is the number of processes to use for parallel tokenization - fix this
    wandb_project: str = "transformer-from-scratch"
    wandb_name: str | None = None
    text_sample_frequency: int = 200
    table_log_frequency: int = 200