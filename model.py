from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from layers import Embedding, PositionalEmbedding, LayerNorm, Attention, MLP, Unembedding
from config import TransformerConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layer_norm_1 = LayerNorm(config)
        self.attention = Attention(config)
        self.layer_norm_2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, residual_pre: Float[Tensor, "B T E"]) -> Float[Tensor, "B T E"]:
        residual_mid = self.attention(self.layer_norm_1(residual_pre)) + residual_pre
        residual_post = self.mlp(self.layer_norm_2(residual_mid)) + residual_mid
        return residual_post

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.positional_embedding = PositionalEmbedding(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_transformer_layers)])
        self.layer_norm_final = LayerNorm(config)
        self.unembedding = Unembedding(config)
        
    def forward(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T V"]:
        residual = self.embedding(tokens) + self.positional_embedding(tokens)
        for transformer_block in self.transformer_blocks:
            residual = transformer_block(residual)
        logits = self.unembedding(self.layer_norm_final(residual))
        return logits