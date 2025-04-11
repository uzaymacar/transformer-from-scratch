import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
import einops
from config import TransformerConfig

class Embedding(nn.Module):
    """This layer is used to embed the input tokens into a dense vector space."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(torch.empty((config.vocabulary_size, config.embedding_dimension)))
        nn.init.normal_(self.W_E, std=self.config.weight_initialization_range)

    def forward(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T E"]:
        x_embedded = self.W_E[tokens, ...]                       
        return x_embedded
    
class PositionalEmbedding(nn.Module):
    """This layer is used to add positional information to the input tokens."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.W_P = nn.Parameter(torch.empty((config.context_length, config.embedding_dimension)))
        nn.init.normal_(self.W_P, std=self.config.weight_initialization_range)

    def forward(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T E"]:
        batch_size, sequence_length = tokens.shape
        x_positional = einops.repeat(self.W_P[:sequence_length], "T E -> B T E", B=batch_size)
        return x_positional

class LayerNorm(nn.Module):
    """This layer is used to normalize outputs of the previous layer before applying the next layer."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.w = nn.Parameter(torch.ones(config.embedding_dimension))
        self.b = nn.Parameter(torch.zeros(config.embedding_dimension))                                        

    def forward(self, residual: Float[Tensor, "B T E"]) -> Float[Tensor, "B T E"]:
        mean: Float[Tensor, "B T 1"] = torch.mean(residual, dim=-1, keepdim=True)   
        variance: Float[Tensor, "B T 1"] = torch.var(residual, dim=-1, unbiased=False, keepdim=True)
        square_root_variance: Float[Tensor, "B S 1"] = torch.sqrt(variance + self.config.layer_normalization_epsilon)
        normalized_residual: Float[Tensor, "B T M"] = (residual - mean) / square_root_variance
        affine_residual: Float[Tensor, "B T M"] = self.w * normalized_residual + self.b
        return affine_residual
    
class Attention(nn.Module):
    """This layer is used to calculate the attention scores and apply the attention mechanism."""
    IGNORE: Float[Tensor, ""]
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        qkv_weight_shape = (config.num_attention_heads, config.embedding_dimension, config.attention_head_dimension)
        qkv_bias_shape = (config.num_attention_heads, config.attention_head_dimension)
        o_weight_shape = (config.num_attention_heads, config.attention_head_dimension, config.embedding_dimension)
        o_bias_shape = (config.embedding_dimension)
        self.W_Q = nn.Parameter(torch.empty(qkv_weight_shape))
        self.W_K = nn.Parameter(torch.empty(qkv_weight_shape))
        self.W_V = nn.Parameter(torch.empty(qkv_weight_shape))
        self.W_O = nn.Parameter(torch.empty(o_weight_shape))
        self.b_Q = nn.Parameter(torch.zeros(qkv_bias_shape))
        self.b_K = nn.Parameter(torch.zeros(qkv_bias_shape))
        self.b_V = nn.Parameter(torch.zeros(qkv_bias_shape))
        self.b_O = nn.Parameter(torch.zeros(o_bias_shape))
        nn.init.normal_(self.W_Q, std=self.config.weight_initialization_range)
        nn.init.normal_(self.W_K, std=self.config.weight_initialization_range)
        nn.init.normal_(self.W_V, std=self.config.weight_initialization_range)
        nn.init.normal_(self.W_O, std=self.config.weight_initialization_range)
        self.register_buffer("IGNORE", torch.tensor(float("-inf"), dtype=torch.float32, device=self.config.device))

    def forward(self, normalized_residual_pre: Float[Tensor, "B T E"]) -> Float[Tensor, "B T E"]:
        # Calculate query (q), key (k), and value (v) vectors
        q = einops.einsum(normalized_residual_pre, self.W_Q, "B T E, N E H -> B T N H") + self.b_Q
        k = einops.einsum(normalized_residual_pre, self.W_K, "B T E, N E H -> B T N H") + self.b_K
        v = einops.einsum(normalized_residual_pre, self.W_V, "B T E, N E H -> B T N H") + self.b_V
        
        # Calculate attention scores
        attention_scores = einops.einsum(q, k, "B T_Q N H, B T_K N H -> B N T_Q T_K")
        
        # Scale and mask the attention scores
        attention_scores /= self.config.attention_head_dimension ** 0.5
        attention_scores = self.apply_causal_mask(attention_scores)
        
        # Apply softmax to get attention probabilities
        attention_probabilities = torch.softmax(attention_scores, dim=-1)
        
        # Take weighted sum of value vectors according to attention probabilities
        z = einops.einsum(v, attention_probabilities, "B T_K N H, B N T_Q T_K -> B T_Q N H")
        
        # Map weighted sum back to model dimension
        z_mapped = einops.einsum(z, self.W_O, "B T_Q N H, N H E -> B T_Q N E")
        
        # Sum over heads and add bias to get the attention output
        attention_output = einops.einsum(z_mapped, "B T_Q N E -> B T_Q E") + self.b_O    
        return attention_output

    def apply_causal_mask(self, attention_scores: Float[Tensor, "B N T_Q T_K"]) -> Float[Tensor, "B N T_Q T_K"]:
        # Generate a mask with 1s on the upper triangular part and 0s on the lower triangular part
        mask = torch.triu(torch.ones(attention_scores.shape, device=self.config.device), diagonal=1).bool()
            
        # Apply the mask to the attention scores
        masked_attention_scores = torch.where(mask, self.IGNORE, attention_scores)
        return masked_attention_scores
    
class MLP(nn.Module):
    """This layer is used to calculate the feed-forward network. MLP stands for Multi-Layer Perceptron."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.W_in = nn.Parameter(torch.empty((config.embedding_dimension, config.mlp_dimension)))
        self.W_out = nn.Parameter(torch.empty((config.mlp_dimension, config.embedding_dimension)))
        self.b_in = nn.Parameter(torch.zeros((config.mlp_dimension)))
        self.b_out = nn.Parameter(torch.zeros((config.embedding_dimension)))
        nn.init.normal_(self.W_in, std=self.config.weight_initialization_range)
        nn.init.normal_(self.W_out, std=self.config.weight_initialization_range)
        self.gelu = GELU()

    def forward(self, normalized_residual_mid: Float[Tensor, "B T E"]) -> Float[Tensor, "B T E"]:
        pre = einops.einsum(normalized_residual_mid, self.W_in, "B T E, E M -> B T M") + self.b_in
        post = self.gelu(pre)
        mlp_output = einops.einsum(post, self.W_out, "B T M, M E -> B T E") + self.b_out
        return mlp_output
    
class Unembedding(nn.Module):
    """This layer is used to unembed the output of the final layer into the vocabulary space."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.W_U = nn.Parameter(torch.empty((config.embedding_dimension, config.vocabulary_size)))
        nn.init.normal_(self.W_U, std=self.config.weight_initialization_range)
        self.b_U = nn.Parameter(torch.zeros((config.vocabulary_size), requires_grad=False))

    def forward(self, normalized_residual_final: Float[Tensor, "B T E"]) -> Float[Tensor, "B T V"]:
        y_unembedded = einops.einsum(normalized_residual_final, self.W_U, "B T E, E V -> B T V") + self.b_U
        return y_unembedded
    
class GELU(nn.Module):
    """
    This is the (new) GELU activation function used by GPT-2. 
    Although a GELU is implemented in [Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html),
    this implementation is different and comes from the Google BERT repository.
    """
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))