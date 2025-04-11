# Transformer from Scratch

A minimal implementation of a GPT-2 style transformer from scratch using PyTorch. Most of the code comes from my answers to the questions in [ARENA's Chapter 1.1](https://arena-chapter1-transformer-interp.streamlit.app/%5B1.1%5D_Transformer_from_Scratch).

There are other great repositories out there that implement a transformer from scratch, such as Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT).

## Setup

1. Install the requirements:

```bash
pip install -r requirements.txt
```

2. Change the parameters in `main.py` to your liking (more on that below)

3. Run `python main.py` to train the model and observe generated samples from the model

## Configuration

### Transformer Configuration

The `TransformerConfig` class in `config.py` contains the core architecture parameters:

- `embedding_dimension`: Size of token embeddings, $E$
- `vocabulary_size`: Number of tokens in vocabulary, $V$
- `context_length`: Maximum sequence length, $T$
- `attention_head_dimension`: Size of each attention head, $H$
- `num_attention_heads`: Number of attention heads, $N$
- `num_transformer_layers`: Number of transformer blocks, $L$
- `mlp_dimension`: Size of the MLP hidden layer, $M$

These can be tweaked in `main.py` to change the model architecture.

### Training Configuration

The `TrainingArgs` class in `config.py` contains the training parameters:

- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Number of sequences in a batch
- `num_epochs`: Number of epochs to train the model
- `max_steps_per_epoch`: Maximum number of steps per epoch
- `num_processes`: Number of processes to use for training

## Running the Model

To train and sample from the model:

```bash
python main.py
```

This will:
1. Initialize the model with the specified configuration
2. Load and tokenize the dataset (currently a [10k sample from the Pile dataset](https://huggingface.co/datasets/NeelNanda/pile-10k))
3. Train the model for the specified number of epochs
4. Generate a sample completion for `"What is the meaning of life?"`

## Implementation Notes

This implementation uses [einops](https://github.com/arogozhnikov/einops) to make tensor operations more readable. The naming conventions for dimensions are:

- $B$: Batch size
- $T$: Sequence length (tokens)
- $E$: Embedding dimension
- $V$: Vocabulary size
- $N$: Number of attention heads
- $H$: Attention head dimension
- $M$: MLP dimension
- $L$: Number of layers

For example, `"B T E, N E H -> B T N H"` means transforming a tensor of shape [batch size, sequence length, embedding dimension] and tensor of shape [number of heads, embedding dimension, attention head dimension] to a tensor of shape [batch size, sequence length, number of heads, attention head dimension].

