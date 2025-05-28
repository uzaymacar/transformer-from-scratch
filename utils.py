import numpy as np
import einops
from typing import Dict, List
from datasets import Dataset
import torch
from torch import Tensor
from jaxtyping import Float, Int
from tokenizer import Tokenizer

def get_log_probabilites(logits: Float[Tensor, "B T V"], tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T-1"]:
    """Get the log probabilities of the next token for each token in the sequence."""
    log_probabilities: Float[Tensor, "B T V"] = torch.log_softmax(logits, dim=-1)
    log_probabilities_for_tokens: Float[Tensor, "B T-1"] = log_probabilities[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probabilities_for_tokens

def keep_single_column(dataset: Dataset, col_name: str):
    """
    Delete all columns apart from a single column name from a HuggingFace dataset.
    Useful when we want to tokenize and mix together different strings.
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset

def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: Tokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_processes: int = 10,
) -> Dataset:
    """
    Helper function to tokenizer and concatenate a dataset of text. 
    This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them 
    into a 2D array of shape (_, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, 
    so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of 
    text of varying lengths (without, e.g. a lot of truncation or padding). Further, for models with absolute positional 
    encodings, this avoids privileging early tokens (e.g., news articles often begin with CNN, and models may learn to use 
    early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (Tokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.
        num_processes (int, optional): The number of processes to use for parallel tokenization. Defaults to 10.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        sequence_length = max_length - 1
    else:
        sequence_length = max_length

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)

        # Handle the case when full_text is empty
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        # Handle cases where num_tokens is less than seq_len
        if num_tokens < sequence_length:
            num_batches = 1
            
            # Pad tokens if necessary
            tokens = tokens[:sequence_length]
            if len(tokens) < sequence_length:
                padding_length = sequence_length - len(tokens)
                padding = np.full(padding_length, tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            num_batches = num_tokens // sequence_length
            # Drop the final tokens if not enough to make a full sequence
            tokens = tokens[: sequence_length * num_batches]

        tokens = einops.rearrange(tokens, "(B T) -> B T", B=num_batches, T=sequence_length)
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_processes if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    
    return tokenized_dataset