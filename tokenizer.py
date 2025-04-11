"""
Byte Pair Encoding (BPE) Tokenizer

This module implements a BPE tokenizer for transformers, which converts text into
sequences of integer tokens. BPE works by iteratively merging the most frequent
pairs of bytes or characters in the text, creating a vocabulary of subword units.

This implementation trains its own BPE vocabulary from scratch on provided text data,
rather than using pre-trained vocabularies.

This implementation is inspired by the tokenization approach used in GPT-2 and 
https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py but has been rewritten
and restructured for clarity and integration with our repository.
"""

import os
import json
import regex as re
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import torch
from torch import Tensor
from jaxtyping import Int
from transformers import AutoTokenizer
from config import TransformerConfig

# -----------------------------------------------------------------------------

def create_byte_mapping() -> Dict[int, str]:
    """
    Creates a mapping from bytes to unicode characters.
    
    This function creates a one-to-one mapping between bytes (0-255) and unicode 
    characters. Characters that display well are kept as-is, while others are 
    mapped to characters in a higher unicode range to ensure they display properly.
    
    Returns:
        Dict[int, str]: A dictionary mapping byte values to unicode characters
    """
    # Characters that display well in their original form
    standard_chars = list(range(ord("!"), ord("~")+1)) + \
                     list(range(ord("¡"), ord("¬")+1)) + \
                     list(range(ord("®"), ord("ÿ")+1))
    
    target_chars = standard_chars.copy()
    
    # Map problematic bytes to higher unicode range
    offset = 0
    for byte in range(256):
        if byte not in standard_chars:
            standard_chars.append(byte)
            target_chars.append(2**8 + offset)
            offset += 1
    
    # Convert to characters and create mapping
    target_chars = [chr(n) for n in target_chars]
    byte_to_unicode = dict(zip(standard_chars, target_chars))
    
    return byte_to_unicode

def extract_bigrams(token: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """
    Extracts all adjacent character pairs from a token.
    
    Args:
        token: A tuple of characters representing a token
        
    Returns:
        A set of all adjacent character pairs in the token
    """
    bigrams = set()
    for i in range(len(token) - 1):
        bigrams.add((token[i], token[i+1]))
    return bigrams

def count_token_frequencies(texts: List[str], tokenization_pattern: re.Pattern) -> Dict[str, int]:
    """
    Count the frequency of each token in the texts after initial tokenization.
    
    Args:
        texts: List of text strings to analyze
        tokenization_pattern: Regex pattern for initial tokenization
        
    Returns:
        Dictionary mapping tokens to their frequencies
    """
    token_freqs = Counter()
    for text in texts:
        tokens = re.findall(tokenization_pattern, text)
        token_freqs.update(tokens)
    return token_freqs

def train_bpe_vocabulary(
    texts: List[str], 
    vocabulary_size: int = 50000, 
    min_frequency: int = 2
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train a BPE vocabulary from scratch on the provided texts.
    
    Args:
        texts: List of text strings to train on
        vocabulary_size: Target vocabulary size (including base characters)
        min_frequency: Minimum frequency for a token to be considered
        
    Returns:
        Tuple of (token_to_id mapping, merge_list)
    """
    # Create byte-to-unicode mapping
    byte_to_unicode = create_byte_mapping()
    
    # Regex pattern for initial tokenization
    tokenization_pattern = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    
    # Count token frequencies
    token_freqs = count_token_frequencies(texts, tokenization_pattern)
    
    # Initialize vocabulary with individual characters
    vocabulary = set()
    for token, freq in token_freqs.items():
        if freq < min_frequency:
            continue
        # Convert token to bytes and then to unicode
        byte_encoded = token.encode('utf-8')
        unicode_chars = ''.join(byte_to_unicode[b] for b in byte_encoded)
        # Add each character to vocabulary
        for char in unicode_chars:
            vocabulary.add(char)
    
    # Initialize merge operations list
    merges = []
    
    # Initialize token-to-parts mapping
    token_to_parts = {}
    for token, freq in token_freqs.items():
        if freq < min_frequency:
            continue
        # Convert token to bytes and then to unicode
        byte_encoded = token.encode('utf-8')
        unicode_chars = ''.join(byte_to_unicode[b] for b in byte_encoded)
        # Initialize as list of individual characters
        token_to_parts[token] = list(unicode_chars)
    
    # Iteratively find and apply the best merge
    while len(vocabulary) < vocabulary_size:
        # Count pair frequencies across all tokens
        pair_freqs = Counter()
        for token, parts in token_to_parts.items():
            token_freq = token_freqs[token]
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                pair_freqs[pair] += token_freq
        
        if not pair_freqs:
            break
        
        # Find the most frequent pair
        best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
        
        # Create new merged token
        new_token = ''.join(best_pair)
        vocabulary.add(new_token)
        
        # Add to merges list
        merges.append(best_pair)
        
        # Apply the merge to all tokens
        for token in list(token_to_parts.keys()):
            parts = token_to_parts[token]
            
            # Find and merge all occurrences of the pair
            i = 0
            while i < len(parts) - 1:
                if (parts[i], parts[i+1]) == best_pair:
                    parts = parts[:i] + [new_token] + parts[i+2:]
                    i = 0  # Start over to catch overlapping pairs
                else:
                    i += 1
            
            token_to_parts[token] = parts
        
        print(f"Vocabulary size: {len(vocabulary)}/{vocabulary_size}, Latest merge: {best_pair} -> {new_token}")
        
        if len(vocabulary) >= vocabulary_size:
            break
    
    # Create token_to_id mapping
    token_to_id = {token: i for i, token in enumerate(sorted(vocabulary))}
    
    # Add special tokens
    special_tokens = ["<|endoftext|>", "<|padding|>"]
    for token in special_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    
    return token_to_id, merges

def save_vocabulary(
    token_to_id: Dict[str, int], 
    merges: List[Tuple[str, str]], 
    vocabulary_dir: str = "vocabulary"
) -> None:
    """
    Save the trained vocabulary to disk.
    
    Args:
        token_to_id: Mapping from tokens to IDs
        merges: List of merge operations
        vocab_dir: Directory to save vocabulary files
    """
    os.makedirs(vocabulary_dir, exist_ok=True)
    
    # Save token_to_id mapping
    with open(os.path.join(vocabulary_dir, "encoder.json"), "w") as f:
        json.dump(token_to_id, f, ensure_ascii=False)
    
    # Save merges
    with open(os.path.join(vocabulary_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("# BPE merges\n")
        for first, second in merges:
            f.write(f"{first} {second}\n")

def load_vocabulary(vocabulary_dir: str = "vocabulary") -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Load a previously trained vocabulary from disk.
    
    Args:
        vocabulary_dir: Directory containing vocabulary files
        
    Returns:
        Tuple of (token_to_id mapping, merge_list)
    """
    # Load token_to_id mapping
    with open(os.path.join(vocabulary_dir, "encoder.json"), "r") as f:
        token_to_id = json.load(f)
    
    # Load merges
    merges = []
    with open(os.path.join(vocabulary_dir, "merges.txt"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            first, second = line.split()
            merges.append((first, second))
    
    return token_to_id, merges

class BPEEncoder:
    """
    Handles the encoding and decoding of text using Byte Pair Encoding.
    """
    def __init__(self, token_to_id: Dict[str, int], merge_list: List[Tuple[str, str]]):
        # Create byte-level encoders/decoders
        self.byte_to_unicode = create_byte_mapping()
        self.unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}
        
        # Token encoders/decoders
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # BPE merge priorities
        self.merge_priorities = dict(zip(merge_list, range(len(merge_list))))
        
        # Regex pattern for initial tokenization
        # Handles contractions, letters, numbers, symbols, and whitespace
        self.tokenization_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Cache for previously processed tokens
        self.bpe_cache = {}

    def apply_bpe_merges(self, token: str) -> str:
        """
        Applies BPE merges to a token according to the learned merge priorities.
        
        Args:
            token: A string representing a single token after byte encoding
            
        Returns:
            The token after applying all relevant BPE merges
        """
        # Check cache first for efficiency
        if token in self.bpe_cache:
            return self.bpe_cache[token]

        # Convert token to tuple of characters
        characters = tuple(token)
        
        # If token is a single character, no merges needed
        if len(characters) <= 1:
            return token

        # Get all character pairs in the token
        pairs = extract_bigrams(characters)
        
        # Iteratively apply merges until no more can be applied
        while True:
            # Find the pair with the highest priority (lowest rank)
            best_pair = min(
                pairs, 
                key=lambda pair: self.merge_priorities.get(pair, float('inf'))
            )
            
            # If no more merges can be applied, we're done
            if best_pair not in self.merge_priorities:
                break
                
            first, second = best_pair
            
            # Apply the merge to all occurrences of the pair
            new_characters = []
            i = 0
            while i < len(characters):
                # Find the next occurrence of first character
                if i < len(characters) - 1 and characters[i] == first and characters[i+1] == second:
                    # Merge the pair
                    new_characters.append(first + second)
                    i += 2
                else:
                    # Keep the character as is
                    new_characters.append(characters[i])
                    i += 1
            
            # Update the characters and pairs
            characters = tuple(new_characters)
            if len(characters) == 1:
                break
                
            pairs = extract_bigrams(characters)
        
        # Join the merged characters with spaces
        result = ' '.join(characters)
        
        # Cache the result
        self.bpe_cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token IDs
        """
        token_ids = []
        
        # Split text into tokens using regex
        raw_tokens = re.findall(self.tokenization_pattern, text)
        
        # Process each token
        for token in raw_tokens:
            # Convert to bytes and then to unicode representation
            byte_encoded = token.encode('utf-8')
            unicode_chars = ''.join(self.byte_to_unicode[b] for b in byte_encoded)
            
            # Apply BPE merges
            merged_token = self.apply_bpe_merges(unicode_chars).split(' ')
            
            # Convert to token IDs
            ids = [self.token_to_id.get(bpe_token, self.token_to_id.get("<|unk|>", 0)) 
                  for bpe_token in merged_token]
            token_ids.extend(ids)
            
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        
        Args:
            token_ids: A list of token IDs to decode
            
        Returns:
            The decoded text
        """
        # Convert IDs to tokens
        tokens = [self.id_to_token.get(id, "") for id in token_ids]
        
        # Join tokens and convert from unicode back to bytes
        text_unicode = ''.join(tokens)
        
        # Convert unicode characters back to bytes
        byte_sequence = bytearray()
        for char in text_unicode:
            if char in self.unicode_to_byte:
                byte_sequence.append(self.unicode_to_byte[char])
        
        # Decode bytes to UTF-8 string
        text = byte_sequence.decode('utf-8', errors='replace')
        
        return text

    def encode_with_details(self, text: str) -> Dict[str, Any]:
        """
        Encodes text and returns detailed information about the encoding process.
        Useful for debugging and understanding the tokenization process.
        
        Args:
            text: The input text to encode
            
        Returns:
            A dictionary containing the encoded IDs and intermediate steps
        """
        token_ids = []
        token_details = []
        raw_tokens = re.findall(self.tokenization_pattern, text)
        
        for token in raw_tokens:
            byte_encoded = token.encode('utf-8')
            unicode_chars = ''.join(self.byte_to_unicode[b] for b in byte_encoded)
            merged_token = self.apply_bpe_merges(unicode_chars).split(' ')
            ids = [self.token_to_id.get(bpe_token, self.token_to_id.get("<|unk|>", 0)) 
                  for bpe_token in merged_token]
            
            token_ids.extend(ids)
            token_details.append({
                'original': token,
                'bytes': byte_encoded,
                'unicode': unicode_chars,
                'merged': merged_token,
                'ids': ids,
            })
        
        return {
            'token_ids': token_ids,
            'raw_tokens': raw_tokens,
            'token_details': token_details,
        }

# -----------------------------------------------------------------------------

class Tokenizer:
    """
    PyTorch-compatible tokenizer that wraps the BPE encoder.
    
    This class provides a simple interface for encoding text into token IDs
    and decoding token IDs back into text, with PyTorch tensor support.
    """

    def __init__(
        self,config: Optional[TransformerConfig] = None, 
        vocab_dir: str = "vocabulary", 
        use_pretrained: bool = False, 
        pretrained_model_name: str = "gpt2"
    ):
        """
        Initialize the tokenizer.
        
        Args:
            config: Optional transformer configuration
            vocab_dir: Directory containing vocabulary files
            use_pretrained: Whether to use a pre-trained tokenizer from Hugging Face
            pretrained_model_name: Name of the pre-trained model to use (if use_pretrained is True)
        """
        self.config = config
        self.pad_token = None
        
        if use_pretrained:
            try:   
                self.encoder = AutoTokenizer.from_pretrained(pretrained_model_name)
                self.pad_token = self.encoder.pad_token
                self.add_special_tokens = self.encoder.add_special_tokens
                self.eos_token = self.encoder.eos_token
                self.bos_token = self.encoder.bos_token
                # TODO: This is a hack, need to fix this
                print(f"Loaded pre-trained tokenizer: {pretrained_model_name}")
            except Exception as e:
                print(f"Error loading pre-trained tokenizer: {e}")
                print("Falling back to local vocabulary.")
                use_pretrained = False
        else:
            # Check if vocabulary exists
            if os.path.exists(vocab_dir) and os.path.isfile(os.path.join(vocab_dir, "encoder.json")):
                # Load existing vocabulary
                token_to_id, merges = load_vocabulary(vocab_dir)
                self.encoder = BPEEncoder(token_to_id, merges)
                self.hf_tokenizer = None
                print(f"Loaded vocabulary with {len(token_to_id)} tokens from {vocab_dir}")
            else:
                # No vocabulary found, will need to train one
                self.encoder = None
                self.hf_tokenizer = None
                print(f"No vocabulary found at {vocab_dir}. Use train_vocabulary() to create one.")

    def train_vocabulary(
        self, 
        texts: List[str], 
        vocabulary_size: int = 50000, 
        min_frequency: int = 2, 
        vocabulary_dir: str = "vocabulary"
    ):
        """
        Train a BPE vocabulary on the provided texts.
        
        Args:
            texts: List of text strings to train on
            vocabulary_size: Target vocabulary size
            min_frequency: Minimum frequency for a token to be considered
            vocabulary_dir: Directory to save vocabulary files
        """
        print(f"Training BPE vocabulary on {len(texts)} texts...")
        token_to_id, merges = train_bpe_vocabulary(texts, vocabulary_size, min_frequency)
        
        # Save vocabulary
        save_vocabulary(token_to_id, merges, vocabulary_dir)
        
        # Initialize encoder with trained vocabulary
        self.encoder = BPEEncoder(token_to_id, merges)
        print(f"Vocabulary training complete. Created {len(token_to_id)} tokens.")

    def __call__(self, text: str, return_tensors: str = 'pt') -> Union[List[int], Tensor]:
        """
        Tokenize text into token IDs.
        
        Args:
            text: The text to tokenize
            return_tensors: Output format ('pt' for PyTorch tensors, 'list' for Python lists)
            
        Returns:
            Token IDs as a PyTorch tensor or list
        """
        assert isinstance(text, str), "Input must be a string"
        assert self.encoder is not None, "Tokenizer not initialized. Train or load a vocabulary first."
        
        # Encode text to token IDs
        token_ids = self.encoder.encode(text)
        
        # Return as tensor or list based on return_tensors
        if return_tensors == 'pt':
            return torch.tensor([token_ids], dtype=torch.long)
        elif return_tensors == 'list':
            return token_ids
        else:
            raise ValueError(f"Unsupported return_tensors value: {return_tensors}")

    def decode(self, token_ids: Union[List[int], Tensor]) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: Token IDs as a list or tensor
            
        Returns:
            The decoded text
        """
        assert self.encoder is not None, "Tokenizer not initialized. Train or load a vocabulary first."
        
        # Take care of integer inputs
        if token_ids.dim() == 0:
            token_ids = token_ids.view(1, 1)
        
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
        
        return self.encoder.decode(token_ids)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: The text to encode
            
        Returns:
            Token IDs as a list
        """
        return self.__call__(text, return_tensors='pt')

    def encode_batch(self, texts: List[str]) -> Int[Tensor, "B S"]:
        """
        Encode a batch of texts into token IDs.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Batched token IDs as a PyTorch tensor
        """
        assert self.encoder is not None, "Tokenizer not initialized. Train or load a vocabulary first."
        
        # Encode each text
        batch_ids = [self.encoder.encode(text) for text in texts]
        
        # Determine max sequence length
        max_length = max(len(ids) for ids in batch_ids)
        
        # Pad sequences to the same length
        padded_ids = [ids + [0] * (max_length - len(ids)) for ids in batch_ids]
        
        # Convert to tensor
        return torch.tensor(padded_ids, dtype=torch.long)


if __name__ == '__main__':
    # Example usage
    tokenizer = Tokenizer()
    
    # Sample texts for training
    sample_texts = [
        "Hello world! This is a test of the tokenizer.",
        "It handles emojis too: 🚀✨",
        "BPE tokenization works by merging frequent character pairs.",
        "This helps with handling rare words and subword units.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language that lets you work quickly and integrate systems effectively.",
    ]
    
    # Train vocabulary on sample texts
    tokenizer.train_vocabulary(sample_texts, vocabulary_size=1000, min_frequency=1)
    
    # Test encoding and decoding
    sample_text = "Hello world! This is a test of the tokenizer. It handles emojis too: 🚀✨"
    tokens = tokenizer(sample_text)
    
    print(f"Original text: {sample_text}")
    print(f"Token IDs: {tokens[0]}")
    print(f"Decoded text: {tokenizer.decode(tokens[0])}")
    
    # Show detailed encoding process
    details = tokenizer.encoder.encode_with_details(sample_text)
    print("\nDetailed encoding process:")
    for i, token_detail in enumerate(details['token_details']):
        print(f"Token {i+1}: {token_detail['original']}")
        print(f"  → Unicode: {token_detail['unicode']}")
        print(f"  → Merged: {token_detail['merged']}")
        print(f"  → IDs: {token_detail['ids']}")
