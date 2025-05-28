import argparse
from utils import tokenize_and_concatenate
from config import TransformerConfig, TrainingArgs
from model import Transformer
from trainer import TransformerTrainer
from sampler import TransformerSampler
from tokenizer import Tokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model from scratch")
    
    # Dataset arguments
    parser.add_argument("-ds", "--dataset", type=str, default="NeelNanda/pile-10k", help="Dataset to use for training")
    parser.add_argument("-dss", "--dataset-split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("-dc", "--dataset-column", type=str, default="text", help="Column name containing text data")
    parser.add_argument("-ts", "--test-size", type=int, default=1000, help="Number of samples for test set")
    
    # Tokenizer arguments
    parser.add_argument("-ut", "--use-pretrained-tokenizer", action="store_true", default=True, help="Use pretrained tokenizer")
    parser.add_argument("-ptn", "--pretrained-tokenizer-name", type=str, default="gpt2", help="Name of pretrained tokenizer")
    
    # Transformer arguments
    parser.add_argument("-dim", "--embedding-dimension", type=int, default=768, help="Embedding dimension")
    parser.add_argument("-ed", "--enable-debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("-len", "--layer-norm-epsilon", type=float, default=1e-5, help="Layer normalization epsilon")
    parser.add_argument("-vs", "--vocabulary-size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("-wir", "--weight-init-range", type=float, default=0.02, help="Weight initialization range")
    parser.add_argument("-cl", "--context-length", type=int, default=256, help="Context length")
    parser.add_argument("-ad", "--attention-head-dimension", type=int, default=64, help="Attention head dimension")
    parser.add_argument("-md", "--mlp-dimension", type=int, default=2048, help="MLP dimension")
    parser.add_argument("-na", "--num-attention-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("-ntl", "--num-transformer-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device to use")
    
    # Training arguments
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("-bs", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-ne", "--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("-mse", "--max-steps-per-epoch", type=int, default=200, help="Maximum steps per epoch")
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("-nw", "--num-workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("-np", "--num-processes", type=int, default=4, help="Number of processes for tokenization")
    parser.add_argument("-wp", "--wandb-project", type=str, default="transformer-from-scratch", help="Weights & Biases project name")
    parser.add_argument("-wn", "--wandb-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("-tsf", "--text-sample-frequency", type=int, default=200, help="Text sampling frequency")
    parser.add_argument("-tlf", "--table-log-frequency", type=int, default=200, help="Table logging frequency")
    
    # Sampling arguments
    parser.add_argument("-sp", "--sample-prompt", type=str, default="What is the meaning of life?", help="Prompt for sampling")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-tp", "--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("-m", "--max-tokens-generated", type=int, default=32, help="Maximum tokens to generate")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the config and training arguments from parsed arguments
    config = TransformerConfig(
        enable_debug=args.enable_debug,
        embedding_dimension=args.embedding_dimension,
        layer_normalization_epsilon=args.layer_norm_epsilon,
        vocabulary_size=args.vocabulary_size,
        weight_initialization_range=args.weight_init_range,
        context_length=args.context_length,
        attention_head_dimension=args.attention_head_dimension,
        mlp_dimension=args.mlp_dimension,
        num_attention_heads=args.num_attention_heads,
        num_transformer_layers=args.num_transformer_layers,
    )
    
    # Set device if specified
    if args.device is not None:
        config.device = args.device
    
    training_args = TrainingArgs(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        num_processes=args.num_processes,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        text_sample_frequency=args.text_sample_frequency,
        table_log_frequency=args.table_log_frequency,
    )

    # Initialize the tokenizer and model (move to GPU if available)
    tokenizer = Tokenizer(
        use_pretrained=args.use_pretrained_tokenizer, 
        pretrained_model_name=args.pretrained_tokenizer_name
    ).encoder
    model = Transformer(config)
    model = model.to(config.device)
    print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

    # Load the dataset, tokenize it, and split into train and test
    dataset = load_dataset(args.dataset, split=args.dataset_split)
    try: 
        dataset = dataset.remove_columns("meta")
    except:
        pass
    
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=config.context_length,
        column_name=args.dataset_column,
        add_bos_token=True,
        num_processes=training_args.num_processes,
    )
    dataset_dict = tokenized_dataset.train_test_split(test_size=args.test_size)

    def sampling_function(model: Transformer, prompt: str) -> str:
        sampler = TransformerSampler(model, tokenizer)
        output = sampler.sample(
            prompt, 
            temperature=args.temperature, 
            top_p=args.top_p,
            max_tokens_generated=args.max_tokens_generated
        )
        return output

    # Initialize the trainer
    trainer = TransformerTrainer(
        training_args, 
        model, 
        dataset_dict, 
        sample_prompt_list=[args.sample_prompt], 
        sampling_function=sampling_function
    )

    # Train the model
    trainer.train()

    # Sample from the model
    sampler = TransformerSampler(model, tokenizer)
    print(sampler.sample(args.sample_prompt))

if __name__ == "__main__":
    main()

