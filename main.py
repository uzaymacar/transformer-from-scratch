from utils import tokenize_and_concatenate
from config import TransformerConfig, TrainingArgs
from model import Transformer
from trainer import TransformerTrainer
from sampler import TransformerSampler
from tokenizer import Tokenizer
from datasets import load_dataset

# Initialize the config and training arguments
config = TransformerConfig(
    enable_debug=False,
    embedding_dimension=768,
    attention_head_dimension=64,
    mlp_dimension=2048,
    num_attention_heads=6,
    num_transformer_layers=6,
    context_length=256,
)
args = TrainingArgs(
    learning_rate=5e-4,
    batch_size=32,
    num_epochs=100,
    max_steps_per_epoch=200,
    num_processes=4,
)

# Initialize the tokenizer and model (move to GPU if available)
tokenizer = Tokenizer(use_pretrained=True, pretrained_model_name="gpt2").encoder
model = Transformer(config)
model = model.to(config.device)
print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

# Load the dataset, tokenize it, and split into train and test
dataset = load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming=False,
    max_length=config.context_length,
    column_name="text",
    add_bos_token=True,
    num_processes=args.num_processes,
)
dataset_dict = tokenized_dataset.train_test_split(test_size=1000)

def sampling_function(model: Transformer, prompt: str) -> str:
    sampler = TransformerSampler(model, tokenizer)
    output = sampler.sample(prompt, temperature=0.7, top_p=0.95, max_tokens_generated=32)
    return output

# Initialize the trainer
trainer = TransformerTrainer(
    args, 
    model, 
    dataset_dict, 
    sample_prompt_list=["What is the meaning of life?"], 
    sampling_function=sampling_function
)

# Train the model
trainer.train()

# Sample from the model
sampler = TransformerSampler(model, tokenizer)
print(sampler.sample("What is the meaning of life?"))

