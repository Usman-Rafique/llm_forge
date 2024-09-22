import argparse
import os
import torch
import tiktoken
from models import GPTModel
import yaml


def load_config(config_file, model_name):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    global_config = config['global']
    model_config = config['models'][model_name]

    return global_config, model_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model")
    parser.add_argument(
        "model_name",
        help="Name of the model configuration to use (e.g., gpt2_medium)")
    parser.add_argument("--config_file",
                        default="configs.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--save_dir",
                        default="checkpoints",
                        help="Directory where the model checkpoint is saved")
    parser.add_argument("--max_length",
                        type=int,
                        default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--start_text",
                        type=str,
                        default="Fruits are good for you because",
                        help="Text to start generation")
    args = parser.parse_args()

    global_config, model_config = load_config(args.config_file, args.model_name)

    tokenizer = tiktoken.get_encoding(global_config['tokenizer_name'])
    model_config['vocab_size'] = tokenizer.n_vocab
    model_config['tokenizer_name'] = global_config['tokenizer_name']

    # Instantiate the model
    model = GPTModel(**model_config)

    # Load the checkpoint
    checkpoint_path = os.path.join(global_config['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model checkpoint loaded from {}".format(checkpoint_path))
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate text using the existing generate_text method
    generated_text = model.generate_text(args.start_text,
                                         max_length=args.max_length,
                                         device=device)
    print(f'\nGenerated text:\n{generated_text}')


if __name__ == "__main__":
    main()
