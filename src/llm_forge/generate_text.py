import argparse
import os
import torch
import tiktoken
from llm_forge.models.factory import ModelFactory
from llm_forge.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained GPT model")
    parser.add_argument(
        "model_name",
        help="Name of the model configuration to use (e.g., gpt2_medium)")
    parser.add_argument("--config_file",
                        default="gpt2",
                        help="Path to the configuration file or config name")
    parser.add_argument("--checkpoint",
                        help="Path to the specific checkpoint file to load")
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

    # Instantiate the model using ModelFactory
    model = ModelFactory.create_model('gpt', model_config)

    device = global_config['devices'][0]
    model.to(device)

    # Load the checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_name = f'model_{args.model_name}.pth'
        checkpoint_path = os.path.join(global_config['save_dir'],
                                       checkpoint_name)

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model checkpoint loaded from {checkpoint_path}")
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with untrained model.")
    else:
        print(
            f"Checkpoint not found at {checkpoint_path}. Continuing with untrained model."
        )

    # Generate text using the existing generate_text method
    generated_text = model.generate_text(args.start_text,
                                         max_length=args.max_length,
                                         device=device)
    print(f'\nGenerated text:\n{generated_text}')


if __name__ == "__main__":
    main()
