# Text Generation CLI Guide

This guide explains how to use the text generation CLI tool in the LLM Forge project.

## Basic Usage

To generate text using a trained model, use the following command:

```
python -m llm_forge.generate_text [MODEL_NAME] [OPTIONS]
```

## Arguments

### Required Arguments

- `MODEL_NAME`: Name of the model configuration to use (e.g., gpt2_medium)

### Optional Arguments

- `--config_file`: Path to the configuration file (default: configs/gpt2.yaml)
- `--checkpoint`: Path to a specific checkpoint file to load (default: None)
- `--max_length`: Maximum length of generated text (default: 100)
- `--start_text`: Text to start generation (default: "Fruits are good for you because")

## Using Pre-trained Checkpoints

1. Download a checkpoint from the [LLM Forge Checkpoints Google Drive folder](https://drive.google.com/drive/folders/1qFAAdg4SYwNkT-jHdES46OVJRHpYvry8?usp=sharing).
2. Place the checkpoint in the `checkpoints` directory of your project.
3. Use the `--checkpoint` argument to specify the checkpoint file:

```
python -m llm_forge.generate_text gpt2_medium --config_file configs/gpt2.yaml --checkpoint checkpoints/model_gpt2_medium.pth
```

## Configuration File

The configuration file (`--config_file`) should have the same structure as the one used for training, containing `global` and `models` sections. This file specifies model architecture, tokenizer, and other settings.

## Examples

1. Generate text using the default settings:
   ```
   python -m llm_forge.generate_text gpt2_medium
   ```

2. Generate text with a specific checkpoint and custom start text:
   ```
   python -m llm_forge.generate_text gpt2_medium --checkpoint checkpoints/model_gpt2_medium.pth --start_text "The future of AI is"
   ```

3. Generate a longer text sequence:
   ```
   python -m llm_forge.generate_text gpt2_medium --max_length 200
   ```

## Troubleshooting

- If you encounter a "checkpoint not found" error, ensure that the checkpoint file is in the correct directory and that the path is specified correctly.
- If you get a "CUDA out of memory" error, try reducing the `max_length` or use a smaller model.

For more details on each argument, run:

```
python -m llm_forge.generate_text --help
```
