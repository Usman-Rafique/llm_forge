# Training CLI Arguments

To train a model, use the following command:

```
python -m llm_forge.train [MODEL_NAME] [OPTIONS]
```

## Required Arguments

- `MODEL_NAME`: Name of the model configuration to use (e.g., gpt2_medium)

## Optional Arguments

- `--config_file`: Path to the configuration file (default: configs.yaml)
- `--ds_path`: Path to the dataset (default: datasets/smol_lm_corpus/fineweb_edu)

## Configuration File

The configuration file (YAML) should contain the following sections:

- `global`: Global training settings
- `models`: Model-specific configurations

### Global Configuration Options

- `tokenizer_name`: Name of the tokenizer to use
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `optimizer`: Optimizer to use (e.g., Adam)
- `learning_rate`: Learning rate for the optimizer
- `weight_decay`: Weight decay for the optimizer
- `save_dir`: Directory to save model checkpoints
- `log_dir`: Directory to save TensorBoard logs
- `eval_frequency`: Frequency of evaluation steps
- `num_val_batches`: Number of validation batches to use
- `mixed_precision`: Whether to use mixed precision training (true/false)

### Model-specific Configuration Options

- `context_length`: Maximum context length for the model
- `emb_dim`: Embedding dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `drop_rate`: Dropout rate
- `qkv_bias`: Whether to use bias in query, key, and value projections (true/false)

For more details on each argument, run:

```
python -m llm_forge.train --help
```
