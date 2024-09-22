# LLM Forge

This repo is my experimental playground for building and training practical LLMs with relatively limited compute resources. The goal is to train real LLMs on actual datasets, starting with pre-training from scratch and eventually moving towards instruction tuning with pre-trained models.

## Goals
1. **From Scratch**: No pre-trained models, no fine-tuning, no API.
2. **Limited Training Time**: ~ 24 hours.
3. **Limited Resources**: 2 GPUs, 24GB RAM each.
4. **Publicly Available Resources**: Using only publicly available datasets and tools.

## Dataset

The dataset used for training is the [FineWebEdu Deduplicated from SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). The dataset is a de-duplicated version of [FineWebEdu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and was filtered using a classifier to retain only high-quality educational content.

Please note that the dataset is much bigger, and we will probably not use all of it for training.

## Models
Models supported so far:

- [x] GPT-2 from Scratch

## Usage

### Training
Use the `train.py` script to train a model. The configs are stored in the file `configs.yaml`.

```bash
python train.py <model_name> --config_file configs.yaml --ds_path /data/datasets/smol_lm_corpus/fineweb_edu
```

### Command Line Arguments:
- `<model_name>`: Name of the model configuration to use (e.g., `gpt2_large`).
- `--config_file`: Path to the configuration file (default: `configs.yaml`).
- `--ds_path`: Path to the dataset (default: `/data/datasets/smol_lm_corpus/fineweb_edu`).

### Example Command:
```bash
python train.py gpt2 --config_file configs.yaml --ds_path /data/datasets/smol_lm_corpus/fineweb_edu
```

### Additional Training Parameters:
You can also specify additional training parameters in the configuration file, such as:
- `max_length`: Maximum sequence length for training.
- `batch_size`: Number of samples per gradient update.
- `epochs`: Number of epochs to train the model.
- `learning_rate`: Learning rate for the optimizer.

### Inference
Use the `generate_text.py` script to generate text with a trained model.

```bash
python generate_text.py <model_name> --config_file configs.yaml --save_dir checkpoints --max_length 100 --start_text "Your prompt here"
```

### Command Line Arguments:
- `<model_name>`: Name of the model configuration to use (e.g., `gpt2_medium`).
- `--config_file`: Path to the configuration file (default: `configs.yaml`).
- `--save_dir`: Directory where the model checkpoint is saved (default: `checkpoints`).
- `--max_length`: Maximum length of generated text (default: `100`).
- `--start_text`: The text to start generation (default: `"Fruits are good for you because"`).

### Example Command:
```bash
python generate_text.py gpt2_medium --config_file configs.yaml --save_dir checkpoints --max_length 100 --start_text "Once upon a time"
```

## Results

Since the model can not even finish the first epoch, both training and validation loss are indicative of how well the model is able to fit the data. So I am inlcuding an approximation of the loss based on the training and val loss.


| Model | Implementation | Training Time | # Training Batches | Batch Size | # Parameters* | Loss | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 Medium | From Scratch | 24 hours | 179K | 4 | 407M | 3.38 | Gradient clipping used |

\* These are the _total_ number of trainable parameters in the model. In case of many models, such as GPT-2, the embedding matrix is not considered a trainable parameter in the respective paper.

### Training Logs

Detailed training logs for each model are available in the `training_logs` folder. You can find the logs for specific models here:

- [GPT-2 Medium Training Logs](training_logs/gpt2_logs.md)

### Generated Text

**GPT-2 Medium**
```
Fruits are good for you because fruits contain vitamin C and vitamin E.
Fruits also contain fiber, which is very important for the health of the brain and the mind.
According to the Food and Agriculture Organization (FAO) it is recommended to consume a wide range of fruits and vegetables for the health of your body and the health of your brain and the mind.
Fruits are rich in vitamins, fiber, and vitamins and are also rich in vitamins and minerals.
Fruits are rich in vitamins and fiber as
```

## Issues

I intend to log the issues that I face and the fixes that I apply here.

- **Handling Exploding/Vanishing Gradients**: 
  - The model training crashed after a few hours, and the loss was reported as `NaN`. This issue was caused by exploding (or vanishing) gradients. 
  - **Fixes**:
    - Applied **Gradient Clipping** to prevent gradients from exceeding a certain threshold.
    - Added checks to ensure that the loss is not `NaN` or `Inf` before the backward pass.

- **Special Character Handling in Tokenization**: 
  - One training session crashed due to a disallowed special token '<|endoftext|>' somewhere in the training data :/ Added `allowed_special` in the tokenization to fix it: `tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})`

- **Learning Rate for GPT-2 Medium**:
  - I first tried a learning rate of `3e-4` which seemed fine, but later found out that the learning rate should be `1e-4` led to faster convergence and lower loss. I still have not tried even lower learning rates or learning rate schedules.

- **Mixed Precision**:
  - I tried PyTorch's Mixed Precision for GPT-2 Medium but it did not reduce GPU memory usage. My setup is 2 x RTX 4090 GPUs with 24GB RAM each.

## Acknowledgements
- The initial code is adapted from [Sebastian Raschka](https://github.com/rasbt) and his [LLM Workshop 2024](https://github.com/rasbt/LLM-workshop-2024?tab=readme-ov-file) on "Pretraining and Finetuning LLMs from the Ground Up"
- This repo was inspired by [karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), which I used as a reference to build my own earlier version of a GPT from scratch: [GPT-Nano](https://github.com/Usman-Rafique/GPT-Nano)
