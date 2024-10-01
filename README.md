# LLM Forge

![icon](icon.png)

*(Image generated using [Imagen3 by Google](https://deepmind.google/technologies/imagen-3/))*

This repo is my experimental playground for building and training practical LLMs with relatively limited compute resources. The goal is to train real LLMs on actual datasets, starting with pre-training from scratch and eventually moving towards instruction tuning with pre-trained models.

## Goals
1. **From Scratch**: No pre-trained models, no fine-tuning, no API.
2. **Limited Training Time**: ~ 24 hours.
3. **Limited Resources**: 2 GPUs, 24GB RAM each.
4. **Publicly Available Resources**: Using only publicly available datasets and tools.

## Dataset

The dataset used for training is the [FineWebEdu Deduplicated from SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). The dataset is a de-duplicated version of [FineWebEdu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and was filtered using a classifier to retain only high-quality educational content.

Please note that the dataset is much bigger, and we will probably not use all of it for training.

## Pre-trained Checkpoints

We provide pre-trained model checkpoints for easy experimentation and text generation. You can download these checkpoints from our [Google Drive folder](https://drive.google.com/drive/folders/1qFAAdg4SYwNkT-jHdES46OVJRHpYvry8?usp=sharing).

Currently available checkpoints:
- `model_gpt2_medium.pth`

To use a pre-trained checkpoint:

1. Download the desired checkpoint file from the Google Drive link.
2. Place it in the `checkpoints` directory of your project.
3. Use it for text generation or continue training from this point.

For detailed instructions on downloading and using checkpoints, please refer to the [Setup Guide](docs/setup.md).

## Quick Start

For detailed setup instructions, please refer to [docs/setup.md](docs/setup.md).

1. Clone the repository:
   ```
   git clone https://github.com/Usman-Rafique/llm_forge.git
   cd llm_forge
   ```

2. Set up the environment and install dependencies:
   ```
   python -m venv llm
   source llm/bin/activate  # On Windows use: llm\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. Train a model:
   ```
   python -m llm_forge.train gpt2_medium --config_file configs/gpt2.yaml
   ```

4. Generate text using a pre-trained checkpoint:
   ```
   python -m llm_forge.generate_text gpt2_medium --config_file configs/gpt2.yaml --checkpoint checkpoints/model_gpt2_medium.pth
   ```

For detailed information on CLI arguments for training and text generation, please refer to:
- [docs/train_cli.md](docs/train_cli.md)
- [docs/generate_cli.md](docs/generate_cli.md)

## Project Structure

```
llm_forge/
├── src/
│   └── llm_forge/
│       ├── models/
│       ├── data/
│       ├── utils/
│       ├── train.py
│       └── generate_text.py
├── tests/
├── configs/
├── docs/
├── checkpoints/
└── README.md
```

## Features

- Modular architecture for easy experimentation
- Support for GPT-style models
- Configurable model sizes and architectures
- Text generation capabilities

## Models
Models supported so far:

- [x] GPT-2 from Scratch

## Results

Since the model can not even finish the first epoch, both training and validation loss are indicative of how well the model is able to fit the data. So I am including an approximation of the loss based on the training and val loss.

| Model | Implementation | Training Time | # Training Batches | Batch Size | # Parameters* | Loss | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 Medium | From Scratch | 24 hours | 179K | 4 | 407M | 3.38 | Gradient clipping used |

\* These are the _total_ number of trainable parameters in the model. In case of many models, such as GPT-2, the embedding matrix is not considered a trainable parameter in the respective paper.

### Training Logs

Detailed training logs for each model are available in the `docs` folder. You can find the logs for specific models here:

- [GPT-2 Medium Training Logs](docs/gpt2_logs.md)

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

## Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](docs/CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The initial code is adapted from [Sebastian Raschka](https://github.com/rasbt) and his [LLM Workshop 2024](https://github.com/rasbt/LLM-workshop-2024?tab=readme-ov-file) on "Pretraining and Finetuning LLMs from the Ground Up"
- This repo was inspired by [karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), which I used as a reference to build my own earlier version of a GPT from scratch: [GPT-Nano](https://github.com/Usman-Rafique/GPT-Nano)
