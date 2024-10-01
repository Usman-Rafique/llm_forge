# Setup Guide

## Repository Setup

1. Clone the repository:
   ```
   git clone https://github.com/Usman-Rafique/llm_forge.git
   cd llm_forge
   ```

2. Set up the environment:

   Choose either `venv` or `conda`:

   A. Using `venv`:
      ```
      python -m venv llm
      ```
      Activate the environment:
      ```
      source llm/bin/activate
      ```

   B. Using `conda`:
      ```
      conda create -n llm python=3.9
      conda activate llm
      ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the project in editable mode:
   ```
   pip install -e .
   ```

## Dataset Setup

1. Create a `datasets` folder in the project root:
   ```
   mkdir datasets
   ```

2. Create a symlink to your actual dataset:
   ```
   ln -s /path/to/actual/dataset/smol_lm_corpus datasets/smol_lm_corpus
   ```

   Replace `/path/to/actual/dataset/smol_lm_corpus` with the actual path to your dataset.

3. (Optional) If you want to use a different dataset path, you can set the `DATASET_PATH` environment variable:
   ```
   export DATASET_PATH=/path/to/your/dataset
   ```

## Running Tests

Ensure your environment is activated, then run the tests using:

```
python -m unittest discover tests
```

## Training

To start training, ensure your environment is activated, then run:

```
python src/llm_forge/train.py
```

You can override the default dataset path by setting the `DATASET_PATH` environment variable before running the script.

## Deactivating the Environment

When you're done working on the project:

- If using venv, run:
  ```
  deactivate
  ```

- If using conda, run:
  ```
  conda deactivate
  ```

## Text Generation

To generate text using a trained model:

```
python -m llm_forge.generate_text gpt2_medium --config_file configs/gpt2.yaml
```

You can specify a custom checkpoint file using the `--checkpoint` argument:

```
python -m llm_forge.generate_text gpt2_medium --config_file configs/gpt2.yaml --checkpoint /path/to/your/checkpoint.pth
```

For more options, run:

```
python -m llm_forge.generate_text --help
```

## Downloading Checkpoints

Pre-trained model checkpoints are available for download:

1. Visit the [LLM Forge Checkpoints Google Drive folder](https://drive.google.com/drive/folders/1qFAAdg4SYwNkT-jHdES46OVJRHpYvry8?usp=sharing).
2. Download the desired checkpoint file (e.g., `model_gpt2_medium.pth`).
3. Place the downloaded file in the `checkpoints` directory of your project.

For detailed instructions on using checkpoints, see the [Text Generation CLI guide](generate_cli.md).
