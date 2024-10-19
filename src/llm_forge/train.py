import os
import argparse
import yaml
import tiktoken
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from llm_forge.models.factory import ModelFactory
from llm_forge.data import create_dataloaders
import torch.nn.utils as utils
from llm_forge.utils.config import load_config


def train(model, train_loader, loss_function, optimizer, num_epochs, save_dir,
          log_dir, devices, eval_frequency, use_mixed_precision, model_name):
    # Move the model to the primary device
    model = model.to(devices[0])

    if len(devices) > 1:
        # Use DataParallel
        model = nn.DataParallel(model, device_ids=devices)

    writer = SummaryWriter(log_dir)
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision)
    best_loss = float('inf')
    last_checkpoint_path = None

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{num_epochs}")

        total_loss = 0
        steps_since_last_log = 0

        for batch in train_loader:  # Iterate over the DataLoader
            # Move each tensor in the batch to the primary device
            batch = {k: v.to(devices[0]) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                loss = process_batch(model, batch, loss_function, devices[0])
                if loss is None:
                    continue

            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            steps_since_last_log += 1
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({'train_loss': f"{loss.item():.4f}"})

            if global_step % eval_frequency == 0:
                avg_loss = total_loss / steps_since_last_log
                writer.add_scalar('Loss/train', avg_loss, global_step)
                print(
                    f"\nStep {global_step}, Average Train Loss: {avg_loss:.4f}")

                # Save the last model
                last_checkpoint_path = save_model(model,
                                                  optimizer,
                                                  global_step,
                                                  avg_loss,
                                                  save_dir,
                                                  model_name,
                                                  is_best=False)

                # Save the best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model(model,
                               optimizer,
                               global_step,
                               avg_loss,
                               save_dir,
                               model_name,
                               is_best=True)

                # Reset loss tracking
                total_loss = 0
                steps_since_last_log = 0

        progress_bar.close()

    # After training is complete, rename the last checkpoint to 'last'
    if last_checkpoint_path:
        last_model_path = os.path.join(save_dir, f'model_{model_name}_last.pth')
        os.rename(last_checkpoint_path, last_model_path)
        print(f"Renamed last checkpoint to: {last_model_path}")

    writer.close()


def process_batch(model, batch, loss_function, device):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    logits = model(input_ids, attention_mask=attention_mask)

    # Reshape logits and labels for loss computation
    logits = logits.flatten(0, 1)
    labels = labels.flatten()

    # Compute loss only on non-padded tokens
    non_pad_mask = (labels != -100)

    # Use the mask to filter out padded elements
    filtered_logits = logits[non_pad_mask]
    filtered_labels = labels[non_pad_mask]

    # Compute the loss on the filtered elements
    loss = loss_function(filtered_logits, filtered_labels)

    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: Loss is NaN or Inf. Skipping batch.")
        return None

    return loss


def save_model(model, optimizer, global_step, avg_loss, save_dir, model_name,
               is_best):
    # Get the original model if it's wrapped in DataParallel
    model_to_save = model.module if isinstance(model,
                                               nn.DataParallel) else model

    # Prepare the checkpoint
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'avg_loss': avg_loss,
    }

    # Save the best model
    if is_best:
        best_checkpoint_name = f'model_{model_name}_best.pth'
        torch.save(checkpoint, os.path.join(save_dir, best_checkpoint_name))
        print(
            f"Best model saved as {best_checkpoint_name} with average loss: {avg_loss:.4f}"
        )
        return None


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    avg_loss = checkpoint['avg_loss']
    print(
        f"Loaded checkpoint from step {global_step} with average loss: {avg_loss:.4f}"
    )
    return global_step, avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT model with specified configuration")
    parser.add_argument(
        "model_name",
        help="Name of the model configuration to use (e.g., gpt2_large)")
    parser.add_argument("--config_file",
                        default="gpt2",
                        help="Path to the configuration file or config name")
    parser.add_argument("--ds_path",
                        default="datasets/smol_lm_corpus/fineweb_edu",
                        help="Path to the dataset")
    parser.add_argument("--checkpoint",
                        help="Path to a checkpoint to resume training from")
    args = parser.parse_args()

    global_config, model_config = load_config(args.config_file, args.model_name)

    tokenizer = tiktoken.get_encoding(global_config['tokenizer_name'])
    model_config['vocab_size'] = tokenizer.n_vocab
    model_config['tokenizer_name'] = global_config['tokenizer_name']

    model = ModelFactory.create_model('gpt', model_config)

    train_loader, _ = create_dataloaders(
        ds_path=args.ds_path,
        batch_size=global_config['batch_size'],
        max_length=model_config['context_length'],
        tokenizer=tokenizer)

    loss_function = nn.CrossEntropyLoss()
    optimizer_name = global_config['optimizer']
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(),
        lr=global_config['learning_rate'],
        weight_decay=global_config['weight_decay'])

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())
              ] if torch.cuda.is_available() else ["cpu"]

    use_mixed_precision = global_config.get('mixed_precision', False)
    if use_mixed_precision:
        print('INFO: using Mixed-Precision Training')

    print(f"INFO: using {len(devices)} GPUs")
    print(f"INFO: using {global_config['num_epochs']} epochs")
    print(f"INFO: using {global_config['save_dir']} as save directory")
    print(f"INFO: using {global_config['log_dir']} as log directory")
    print(
        f"INFO: using {global_config['eval_frequency']} as evaluation frequency"
    )
    print(
        f"INFO: using {global_config['mixed_precision']} as mixed precision training"
    )
    print(f"INFO: model name: {args.model_name}")
    print(f"INFO: model config: {model_config}")

    train(model, train_loader, loss_function, optimizer,
          global_config['num_epochs'], global_config['save_dir'],
          global_config['log_dir'], devices, global_config['eval_frequency'],
          use_mixed_precision, args.model_name)


if __name__ == "__main__":
    main()
