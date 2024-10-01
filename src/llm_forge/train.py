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


def train(model, train_loader, val_loader, loss_function, optimizer, num_epochs,
          save_dir, log_dir, devices, eval_frequency, num_val_batches,
          use_mixed_precision, model_name):
    # Move the model to the primary device
    model = model.to(devices[0])

    if len(devices) > 1:
        # Use DataParallel
        model = nn.DataParallel(model, device_ids=devices)

    writer = SummaryWriter(log_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    global_step = 0

    scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision)

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in train_loader:
            # Move the batch to the primary device
            for key in batch:
                batch[key] = batch[key].to(devices[0])

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                loss = process_batch(model, batch, loss_function, devices[0])
                if loss is None:
                    continue

            if use_mixed_precision:
                scaler.scale(
                    loss).backward()  # Scale the loss for mixed precision
                # apply gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0)  # You can adjust max_norm as needed
                scaler.step(optimizer)  # Step the optimizer
                scaler.update()  # Update the scaler
            else:
                loss.backward()  # Standard backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # Step the optimizer

            # torch.cuda.empty_cache()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({'train_loss': f"{loss.item():.4f}"})

            if global_step % eval_frequency == 0:
                # Use the primary device for validation
                avg_val_loss = validate(model,
                                        val_loader,
                                        loss_function,
                                        num_val_batches,
                                        device=devices[0])
                best_val_loss = log_and_save(writer,
                                             model, optimizer, global_step,
                                             loss.item(), avg_val_loss,
                                             save_dir, best_val_loss,
                                             model_name)

        progress_bar.close()

    writer.close()


def process_batch(model, batch, loss_function, device):
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)

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


def validate(model, val_loader, loss_function, num_val_batches, device):
    model.eval()
    total_val_loss = 0
    val_steps = 0
    val_iter = iter(val_loader)

    with torch.no_grad():
        for _ in range(num_val_batches):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)

            val_loss = process_batch(model, val_batch, loss_function, device)
            if val_loss is not None:
                total_val_loss += val_loss.item()
                val_steps += 1

    avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
    return avg_val_loss


def log_and_save(writer, model, optimizer, global_step, train_loss, val_loss,
                 save_dir, best_val_loss, model_name):
    writer.add_scalar('Loss/train', train_loss, global_step)
    writer.add_scalar('Loss/val', val_loss, global_step)

    print(
        f"\nStep {global_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # Get the original model if it's wrapped in DataParallel
        model_to_save = model.module if isinstance(model,
                                                   nn.DataParallel) else model

        # Save the model state dict with the new naming convention
        checkpoint_name = f'model_{model_name}.pth'
        torch.save(
            {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_val_loss': best_val_loss,
            }, os.path.join(save_dir, checkpoint_name))

        print(
            f"New best model saved as {checkpoint_name} with validation loss: {best_val_loss:.4f}"
        )

    return best_val_loss


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
    args = parser.parse_args()

    global_config, model_config = load_config(args.config_file, args.model_name)

    tokenizer = tiktoken.get_encoding(global_config['tokenizer_name'])
    model_config['vocab_size'] = tokenizer.n_vocab
    model_config['tokenizer_name'] = global_config['tokenizer_name']

    model = ModelFactory.create_model('gpt', model_config)

    train_loader, val_loader = create_dataloaders(
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
        f"INFO: using {global_config['num_val_batches']} as number of validation batches"
    )
    print(
        f"INFO: using {global_config['mixed_precision']} as mixed precision training"
    )
    print(f"INFO: model name: {args.model_name}")
    print(f"INFO: model config: {model_config}")

    train(model, train_loader, val_loader, loss_function, optimizer,
          global_config['num_epochs'], global_config['save_dir'],
          global_config['log_dir'], devices, global_config['eval_frequency'],
          global_config['num_val_batches'], use_mixed_precision,
          args.model_name)


if __name__ == "__main__":
    main()
