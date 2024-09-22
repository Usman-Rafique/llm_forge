from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.nn.utils.rnn import pad_sequence


class FineWebEduDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']

        # Tokenization
        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Truncate if necessary
        if len(tokens) > self.max_length + 1:
            tokens = tokens[:self.max_length + 1]

        # Create input_ids and labels
        input_ids = tokens[:-1]  # All tokens except the last one
        labels = tokens[1:]  # All tokens except the first one

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch):
    # Sort the batch by sequence length (descending order)
    batch.sort(key=lambda x: len(x['input_ids']), reverse=True)

    input_ids = [item['input_ids'].clone().detach() for item in batch]
    labels = [item['labels'].clone().detach() for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids,
                                    batch_first=True,
                                    padding_value=0)
    labels_padded = pad_sequence(
        labels, batch_first=True,
        padding_value=-100)  # Use -100 for ignored label in loss computation

    # Create attention masks
    attention_mask = (input_ids_padded != 0).long()

    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'attention_mask': attention_mask
    }


def create_dataloaders(ds_path,
                       batch_size,
                       max_length,
                       tokenizer,
                       val_split=0.01,
                       num_workers=4,
                       seed=42):
    full_dataset = load_from_disk(ds_path)

    # Calculate split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))

    # Create FineWebEduDataset instances
    train_dataset = FineWebEduDataset(train_dataset, tokenizer, max_length)
    val_dataset = FineWebEduDataset(val_dataset, tokenizer, max_length)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn)

    return train_loader, val_loader


if __name__ == "__main__":
    # This section is for testing purposes
    import tiktoken

    ds_path = '/data/datasets/smol_lm_corpus/fineweb_edu'
    tokenizer_name = 'r50k_base'
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    max_length = 2048
    batch_size = 4

    train_loader, val_loader = create_dataloaders(ds_path, batch_size,
                                                  max_length, tokenizer)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Test a batch from train_loader
    train_iter = iter(train_loader)
    train_batch = next(train_iter)
    print("\nTraining batch:")
    print("Input IDs shape:", train_batch['input_ids'].shape)
    print("Labels shape:", train_batch['labels'].shape)
    print("Attention mask shape:", train_batch['attention_mask'].shape)

    # Test a batch from val_loader
    val_iter = iter(val_loader)
    val_batch = next(val_iter)
    print("\nValidation batch:")
    print("Input IDs shape:", val_batch['input_ids'].shape)
    print("Labels shape:", val_batch['labels'].shape)
    print("Attention mask shape:", val_batch['attention_mask'].shape)
