import unittest
import torch
import tiktoken
from data import FineWebEduDataset, create_dataloaders, collate_fn


class TestData(unittest.TestCase):

    def setUp(self):
        self.tokenizer = tiktoken.get_encoding('r50k_base')
        self.max_length = 128
        self.batch_size = 2

    def test_fineweb_edu_dataset(self):
        # Create a mock dataset
        mock_dataset = [{
            'text': 'This is a test sentence.'
        }, {
            'text': 'Another test sentence.'
        }]
        dataset = FineWebEduDataset(mock_dataset, self.tokenizer,
                                    self.max_length)

        self.assertEqual(len(dataset), 2)

        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('labels', item)
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['labels'], torch.Tensor)

    def test_collate_fn(self):
        # Create mock batch items
        batch = [{
            'input_ids': torch.tensor([1, 2, 3], dtype=torch.long),
            'labels': torch.tensor([2, 3, 4], dtype=torch.long)
        }, {
            'input_ids': torch.tensor([1, 2], dtype=torch.long),
            'labels': torch.tensor([2, 3], dtype=torch.long)
        }]

        collated_batch = collate_fn(batch)

        self.assertIn('input_ids', collated_batch)
        self.assertIn('labels', collated_batch)
        self.assertIn('attention_mask', collated_batch)
        self.assertEqual(collated_batch['input_ids'].shape, (2, 3))
        self.assertEqual(collated_batch['labels'].shape, (2, 3))
        self.assertEqual(collated_batch['attention_mask'].shape, (2, 3))

    def test_create_dataloaders(self):
        # Use a small test dataset path
        test_ds_path = 'path/to/test/dataset'

        # Mock the load_from_disk function
        def mock_load_from_disk(path):
            return [{'text': f'Test sentence {i}'} for i in range(100)]

        with unittest.mock.patch('data.load_from_disk',
                                 side_effect=mock_load_from_disk):
            train_loader, val_loader = create_dataloaders(test_ds_path,
                                                          self.batch_size,
                                                          self.max_length,
                                                          self.tokenizer,
                                                          val_split=0.2)

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)


if __name__ == '__main__':
    unittest.main()
