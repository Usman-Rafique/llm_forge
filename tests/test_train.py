import unittest
from unittest.mock import patch, MagicMock
import torch
from llm_forge.train import train, process_batch


class TestTrain(unittest.TestCase):

    @patch('llm_forge.train.tqdm')
    @patch('llm_forge.train.SummaryWriter')
    @patch('llm_forge.data.dataset.create_dataloaders')
    @patch('llm_forge.train.process_batch')
    def test_train_function(self, mock_process_batch, mock_create_dataloaders,
                            mock_summary_writer, mock_tqdm):
        # Create mock objects
        mock_model = MagicMock()
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_loss_function = MagicMock()
        mock_optimizer = MagicMock()

        # Set up mock behavior
        mock_model.to.return_value = mock_model
        mock_train_loader.__len__.return_value = 1
        mock_train_loader.__iter__.return_value = iter([{
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }])
        mock_val_loader.__iter__.return_value = iter([{
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }])
        mock_create_dataloaders.return_value = (mock_train_loader,
                                                mock_val_loader)

        # Mock process_batch to return a tensor that requires gradients
        mock_process_batch.return_value = torch.tensor(0.5, requires_grad=True)

        # Set up mock_tqdm to return a MagicMock that can be iterated over
        mock_tqdm_instance = MagicMock()
        mock_tqdm_instance.__iter__.return_value = iter(mock_train_loader)
        mock_tqdm.return_value = mock_tqdm_instance

        # Call the train function with the new model_name parameter
        train(mock_model,
              mock_train_loader,
              mock_val_loader,
              mock_loss_function,
              mock_optimizer,
              num_epochs=1,
              save_dir='dummy_save_dir',
              log_dir='dummy_log_dir',
              devices=['cpu'],
              eval_frequency=5,
              num_val_batches=1,
              use_mixed_precision=False,
              model_name='gpt2_medium')

        # Debugging call counts
        print(f"train() called: {mock_model.train.call_count}")
        print(f"zero_grad() called: {mock_optimizer.zero_grad.call_count}")
        print(f"step() called: {mock_optimizer.step.call_count}")
        print(f"process_batch() called: {mock_process_batch.call_count}")

        # Assert that certain methods were called
        self.assertTrue(mock_model.train.called, "model.train() was not called")
        self.assertTrue(mock_optimizer.zero_grad.called,
                        "optimizer.zero_grad() was not called")
        self.assertTrue(mock_optimizer.step.called,
                        "optimizer.step() was not called")
        self.assertTrue(mock_process_batch.called,
                        "process_batch() was not called")

        # Assert minimum call counts
        self.assertGreaterEqual(mock_model.train.call_count, 1,
                                "model.train() should be called at least once")
        self.assertGreaterEqual(
            mock_optimizer.zero_grad.call_count, 1,
            "optimizer.zero_grad() should be called at least once")
        self.assertGreaterEqual(
            mock_optimizer.step.call_count, 1,
            "optimizer.step() should be called at least once")
        self.assertGreaterEqual(
            mock_process_batch.call_count, 1,
            "process_batch() should be called at least once")

        # Check if the train loader was iterated
        mock_train_loader.__iter__.assert_called()

    def test_process_batch(self):
        # Create mock objects
        mock_model = MagicMock()
        mock_loss_function = MagicMock()

        # Create a dummy batch
        batch = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        # Set up mock behavior
        mock_model.return_value = torch.randn(2, 10, 100)
        mock_loss_function.return_value = torch.tensor(0.5)

        # Call the process_batch function
        loss = process_batch(mock_model, batch, mock_loss_function, 'cpu')

        # Assert that the loss is not None and is a tensor
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
