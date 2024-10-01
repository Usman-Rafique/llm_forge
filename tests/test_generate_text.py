import unittest
from unittest.mock import patch, MagicMock
from llm_forge.generate_text import main


class TestGenerateText(unittest.TestCase):

    @patch('llm_forge.generate_text.load_config')
    @patch('llm_forge.generate_text.ModelFactory.create_model')
    @patch('llm_forge.generate_text.tiktoken.get_encoding')
    @patch('llm_forge.generate_text.torch.load')
    @patch('llm_forge.generate_text.os.path.exists')
    def test_main(self, mock_exists, mock_torch_load, mock_get_encoding,
                  mock_create_model, mock_load_config):
        # Mock the arguments
        with patch('sys.argv', ['generate_text.py', 'gpt2_medium']):
            # Set up mock returns
            mock_load_config.return_value = ({
                'tokenizer_name': 'r50k_base',
                'devices': ['cpu'],
                'save_dir': '/dummy/path'
            }, {
                'context_length': 1024,
                'emb_dim': 1024
            })
            mock_get_encoding.return_value = MagicMock(n_vocab=50000)
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            mock_exists.return_value = True
            mock_torch_load.return_value = {'model_state_dict': {}}

            # Call the main function
            main()

            # Assert that the necessary functions were called
            mock_load_config.assert_called_once()
            mock_get_encoding.assert_called_once()
            mock_create_model.assert_called_once()
            mock_model.to.assert_called_once()
            mock_model.load_state_dict.assert_called_once()
            mock_model.generate_text.assert_called_once()


if __name__ == '__main__':
    unittest.main()
