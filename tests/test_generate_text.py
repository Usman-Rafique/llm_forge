import unittest
from unittest.mock import patch, mock_open
from generate_text import load_config


class TestGenerateText(unittest.TestCase):

    @patch(
        'generate_text.open',
        new_callable=mock_open,
        read_data=
        '{"global": {"tokenizer_name": "r50k_base"}, "models": {"gpt2_medium": {"context_length": 1024, "emb_dim": 1024, "n_heads": 16, "n_layers": 24, "drop_rate": 0.1}}}'
    )
    @patch('generate_text.yaml.safe_load')
    def test_load_config(self, mock_safe_load, mock_file):
        mock_safe_load.return_value = {
            'global': {
                'tokenizer_name': 'r50k_base'
            },
            'models': {
                'gpt2_medium': {
                    'context_length': 1024,
                    'emb_dim': 1024,
                    'n_heads': 16,
                    'n_layers': 24,
                    'drop_rate': 0.1
                }
            }
        }

        global_config, model_config = load_config('dummy_path', 'gpt2_medium')

        self.assertEqual(global_config['tokenizer_name'], 'r50k_base')
        self.assertEqual(model_config['context_length'], 1024)
        self.assertEqual(model_config['emb_dim'], 1024)


if __name__ == '__main__':
    unittest.main()
