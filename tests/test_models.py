import unittest
import torch
from models import GPTModel
import tiktoken


class TestModels(unittest.TestCase):

    def test_gpt_model_forward(self):
        model_config = {
            "vocab_size": 1000,
            "context_length": 128,
            "emb_dim": 256,
            "n_heads": 4,
            "n_layers": 2,
            "drop_rate": 0.1,
            "tokenizer_name": "r50k_base",
            "qkv_bias": True
        }
        model = GPTModel(**model_config)

        # Create a dummy input
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))

        # Test forward pass
        output = model(input_ids, attention_mask)
        self.assertEqual(output.shape, (batch_size, seq_length, 1000))

    def test_gpt_model_generate(self):
        model_config = {
            "context_length": 128,
            "emb_dim": 256,
            "n_heads": 4,
            "n_layers": 2,
            "drop_rate": 0.1,
            "tokenizer_name": "r50k_base",
            "qkv_bias": True
        }
        tokenizer = tiktoken.get_encoding(model_config["tokenizer_name"])
        model_config['vocab_size'] = tokenizer.n_vocab
        model = GPTModel(**model_config)

        start_text = "To test"
        max_length = 50

        # Generate text
        generated_text = model.generate_text(start_text=start_text,
                                             max_length=max_length)

        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > len(start_text))


if __name__ == '__main__':
    unittest.main()
