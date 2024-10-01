import torch
import torch.nn as nn
import tiktoken
import torch.nn.functional as F
import time


# Most of the code is taken from Sebastian Raschka's LLM workshop
# https://github.com/rasbt/LLM-workshop-2024?tab=readme-ov-file
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        context_length (int): Maximum context length.
        dropout (float): Dropout rate.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in query, key, and value projections.
    """

    def __init__(self,
                 d_in,
                 d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out).
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads,
                         self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads,
                               self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads,
                             self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        causal_mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(causal_mask, -torch.inf)

        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            attn_scores.masked_fill_(~expanded_mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        emb_dim (int): Embedding dimension.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """
    Feed-forward neural network.

    Args:
        emb_dim (int): Embedding dimension.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of multi-head attention and feed-forward layers.

    Args:
        context_length (int): Maximum context length.
        emb_dim (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        drop_rate (float): Dropout rate.
        qkv_bias (bool): Whether to use bias in query, key, and value projections.
    """

    def __init__(self, context_length, emb_dim, n_heads, drop_rate, qkv_bias):
        super().__init__()
        self.att = MultiHeadAttention(d_in=emb_dim,
                                      d_out=emb_dim,
                                      context_length=context_length,
                                      num_heads=n_heads,
                                      dropout=drop_rate,
                                      qkv_bias=qkv_bias)
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x, attention_mask=None):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, attention_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    GPT-style language model with configurable size and architecture.

    Args:
        vocab_size (int): Size of the vocabulary.
        tokenizer_name (str): Name of the tokenizer to use.
        context_length (int): Maximum length of the input sequence.
        emb_dim (int): Dimension of the embedding vectors.
        n_heads (int): Number of attention heads in each transformer block.
        n_layers (int): Number of transformer blocks.
        drop_rate (float): Dropout rate.
        qkv_bias (bool): Whether to use bias in query, key, and value projections.
    """

    def __init__(self, vocab_size, tokenizer_name, context_length, emb_dim,
                 n_heads, n_layers, drop_rate, qkv_bias):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)
        self.context_length = context_length

        self.trf_blocks = nn.Sequential(*[
            TransformerBlock(context_length, emb_dim, n_heads, drop_rate,
                             qkv_bias) for _ in range(n_layers)
        ])

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, in_idx, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            in_idx (torch.Tensor): Input token indices of shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x, attention_mask)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def generate_text(self,
                      start_text="",
                      max_length=100,
                      temperature=0.7,
                      max_time=60,
                      device=None):
        """
        Generate text using the trained model.

        Args:
            start_text (str, optional): Initial text to start generation. Defaults to "".
            max_length (int, optional): Maximum number of tokens to generate. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_time (int, optional): Maximum generation time in seconds. Defaults to 60.
            device (torch.device, optional): Device to use for generation. Defaults to None.

        Returns:
            str: Generated text.
        """
        self.eval()
        start_time = time.time()

        if device is None:
            device = next(self.parameters()).device

        if start_text:
            context = torch.tensor(self.tokenizer.encode(start_text),
                                   dtype=torch.long).unsqueeze(0).to(device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long).to(device)

        generated_text = start_text

        with torch.no_grad():
            for _ in range(max_length):
                if time.time() - start_time > max_time:
                    print(f"Generation stopped after {max_time} seconds.")
                    break

                logits = self(context)
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)

                if torch.all(next_token_probs == 0):
                    next_token_probs = torch.ones_like(
                        next_token_probs) / next_token_probs.size(-1)

                next_token_probs = torch.clamp(next_token_probs, min=1e-8)
                next_token_probs = next_token_probs / next_token_probs.sum()

                next_token = torch.multinomial(next_token_probs, num_samples=1)

                generated_token = self.tokenizer.decode([next_token.item()])
                generated_text += generated_token

                context = torch.cat([context, next_token], dim=1)

                if next_token.item() == self.tokenizer.eot_token:
                    break

                if context.size(1) > self.context_length:
                    context = context[:, -self.context_length:]

        return generated_text.strip()


if __name__ == "__main__":
    # Test configuration for GPT2 medium
    max_length = 2048
    tokenizer_name = 'r50k_base'
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    vocab_size = tokenizer.n_vocab
    context_length = max_length
    emb_dim = 1024
    n_heads = 16
    n_layers = 24
    drop_rate = 0.1
    qkv_bias = True

    # Instantiate the model
    gpt2_medium = GPTModel(vocab_size=vocab_size,
                           tokenizer_name=tokenizer_name,
                           context_length=context_length,
                           emb_dim=emb_dim,
                           n_heads=n_heads,
                           n_layers=n_layers,
                           drop_rate=drop_rate,
                           qkv_bias=qkv_bias)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_medium.to(device)

    # Print model summary
    print(gpt2_medium)

    # Count and print the number of parameters
    num_params = sum(
        p.numel() for p in gpt2_medium.parameters() if p.requires_grad)
    print(f'\nThe model has {num_params:,} trainable parameters')

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (1, 100)).to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = gpt2_medium(input_ids, attention_mask)

    print(f'\nOutput shape: {output.shape}')

    # Test text generation
    print("\nGenerating text (this may take a while)...")
    generated_text = gpt2_medium.generate_text("Once upon a time",
                                               max_length=200,
                                               max_time=120,
                                               device=device)
    print(f'\nGenerated text:\n{generated_text}')
    print(f'\nGenerated text length: {len(generated_text)} characters')