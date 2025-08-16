import numpy as np
import string

# MATH GR5470 Assignment 2
# @author Minze Li ml5163
# Feb 20, 2025

class Tokenizer:
    """
    A simple character-level tokenizer that handles lowercase letters, digits, and punctuation.
    """
    def __init__(self):
        chars = string.ascii_lowercase + string.digits + string.punctuation
        # Create dictionary mappings
        self.vocab = {c:i for i, c in enumerate(chars)}
        self.inverse_vocab = {i:c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        """Convert text to token indices."""
        return [self.vocab[c] for c in text if c in self.vocab]

    def decode(self, tokens):
        """Convert token indices back to text."""
        return ''.join([self.inverse_vocab[t] for t in tokens])

class PositionalEncoding:
    """
    Implements sinusoidal positional encoding as described in 'Attention Is All You Need'.
    """
    def __init__(self, d_model):
        self.d_model = d_model

    def forward(self, seq_len):
        # Create positional encoding matrix
        pe = np.zeros((seq_len, self.d_model))
        position = np.arange(seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class ScaledDotProductAttention:
    """
    Implements scaled dot-product attention mechanism.
    """
    def forward(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q, K, V: Query, Key, and Value matrices
            mask: Optional attention mask
        
        Returns:
            attention output and attention weights
        """
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose()) / np.sqrt(d_k)
        if mask is not None:
            scores += mask
        attn_weights = softmax(scores)
        output = np.matmul(attn_weights, V)
        return output

class MultiHeadAttention:
    """
    Implements multi-head attention with masking capability.
    """
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = np.random.normal(0, 1, (d_model, d_model))
        self.W_K = np.random.normal(0, 1, (d_model, d_model))
        self.W_V = np.random.normal(0, 1, (d_model, d_model))
        self.W_O = np.random.normal(0, 1, (d_model, d_model))
        self.attention = ScaledDotProductAttention()

    def forward(self, X, mask):
        # Linear projections and split heads
        seq_len = X.shape[0]
        Q = np.matmul(X, self.W_Q)
        K = np.matmul(X, self.W_K)
        V = np.matmul(X, self.W_V)

        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        # Compute attention
        attn_outputs = []
        for i in range(self.num_heads):
            Q_head = Q[i]
            K_head = K[i]
            V_head = V[i]
            attn_out = self.attention.forward(Q_head, K_head, V_head, mask)
            attn_outputs.append(attn_out)
        # Combine heads
        concatenated = np.concatenate(attn_outputs, axis=1)
        output = np.matmul(concatenated, self.W_O)
        return output

class FeedForward:
    """
    Implements the feed-forward network used in transformer blocks.
    """
    def __init__(self, d_model, hidden_dim):
        self.W1 = np.random.normal(0, 1, (d_model, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.normal(0, 1, (hidden_dim, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """Apply feed-forward transformation with ReLU activation."""
        x = np.matmul(x, self.W1) + self.b1
        x = np.maximum(x, 0)
        x = np.matmul(x, self.W2) + self.b2
        return x

class DecoderBlock:
    """
    Implements a single decoder block with masked self-attention and feed-forward network.
    """
    def __init__(self, d_model, num_heads, hidden_dim):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, hidden_dim)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask):
        attn_output = self.self_attn.forward(x, mask)
        x = x + attn_output
        x = self.norm1.forward(x)
        ffn_output = self.ffn.forward(x)
        x = x + ffn_output
        x = self.norm2.forward(x)
        return x

class Transformer:
    """
    Implements a complete a transformer for character-level prediction.
    """
    def __init__(self, tokenizer, d_model=16, num_heads=2, hidden_dim=32, num_blocks=2):
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.embedding = np.random.normal(0, 1, (tokenizer.vocab_size, d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_blocks)]
        self.fc = np.random.normal(0, 1, (d_model, tokenizer.vocab_size))

    def get_mask(self, seq_len):
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
        return mask

    def forward(self, input_ids):
        seq_len = len(input_ids)
        x = self.embedding[input_ids]
        x += self.pos_encoder.forward(seq_len)
        mask = self.get_mask(seq_len)
        for block in self.decoder_blocks:
            x = block.forward(x, mask)
        logits = np.matmul(x, self.fc)
        return logits

def main():
    tokenizer = Tokenizer()
    transformer = Transformer(tokenizer)
    input_text = "I love GR5470"
    input_ids = tokenizer.encode(input_text)
    if not input_ids:
        print("Input contains invalid characters.")
        return
    logits = transformer.forward(input_ids)
    next_logits = logits[-1]
    probabilities = softmax(next_logits)
    predicted_id = np.argmax(probabilities)
    predicted_char = tokenizer.decode([predicted_id])
    print(f"Input: {input_text}")
    print(f"Generated: {predicted_char}")

if __name__ == "__main__":
    main()