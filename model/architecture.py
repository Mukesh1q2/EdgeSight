"""
FallNet Architecture for EdgeSight fall detection.

A PyTorch model combining frame-level embeddings, LSTM temporal modeling,
and attention-based pooling for fall detection from pose sequences.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learnable attention pooling over temporal dimension.

    Uses a learnable query vector to compute attention weights
    over all time steps, producing a weighted average.
    """

    def __init__(self, hidden_dim: int):
        """Initialize attention pooling.

        Args:
            hidden_dim: Dimension of LSTM hidden state
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling.

        Args:
            lstm_outputs: (batch, seq_len, hidden_dim) LSTM outputs

        Returns:
            (batch, hidden_dim) pooled representation
        """
        # Project keys: (batch, seq_len, hidden_dim)
        keys = self.key_proj(lstm_outputs)

        # Compute attention scores: (batch, seq_len)
        # query: (1, hidden_dim) -> broadcast to (batch, hidden_dim)
        scores = torch.matmul(keys, self.query.T) * self.scale
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Softmax over sequence dimension
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum: (batch, hidden_dim)
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_outputs               # (batch, seq_len, hidden_dim)
        ).squeeze(1)

        return pooled


class FallNet(nn.Module):
    """Fall detection network for pose sequences.

    Architecture:
        Input: (batch, 30, 51) - 30 frames × 17 keypoints × 3 (x,y,conf)
        1. Linear(51, 128) + LayerNorm + ReLU - per-frame embedding
        2. LSTM(128, 256, 2 layers, dropout=0.3) - temporal modeling
        3. Attention pooling over time
        4. Linear(256, 64) + ReLU + Dropout(0.5)
        5. Linear(64, 1) + Sigmoid
        Output: fall probability (batch,)

    Designed for ONNX export compatibility - no dynamic control flow.
    """

    def __init__(
        self,
        input_dim: int = 51,
        frame_embed_dim: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        classifier_hidden_dim: int = 64,
        classifier_dropout: float = 0.5,
        bidirectional: bool = False
    ):
        """Initialize FallNet.

        Args:
            input_dim: Features per frame (17 keypoints × 3 = 51)
            frame_embed_dim: Dimension of frame-level embeddings
            lstm_hidden_dim: LSTM hidden state dimension
            lstm_layers: Number of LSTM layers
            lstm_dropout: Dropout between LSTM layers
            classifier_hidden_dim: Hidden dimension for classifier
            classifier_dropout: Dropout before final layer
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.frame_embed_dim = frame_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # Frame-level embedding
        self.frame_embed = nn.Sequential(
            nn.Linear(input_dim, frame_embed_dim),
            nn.LayerNorm(frame_embed_dim),
            nn.ReLU(inplace=True)
        )

        # LSTM for temporal modeling
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=frame_embed_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention pooling
        self.attention_pool = AttentionPooling(lstm_output_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize attention query
        nn.init.xavier_uniform_(self.attention_pool.query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               Default: (batch, 30, 51)

        Returns:
            Fall probability of shape (batch_size,)
        """
        batch_size = x.shape[0]

        # Frame-level embedding: (batch, seq_len, embed_dim)
        embedded = self.frame_embed(x)

        # LSTM: (batch, seq_len, lstm_output_dim)
        lstm_out, _ = self.lstm(embedded)

        # Attention pooling: (batch, lstm_output_dim)
        pooled = self.attention_pool(lstm_out)

        # Classification: (batch, 1)
        logits = self.classifier(pooled)

        # Sigmoid for probability
        prob = torch.sigmoid(logits).squeeze(-1)  # (batch,)

        return prob

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Attention weights of shape (batch, seq_len)
        """
        self.eval()
        with torch.no_grad():
            embedded = self.frame_embed(x)
            lstm_out, _ = self.lstm(embedded)

            # Compute attention weights manually
            keys = self.attention_pool.key_proj(lstm_out)
            scores = torch.matmul(keys, self.attention_pool.query.T) * self.attention_pool.scale
            scores = scores.squeeze(-1)
            attn_weights = F.softmax(scores, dim=1)

        return attn_weights


def create_model(config: Optional[dict] = None) -> FallNet:
    """Factory function to create FallNet with optional config.

    Args:
        config: Dictionary of model hyperparameters

    Returns:
        Initialized FallNet model
    """
    if config is None:
        config = {}

    return FallNet(
        input_dim=config.get('input_dim', 51),
        frame_embed_dim=config.get('frame_embed_dim', 128),
        lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
        lstm_layers=config.get('lstm_layers', 2),
        lstm_dropout=config.get('lstm_dropout', 0.3),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 64),
        classifier_dropout=config.get('classifier_dropout', 0.5),
        bidirectional=config.get('bidirectional', False)
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model() -> None:
    """Test FallNet with random input."""
    print("Testing FallNet architecture...")

    # Create model
    model = FallNet()
    print(f"Model created with {count_parameters(model):,} trainable parameters")

    # Test forward pass
    batch_size = 8
    seq_len = 30
    input_dim = 51

    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test attention weights
    attn_weights = model.get_attention_weights(x)
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum: {attn_weights.sum(dim=1).mean():.4f} (should be ~1.0)")

    # Test with different batch sizes (ONNX compatibility)
    for bs in [1, 16, 32]:
        x_test = torch.randn(bs, seq_len, input_dim)
        out = model(x_test)
        assert out.shape == (bs,), f"Batch size {bs} failed"
    print("Dynamic batch size test passed!")

    print("\nAll tests passed! Model is ONNX-ready.")


if __name__ == "__main__":
    test_model()
