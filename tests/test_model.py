"""
Unit Tests for EdgeSight - Model Architecture

Run with: pytest tests/test_model.py -v
"""

import sys
import os
import pytest
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import FallNet, FallNetWithAttention, count_parameters


class TestFallNet:
    """Tests for FallNet model architecture."""

    def test_fallnet_creation(self):
        """Test that FallNet can be created."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        assert model is not None

    def test_fallnet_forward_pass(self):
        """Test forward pass through FallNet."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_len = 16
        input_size = 51
        x = torch.randn(batch_size, seq_len, input_size)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_fallnet_output_is_probability(self):
        """Test that output is in [0, 1] range."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        x = torch.randn(10, 16, 51)
        
        with torch.no_grad():
            output = model(x)
        
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_fallnet_different_batch_sizes(self):
        """Test that FallNet handles different batch sizes."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 16, 51)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 1)

    def test_fallnet_parameter_count(self):
        """Test parameter counting function."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        num_params = count_parameters(model)
        assert num_params > 0
        assert num_params < 1_000_000  # Should be under 1M params


class TestFallNetWithAttention:
    """Tests for FallNetWithAttention architecture."""

    def test_attention_model_creation(self):
        """Test that FallNetWithAttention can be created."""
        model = FallNetWithAttention(input_size=51, hidden_size=64, num_layers=2)
        assert model is not None

    def test_attention_model_forward_pass(self):
        """Test forward pass through FallNetWithAttention."""
        model = FallNetWithAttention(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        x = torch.randn(2, 16, 51)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)

    def test_attention_model_output_range(self):
        """Test that attention model output is in [0, 1]."""
        model = FallNetWithAttention(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        x = torch.randn(10, 16, 51)
        
        with torch.no_grad():
            output = model(x)
        
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


class TestModelTraining:
    """Tests for model training behavior."""

    def test_model_can_train_one_step(self):
        """Test that model can perform one training step."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        model.train()
        
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(4, 16, 51)
        y = torch.randint(0, 2, (4, 1)).float()
        
        # Forward
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        model.train()
        
        criterion = torch.nn.BCELoss()
        x = torch.randn(2, 16, 51)
        y = torch.randint(0, 2, (2, 1)).float()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients


class TestModelExport:
    """Tests for ONNX export behavior."""

    def test_model_can_be_exported_to_onnx(self):
        """Test that model can be exported to ONNX format."""
        model = FallNet(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        # Create dummy input for export
        dummy_input = torch.randn(1, 16, 51)
        
        # Try export (will fail if onnx not installed, which is OK for unit test)
        try:
            import tempfile
            import onnx
            
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                temp_path = f.name
            
            torch.onnx.export(
                model,
                dummy_input,
                temp_path,
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output']
            )
            
            # Verify export
            onnx_model = onnx.load(temp_path)
            onnx.checker.check_model(onnx_model)
            
            # Cleanup
            os.unlink(temp_path)
        except ImportError:
            pytest.skip("ONNX not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])