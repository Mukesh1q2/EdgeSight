"""
Generate a test ONNX model for EdgeSight demo.
This creates a minimal LSTM model that can be loaded for testing.
"""

import torch
import torch.nn as nn
import os

class TestFallNet(nn.Module):
    def __init__(self, input_size=51, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        
        # Classification
        output = self.classifier(context)  # (batch, 1)
        return output

def generate_test_model():
    """Generate a test ONNX model for the demo."""
    
    model = TestFallNet(input_size=51, hidden_size=64, num_layers=2)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 16
    features = 51
    dummy_input = torch.randn(batch_size, seq_len, features)
    
    # Export to ONNX
    output_path = "model/exported/fall_detection.onnx"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Test model exported to: {output_path}")
    
    # Test the model
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    
    # Run test inference
    test_input = torch.randn(1, 16, 51).numpy()
    outputs = session.run(None, {'input': test_input})
    
    print(f"✓ Model test passed - output shape: {outputs[0].shape}")
    print(f"✓ Sample output: {outputs[0][0][0]:.4f}")
    
    return output_path

if __name__ == "__main__":
    model_path = generate_test_model()
    print(f"\nDemo model ready at: {model_path}")
