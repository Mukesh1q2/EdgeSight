"""
Unit tests for Python ONNX Runtime inference baseline.

Tests the `PythonFallDetector` class from `inference/python_infer.py`.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import FallNet
from inference.python_infer import PythonFallDetector


@pytest.fixture(scope="module")
def mock_onnx_model():
    """Create a temporary mock ONNX model for testing."""
    # Create a simple untrained model
    model = FallNet()
    model.eval()
    
    # Export to a temporary file
    temp_dir = tempfile.mkdtemp()
    onnx_path = os.path.join(temp_dir, "test_model.onnx")
    
    dummy_input = torch.randn(1, 30, 51)
    
    # Simple export without dynamic axes just for testing structure
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        opset_version=17,
        input_names=["pose_sequence"],
        output_names=["fall_probability"],
        dynamic_axes={"pose_sequence": {0: "batch_size"}, "fall_probability": {0: "batch_size"}}
    )
    
    yield onnx_path
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    os.rmdir(temp_dir)


def test_python_infer_init(mock_onnx_model):
    """Test initialization of PythonFallDetector."""
    detector = PythonFallDetector(mock_onnx_model)
    assert detector is not None
    assert detector.input_name == "pose_sequence"
    assert detector.output_name == "fall_probability"


def test_python_infer_predict(mock_onnx_model):
    """Test single sequence inference."""
    detector = PythonFallDetector(mock_onnx_model)
    
    # Create random sequence
    sequence = np.random.randn(30, 51).tolist()
    
    # Predict
    prob = detector.predict(sequence)
    
    # Assert output is valid probability
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    
    # Assert latency is populated
    assert detector.get_last_latency_ms() > 0.0


def test_python_infer_predict_batch(mock_onnx_model):
    """Test batch inference."""
    detector = PythonFallDetector(mock_onnx_model)
    
    batch_size = 4
    sequences = [np.random.randn(30, 51).tolist() for _ in range(batch_size)]
    
    probs = detector.predict_batch(sequences)
    
    assert isinstance(probs, list)
    assert len(probs) == batch_size
    for prob in probs:
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0


def test_python_infer_invalid_model_path():
    """Test that valid exception is raised for invalid model path."""
    with pytest.raises(FileNotFoundError):
        PythonFallDetector("non_existent_model.onnx")
