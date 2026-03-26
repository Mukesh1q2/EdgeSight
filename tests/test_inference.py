"""
Unit Tests for EdgeSight - Inference Module

Run with: pytest tests/test_inference.py -v
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPythonInference:
    """Tests for Python ONNX inference."""

    def test_onnx_runtime_available(self):
        """Test that ONNX Runtime is available."""
        try:
            import onnxruntime as ort
            assert ort is not None
        except ImportError:
            pytest.skip("ONNX Runtime not installed")

    def test_model_file_exists(self):
        """Test that model file exists."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated - run scripts/generate_test_model.py")
        assert model_path.exists()

    def test_inference_session_creation(self):
        """Test that inference session can be created."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        assert session is not None

    def test_inference_input_shape(self):
        """Test inference with correct input shape."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        
        # Get input shape
        input_info = session.get_inputs()[0]
        assert input_info.shape == [1, 16, 51] or 'batch' in str(input_info.shape)

    def test_inference_output_shape(self):
        """Test inference output shape."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        
        # Create input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 16, 51).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 1)

    def test_inference_output_range(self):
        """Test that inference output is in [0, 1]."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 16, 51).astype(np.float32)
        
        outputs = session.run(None, {input_name: dummy_input})
        
        output_value = outputs[0][0][0]
        assert 0.0 <= output_value <= 1.0

    def test_inference_batch_processing(self):
        """Test inference with batch of inputs."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        
        input_name = session.get_inputs()[0].name
        
        # Run multiple inferences
        for _ in range(10):
            dummy_input = np.random.randn(1, 16, 51).astype(np.float32)
            outputs = session.run(None, {input_name: dummy_input})
            assert outputs[0].shape == (1, 1)

    def test_inference_latency(self):
        """Test inference latency is reasonable."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        import onnxruntime as ort
        import time
        session = ort.InferenceSession(str(model_path))
        
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(1, 16, 51).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Measure latency
        start = time.time()
        for _ in range(100):
            session.run(None, {input_name: dummy_input})
        elapsed = time.time() - start
        
        avg_latency_ms = (elapsed / 100) * 1000
        assert avg_latency_ms < 100  # Should be under 100ms


class TestPythonFallDetector:
    """Tests for PythonFallDetector class."""

    def test_detector_creation(self):
        """Test PythonFallDetector can be created."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        from inference.python_infer import PythonFallDetector
        detector = PythonFallDetector(str(model_path))
        assert detector is not None

    def test_detector_predict(self):
        """Test detector predict method."""
        model_path = Path("model/exported/fall_detection.onnx")
        if not model_path.exists():
            pytest.skip("Model not generated")
        
        from inference.python_infer import PythonFallDetector
        detector = PythonFallDetector(str(model_path))
        
        # Create sequence
        sequence = np.random.randn(16, 51).astype(np.float32)
        
        result = detector.predict(sequence)
        assert 0.0 <= result <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])