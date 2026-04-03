import sys
from unittest.mock import MagicMock, patch

# Mock all the missing dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['torch'] = MagicMock()

import pytest
sys.exit(pytest.main(["-v"]))
