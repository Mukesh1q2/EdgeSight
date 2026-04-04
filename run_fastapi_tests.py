import sys
from unittest.mock import MagicMock
sys.modules['mediapipe'] = MagicMock()

import fastapi_server

# Verify the changes in DetectionState
state = fastapi_server.DetectionState()
assert hasattr(state, 'last_alert_time')
assert state.last_alert_time == 0.0
print("Verified DetectionState changes.")
