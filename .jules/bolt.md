## 2024-05-21 - Avoiding String Parsing in High-Frequency Loops
**Learning:** Parsing string timestamps with `datetime.strptime` inside a high-frequency loop (like a video frame detection loop) is surprisingly expensive and a CPU bottleneck. In `fastapi_server.py`, calculating the cooldown of 2 seconds by parsing the string time of the last alert caused unnecessary overhead.
**Action:** Always prefer using float timestamps (like `time.time()`) for rate-limiting, cooldown tracking, and state management in hot paths, and only format to strings when preparing data for API responses or display.
