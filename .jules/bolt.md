## 2024-05-10 - Optimize datetime parsing in high-frequency detection loop
**Learning:** In high-frequency loops (like video frame processing at 30+ FPS), parsing strings to datetime objects using `datetime.strptime()` for simple time delta calculations creates a significant and unnecessary CPU overhead.
**Action:** Use float timestamps (e.g., `time.time()`) for rate-limiting, cooldowns, and state tracking within hot loops, converting to formatted strings only when data is being presented or serialized for external consumption.
