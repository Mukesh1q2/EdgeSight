## 2024-03-27 - Remove datetime.strptime bottleneck in real-time loop
**Learning:** In a high-frequency loop (like a 30 FPS video feed detection loop), parsing strings into datetime objects using `datetime.strptime()` every frame is a massive performance drain. It creates unnecessary object instantiations, string formatting, and parsing CPU overhead.
**Action:** When tracking time between events or for debouncing operations inside tight loops, always store and compare native `datetime` or `time.time()` objects directly, and format strings only when returning data or logging.
