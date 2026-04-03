## 2024-05-24 - Avoid string parsing in high-frequency CV loops
**Learning:** The detection loop runs at ~30 FPS, making computationally expensive operations like `datetime.strptime()` surprisingly costly over time when evaluating state (such as alert throttling).
**Action:** Always use native data types (like storing a `datetime` object) and evaluate state changes via delta operations (`(now - last).total_seconds()`) to avoid parsing overhead in hot loops.
