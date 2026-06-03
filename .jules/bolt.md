## 2024-05-24 - High-Frequency Loop Bottlenecks
**Learning:** String parsing (`datetime.strptime`) inside hot loops (like a ~30FPS video inference loop) creates significant CPU overhead in Python. Using `strptime` for a simple cooldown check was taking >1s for 100k iterations compared to 0.02s with float timestamps.
**Action:** Always prefer `time.time()` epoch comparisons over `datetime` string conversions for rate-limiting, cooldowns, and performance tracking in high-frequency loops.
