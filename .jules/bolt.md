## 2024-05-24 - [Avoid datetime parsing in high-frequency loops]
**Learning:** Parsing datetime strings with `datetime.strptime` inside high-frequency loops (like a 30 FPS video detection loop) is significantly slower than using float timestamps (`time.time()`).
**Action:** Replace `strptime` and string timestamps with `time.time()` for tracking state/cooldowns to optimize performance.
