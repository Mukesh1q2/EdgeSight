## 2024-03-20 - Expensive String Parsing in High-Frequency Loops
**Learning:** Parsing and formatting datetime strings using `strptime` and `strftime` inside a high-frequency loop (like a 30 FPS video processing loop) is a significant performance bottleneck. In this codebase, the alert cooldown was computing string parsing on every frame where a fall was detected.
**Action:** Always use primitive types like floats (`time.time()`) for high-frequency operations such as rate-limiting, debouncing, or cooldown tracking, to avoid the overhead of string manipulation.
