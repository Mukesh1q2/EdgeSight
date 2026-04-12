## 2024-06-25 - Avoid String Ops in High-Frequency Loops
**Learning:** Parsing timestamps inside a 30 FPS hot loop using `datetime.strptime()` is a significant performance anti-pattern, causing high CPU overhead. String formatting (`datetime.strftime()`) is also expensive when executed on every iteration even when its output is discarded by a throttling check.
**Action:** Always use native numeric types (e.g., `time.time()` float timestamps) for threshold or cooldown logic, and defer computationally expensive formatting operations until after all conditions and rate limit checks have passed.
