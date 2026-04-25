## 2025-04-25 - Avoid String Parsing in Hot Loops
**Learning:** Using `datetime.strptime()` for simple rate limiting inside a high-frequency (30 FPS) loop is an unnecessary performance bottleneck. Evaluating string dates every frame consumes significant CPU time compared to simple arithmetic.
**Action:** Use native floating point timestamps (e.g., `time.time()`) and simple difference comparisons for cooldown and rate limiting logic, deferring string formatting operations until the limit has actually been triggered.
