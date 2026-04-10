## 2024-04-10 - Avoid String Parsing in High-Frequency Loops
**Learning:** Avoid using computationally expensive string operations like `datetime.strptime()` and `datetime.strftime()` inside high-frequency computer vision loops (e.g. 30 FPS). It causes unnecessary CPU load and latency.
**Action:** Use native data types (like float timestamps from `time.time()`) and pre-calculated deltas for time tracking, and defer string formatting operations until they are strictly necessary (e.g. only after a conditional rate limit check has passed).
