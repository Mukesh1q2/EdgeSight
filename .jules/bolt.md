## 2024-05-18 - [Optimize datetime parsing in detection loop]
**Learning:** In a high-frequency event loop (like FastAPI's `detection_loop`), using `datetime.strptime` and `strftime` to compare times for rate-limiting is significantly slower (~189ms vs ~3ms for 10k iterations) and can introduce subtle logic bugs (e.g., date rollovers when parsing time-only strings like "%H:%M:%S", which defaults to 1900-01-01).
**Action:** Always prefer `time.time()` float subtraction for rate-limiting or time tracking in high-frequency loops instead of string-based datetime parsing.
