## 2024-05-15 - Optimizing High-Frequency Loops
**Learning:** Using `time.time()` (float) instead of `datetime.now().strftime()` and `datetime.strptime()` for rate limiting inside the high-frequency detection loop is ~40x faster.
**Action:** Always use float timestamps for rate limiting and intervals instead of string parsing/formatting in high-frequency paths.
