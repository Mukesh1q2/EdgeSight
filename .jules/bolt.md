## 2024-05-24 - Time-Based Rate Limiting
**Learning:** Using `datetime.strptime` with time-only formats defaults to 1900-01-01 and is slow.
**Action:** Always use `time.time()` for safe and performant rate limiting in high-frequency loops.
