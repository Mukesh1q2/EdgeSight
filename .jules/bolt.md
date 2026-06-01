## 2024-06-01 - Avoid parsing datetimes in high-frequency loops
**Learning:** Using `datetime.strptime()` in a high-frequency loop (like a 30 FPS video detection loop) to calculate time differences is extremely slow and causes unnecessary CPU overhead. In our benchmark, `time.time()` was over 300x faster than formatting and parsing time strings.
**Action:** Always use float timestamps (`time.time()`) for tracking durations and rate-limiting instead of converting between strings and datetimes. Only format times to strings when preparing data for API output or display.
