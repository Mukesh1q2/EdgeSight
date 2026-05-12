## 2024-05-15 - Initial Bolt Journal\n**Learning:** Started tracking performance optimizations.\n**Action:** Keep documenting critical learnings.
## 2024-05-15 - Optimize alert cooldown tracking
**Learning:** Found a performance bottleneck in the high-frequency detection loop caused by repeatedly parsing and formatting timestamps using `datetime.strptime` and `strftime`. Even though the code path is only hit when the fall probability exceeds the threshold, this can cause significant delays.
**Action:** Use `time.time()` for rate-limiting and state tracking instead of string manipulation for much better performance.
