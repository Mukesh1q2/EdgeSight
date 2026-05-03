## 2024-05-24 - Throttling Expensive String Operations in Hot Loops
**Learning:** In Python, calling `datetime.strptime()` inside a hot video processing loop (30 FPS) introduces unnecessary string parsing overhead when used for rate-limiting.
**Action:** Use `time.time()` (float comparisons) for rate-limiting logic, and defer string formatting (like timestamp generation) until *after* the rate-limit condition has been met.
