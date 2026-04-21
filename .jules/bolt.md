## 2024-05-24 - Avoid `datetime.strptime` in High-Frequency Loops
**Learning:** Parsing strings with `datetime.strptime()` inside an asynchronous loop running at 30 FPS introduces significant unnecessary overhead. In this codebase, it was being used just to calculate a 2-second cooldown on alerts.
**Action:** Use fast, native float arithmetic (e.g., `time.time()`) for tracking durations and rate limits in the hot path. Defer expensive string formatting (like `strftime`) until *after* all rate limit or conditional checks have passed.
