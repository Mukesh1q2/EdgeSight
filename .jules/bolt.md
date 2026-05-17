## 2024-05-16 - Replace expensive datetime strptime with float timestamps
**Learning:** Parsing timestamps using `datetime.strptime()` inside high-frequency real-time loops like `detection_loop` adds unnecessary string parsing overhead, especially when simple rate limiting is all that is needed.
**Action:** Use fast float timestamps via `time.time()` for checking cooldowns and intervals instead of calculating time deltas with `datetime.strptime` and `datetime.now()`.
