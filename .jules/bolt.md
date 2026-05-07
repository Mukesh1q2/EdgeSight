## 2024-05-19 - Fast Timestamping in High-Frequency Loops
**Learning:** In the 30Hz `detection_loop` and 10Hz `websocket_endpoint`, using `datetime.strptime()` for cooldown logic and `datetime.now().strftime()` for JSON timestamps introduced massive string parsing overhead.
**Action:** Replace `strptime()` logic with float comparisons (`time.time()`) for cooldowns, and generate timestamps using `time.strftime` (via `localtime`) when strings are strictly necessary for payloads.
