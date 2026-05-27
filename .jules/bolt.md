## 2024-05-27 - Datetime String Parsing Bottleneck in Real-Time Loops
**Learning:** Using `datetime.strptime()` with time-only formats (e.g., "%H:%M:%S") in a high-frequency real-time loop is an expensive operation due to string parsing. Additionally, parsing time-only strings without a date defaults the date to 1900-01-01, leading to incorrect calculations and potential bugs when comparing against `datetime.now()`.
**Action:** Always use `time.time()` float representations for tracking durations, cooldowns, and rate limiting in high-frequency loops instead of string formatting and parsing.
