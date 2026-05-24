## 2024-05-24 - Optimizing Rate-limiting with Time
**Learning:** Checking timestamps with `datetime.now()` and string formatting/parsing (`strptime`) in high-frequency loops (like video frame rate-limiting) is significantly slower (up to 60x) than using `time.time()` float calculations.
**Action:** Always prefer float timestamps (`time.time()`) over `datetime` string conversions for high-frequency internal rate-limiting or cooldown checks.
