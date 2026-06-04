## 2024-06-04 - Float timestamps over strptime
**Learning:** In Python, parsing time strings in high-frequency loops (like video detection streams) using `datetime.strptime()` and `.strftime()` is significantly slower (~60x) and prone to latent logic bugs (e.g. date defaulting to 1900-01-01 when only parsing time) compared to using float timestamps via `time.time()`.
**Action:** Always use float timestamps (`time.time()`) for rate-limiting, cooldown tracking, and state comparisons instead of expensive string parsing. Format dates to strings only when necessary for presentation/API responses.
