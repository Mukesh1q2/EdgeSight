## 2024-05-23 - High-frequency Loop Time Checking
**Learning:** In Python, tracking time intervals using `time.time()` float subtraction in high-frequency loops (like video stream detection loops) is orders of magnitude faster (e.g. ~100x) than parsing and formatting strings with `datetime.strptime` and `strftime`.
**Action:** Use float timestamps via `time.time()` or `time.perf_counter()` for internal state tracking and rate-limiting, and only convert to string representations (via `strftime`) when the data is finally needed for API responses or UI rendering.
