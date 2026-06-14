## 2024-06-14 - Python strptime is too slow for high-frequency loops
**Learning:** Using `datetime.strptime()` with time-only formats (e.g., "%H:%M:%S") defaults the date to 1900-01-01, making timedelta calculations against `datetime.now()` incorrect if crossing midnight, and it is extremely slow (~140x slower) compared to float timestamps for high-frequency operations.
**Action:** Always use float timestamps (`time.time()`) for rate-limiting, time difference calculations, and state tracking in performance-critical loops instead of expensive string parsing.
