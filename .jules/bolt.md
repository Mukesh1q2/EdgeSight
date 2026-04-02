## 2026-04-02 - String Parsing in High-Frequency Loops
**Learning:** Parsing strings back to datetimes using `datetime.strptime()` inside a 30 FPS computer vision hot loop introduces significant CPU overhead and processing latency.
**Action:** Always maintain native data types (e.g., `datetime` objects or timestamps) in state objects for performance-critical timing checks rather than parsing serialized string representations repeatedly.
