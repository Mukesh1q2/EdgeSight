## 2025-01-28 - Avoid Expensive String/Datetime Operations in Hot Loops
**Learning:** Parsing datetimes (`datetime.strptime`) and formatting them (`datetime.now().strftime`) inside high-frequency loops (e.g., 30 FPS video processing) causes unnecessary CPU overhead, especially when parsing history arrays to calculate time deltas for throttling.
**Action:** Use primitive floats like `time.time()` for all latency, throttling, and duration checks in the CV hot loops. Defer expensive string formatting (like `strftime`) strictly to the point right before returning/logging the data.
