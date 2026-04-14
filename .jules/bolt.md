## 2024-05-24 - Avoid datetime.strptime in hot loop
**Learning:** `datetime.strptime()` is computationally expensive and should not be used inside high-frequency hot loops (like a 30 FPS video detection loop), as it creates unnecessary overhead compared to native data types or simple pre-calculated comparisons.
**Action:** Replace `datetime.strptime()` with a simple float timestamp (e.g. `time.time()`) to track delta times between events in high-frequency loops.
