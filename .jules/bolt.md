## 2024-05-24 - Hot Loop Alert Throttling
**Learning:** In the high-frequency (30 FPS) `detection_loop`, computationally expensive string operations like `datetime.strptime()` caused unnecessary overhead when validating alert cooldown periods.
**Action:** Avoid string parsing in hot loops. Store tracking state as native data types (e.g., storing the raw `datetime` object in `last_alert_time`) to compute deltas efficiently, keeping formatting strictly for presentation/API payloads.
