## 2024-05-01 - Optimizing Screen Reader Announcements

**Learning:** When dealing with real-time telemetry dashboards (like EdgeSight), applying `aria-live` regions indiscriminately causes screen reader overload. High-frequency updates (e.g., 10Hz charts or gauges) overwhelm the reader.
**Action:** Only apply `aria-live="polite"` to low-frequency, critical state changes, such as the risk badge (`#risk-badge`) and the alert log (`#alert-list`). Decorative UI icons in buttons should also use `aria-hidden="true"` to prevent redundant reading.
