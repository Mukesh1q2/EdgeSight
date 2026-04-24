## 2026-04-24 - [Selective aria-live Usage]
**Learning:** In high-frequency real-time dashboards (like 30fps telemetry updates), applying `aria-live` globally overwhelms screen readers. It must be applied selectively only to low-frequency, critical updates (like alert logs and risk level changes).
**Action:** Apply `aria-live="polite"` to isolated containers that hold critical, infrequent text updates, ensuring screen reader users get meaningful alerts without telemetry spam.
