
## 2024-05-18 - Real-Time Dashboard Screen Reader Optimization
**Learning:** In a real-time dashboard environment with telemetry updates, applying `aria-live` to everything overwhelming for screen reader users. High-frequency elements (like a 10Hz gauge updating) should not have it, whereas low-frequency, important changes (like discrete fall alerts and risk level badges) must use `aria-live="polite"` to be announced without interrupting the user unnecessarily.
**Action:** Always selectively apply `aria-live="polite"` only to critical, low-frequency event streams (e.g., alert logs, risk badges), and omit it from continuous real-time numerical streams.
