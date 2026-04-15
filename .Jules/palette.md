
## 2024-05-18 - Optimizing Screen Reader Flow for High-Frequency Dashboards
**Learning:** High-frequency, real-time dashboards can easily overwhelm screen readers. In EdgeSight, applying `aria-live="polite"` indiscriminately to 10Hz telemetry components causes chaotic announcements. It is crucial to restrict `aria-live` to low-frequency, actionable updates (like alert logs and risk level changes) while avoiding it for rapid numeric updates (like FPS, latency, and the primary probability gauge).
**Action:** When adding accessibility to real-time telemetry dashboards, isolate low-frequency state changes for `aria-live` and ensure decorative UI text/icons (like ▶, ⏹, 👁️) have `aria-hidden="true"` to prevent redundant "play start", "stop stop" announcements.
