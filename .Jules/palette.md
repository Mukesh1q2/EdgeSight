## 2024-04-12 - Real-time ARIA Announcers
**Learning:** For low-frequency real-time updates like alert logs or status changes, `aria-live="polite"` should be used. However, it should NOT be applied to high-frequency updates like telemetry gauges or charts, as it would severely overwhelm screen readers.
**Action:** When adding accessibility to real-time dashboards, explicitly partition low-frequency critical updates (add `aria-live`) from high-frequency metric updates (hide or simplify for screen readers).
