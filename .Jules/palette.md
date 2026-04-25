## 2026-04-25 - ARIA Live Region Performance
**Learning:** Real-time UI components with low-frequency updates (like alert logs and risk badges) must implement aria-live regions (e.g., aria-live="polite") to ensure they are announced by screen readers. Avoid applying aria-live to high-frequency updates (like 10Hz telemetry charts or gauges) as it overwhelms screen readers.
**Action:** Apply `aria-live="polite"` selectively to meaningful state changes (alerts, risk level shifts) and explicitly omit it from high-frequency metric displays.
