## 2026-03-29 - Real-time Telemetry Accessibility
**Learning:** Real-time telemetry dashboards (like updating charts and alert logs) require `aria-live` regions so screen readers can announce new data or risks to visually impaired users.
**Action:** Always add `aria-live="polite"` or `aria-live="assertive"` to containers that dynamically update with critical information.
