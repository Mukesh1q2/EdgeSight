## 2024-04-14 - Real-time ARIA Updates
**Learning:** Adding `aria-live` to high-frequency components (like 10Hz telemetry charts or gauges) can overwhelm screen readers, rendering the interface unusable. It must be isolated.
**Action:** Only apply `aria-live="polite"` to low-frequency, event-driven dynamic UI components (like Alert Logs or Risk Badges), while leaving high-frequency updates silent.
