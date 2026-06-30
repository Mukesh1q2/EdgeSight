## 2024-05-24 - [Critical Alert Logging]
**Learning:** Real-time safety alert logs (such as fall detections) represent critical health/safety emergencies and must use `aria-live='assertive'` rather than `polite` to interrupt screen readers immediately.
**Action:** Always use `aria-live='assertive'` for life-critical alerts in this system.
