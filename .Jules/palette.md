## 2024-07-01 - [Real-time Safety Alert Logging]
**Learning:** Real-time safety alert logs for critical health/safety emergencies (like fall detections) must use `aria-live='assertive'` rather than `polite` to interrupt the screen reader immediately.
**Action:** Always use `aria-live='assertive'` on alert lists that convey critical safety events.