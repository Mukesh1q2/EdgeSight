## 2024-05-24 - [Alert Log Accessibility]
**Learning:** Real-time safety alert logs for critical health/safety emergencies (like fall detections) must use `aria-live='assertive'` rather than `polite` to ensure they interrupt the screen reader immediately, prioritizing user safety over uninterrupted reading.
**Action:** Always use `aria-live='assertive'` for life-safety critical alerts and notifications.
