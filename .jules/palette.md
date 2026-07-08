## 2026-07-08 - Critical Accessibility for Safety Alerts
**Learning:** Real-time safety alert logs (such as fall detections in `#alert-list`) must use `aria-live='assertive'` rather than `polite`. These represent critical health/safety emergencies that must interrupt the screen reader immediately.
**Action:** Always use `aria-live='assertive'` for critical emergency logs in this application instead of standard `polite` announcements.
