## 2026-07-05 - Alert Log Accessibility
**Learning:** Real-time safety alert logs (such as `#alert-list` in `web/index.html` for fall detections) represent critical health/safety emergencies. They must use `aria-live='assertive'` rather than `polite` because they must interrupt the screen reader immediately.
**Action:** Always use `aria-live='assertive'` for critical emergency logs.
