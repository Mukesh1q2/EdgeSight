## 2026-06-29 - [Alert List Accessibility]
**Learning:** Real-time safety alert logs (such as `#alert-list` in `web/index.html` for fall detections) must use `aria-live='assertive'` rather than `polite` because they represent critical health/safety emergencies that must interrupt the screen reader immediately.
**Action:** Always use `aria-live="assertive"` for emergency/critical alerts.
