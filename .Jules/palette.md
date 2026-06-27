## 2024-05-20 - Safety Alerts Accessibility
**Learning:** Real-time safety alert logs (such as `#alert-list` in `web/index.html` for fall detections) must use `aria-live='assertive'` rather than `polite` because they represent critical health/safety emergencies that must interrupt the screen reader immediately.
**Action:** Use `aria-live='assertive'` for critical emergency logs in the future.
