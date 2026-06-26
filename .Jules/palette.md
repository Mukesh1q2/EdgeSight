## 2024-06-26 - Critical Alert Accessibility
**Learning:** Real-time safety alert logs (such as `#alert-list` in `web/index.html` for fall detections) represent critical health/safety emergencies. They must use `aria-live='assertive'` rather than `polite` because they must interrupt the screen reader immediately.
**Action:** Always use `aria-live='assertive'` for emergency health/safety updates, not `polite`.
