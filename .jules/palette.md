## 2024-07-06 - App-specific aria-live pattern
**Learning:** Real-time safety alert logs (such as `#alert-list` in `web/index.html` for fall detections) represent critical health/safety emergencies that must interrupt the screen reader immediately, thus must use `aria-live='assertive'` rather than `polite`.
**Action:** Always use `aria-live='assertive'` for critical emergency logs in this app.
