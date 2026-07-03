## 2026-07-03 - Critical Safety Alerts Accessibility
**Learning:** Real-time safety alert logs (e.g., fall detections) represent critical health/safety emergencies that must interrupt the screen reader immediately.
**Action:** Always use `aria-live="assertive"` rather than `polite` for life-critical or safety-related real-time UI updates.
