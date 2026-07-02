## 2026-07-02 - Real-time Fall Detection Alerts Accessibility
**Learning:** Real-time safety alert logs for fall detections must interrupt the screen reader immediately because they represent critical health/safety emergencies.
**Action:** Use `aria-live='assertive'` rather than `polite` on real-time safety alert lists.
