## 2024-07-07 - Critical Health Alerts Require Assertive ARIA Live
**Learning:** Real-time safety alert logs for fall detections must use `aria-live='assertive'` rather than `polite`. Because they represent critical health/safety emergencies, they must interrupt the screen reader immediately to ensure the user is notified without delay.
**Action:** Always apply `aria-live='assertive'` to real-time, life-safety alert containers in this application.
