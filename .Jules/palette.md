## 2024-06-25 - Safety-Critical Alert Announcements
**Learning:** Real-time safety alert logs require `aria-live='assertive'` instead of `polite` because fall detection events are critical emergencies that must interrupt the screen reader immediately.
**Action:** Always use `aria-live='assertive'` and `role='log'` for health/safety-critical event streams in this application.