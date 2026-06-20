## 2024-05-20 - Adding ARIA live regions for critical dynamic alerts
**Learning:** Dynamically updated UI elements in safety-critical applications, such as the real-time `#alert-list` in `web/index.html`, require `aria-live='assertive'` and `role='log'` to ensure screen readers immediately announce new items and interrupt other speech if necessary.
**Action:** Always add `aria-live='assertive'` and `role='log'` to containers that receive critical, time-sensitive dynamic updates (like fall detection alerts) to ensure accessibility for screen reader users.
