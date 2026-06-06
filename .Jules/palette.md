## 2024-05-16 - Dynamic Alert Screen Reader Announcement
**Learning:** Dynamically updated UI elements like the real-time alert log in the dashboard need `aria-live` attributes. Without it, screen reader users miss critical fall detection alerts.
**Action:** Always add `aria-live='assertive'` and `role='log'` to any container that acts as a real-time append-only list of critical events.
