## 2026-06-17 - Dynamic Alert Accessibility
**Learning:** Dynamically updated UI elements like real-time alert logs require ARIA live regions for screen readers to announce new items as they appear.
**Action:** Always add aria-live='assertive' and role='log' to dynamic lists/logs that update via JavaScript.
