## 2024-06-16 - Dynamic Alert Accessibility
**Learning:** Screen readers miss dynamically added alerts in real-time interfaces unless explicitly configured with ARIA live regions.
**Action:** Always add aria-live="assertive" and role="log" to dynamic alert lists like #alert-list.