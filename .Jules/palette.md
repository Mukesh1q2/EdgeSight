## 2024-05-24 - Real-time Alert List Accessibility
**Learning:** Dynamically updated UI elements, such as the real-time `#alert-list`, require `aria-live='assertive'` and `role='log'` to ensure screen readers announce new items immediately as they are added.
**Action:** Always apply `aria-live` and an appropriate `role` (like `log` or `status`) to containers that receive real-time streaming updates.
