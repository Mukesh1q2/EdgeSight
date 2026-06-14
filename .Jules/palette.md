## 2024-06-14 - Add ARIA live regions to dynamic alert logs
**Learning:** Dynamically updated UI elements, such as the real-time `#alert-list` in `web/index.html`, do not notify screen readers when new items are added by default.
**Action:** Use `aria-live='assertive'` and `role='log'` on containers that frequently receive important dynamic updates (like real-time alerts) to ensure screen readers announce new items to visually impaired users.
