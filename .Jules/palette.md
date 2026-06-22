## 2026-06-22 - Added ARIA live region to alert log
**Learning:** Dynamically updated elements like the real-time alert log require ARIA live regions for screen readers to announce new items.
**Action:** Always use `aria-live='assertive'` and `role='log'` for dynamic alert lists.
