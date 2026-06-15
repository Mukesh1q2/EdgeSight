## 2024-06-15 - Dynamic UI Elements and Screen Readers
**Learning:** Dynamically updated UI elements, such as the real-time `#alert-list` in `web/index.html`, need explicit ARIA attributes for screen readers to announce new items. Without them, users relying on assistive technology miss critical updates.
**Action:** Use `aria-live='assertive'` and `role='log'` on dynamic alert lists to ensure screen readers immediately announce new items as they are added.
