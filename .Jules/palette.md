## 2024-06-01 - Keyboard Nav & Dynamic List Announcement
**Learning:** Real-time log lists like `#alert-list` require `aria-live="assertive"` and `role="log"` to ensure screen readers immediately announce dynamically appended elements. In Aegis Vision design system, interactive elements lack default keyboard focus visibility.
**Action:** Always verify `:focus-visible` states are explicitly defined for interactive elements (like `.btn`, `.slider`, `select`) in `.css` when adding new UI widgets, and ensure real-time dynamic lists utilize appropriate `aria-live` regions.
