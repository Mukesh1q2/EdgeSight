## 2024-05-24 - Accessibility improvements for dashboard
**Learning:** High-frequency updates (like FPS/Latency) should not use `aria-live`, but low-frequency alerts (Alert Log, Risk Badge) need `aria-live="polite"`. Decorative icons inside text buttons need `aria-hidden="true"` to prevent redundant screen reader announcements.
**Action:** Always use `aria-hidden="true"` on decorative visual icons inside actionable text elements, and selectively apply `aria-live` only to meaningful low-frequency status changes.
