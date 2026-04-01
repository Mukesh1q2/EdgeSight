# Palette's UX Journal

## 2024-05-24 - Accessible Real-time Data Overlays
**Learning:** High-frequency, real-time UI components (like telemetry gauges, risk badges, or live logs) are often entirely invisible to screen readers without explicit ARIA live regions. Users simply miss the dynamic updates if they are not announced.
**Action:** Always add `aria-live="polite"` (or "assertive" for critical alerts) alongside `aria-atomic="true"` on the parent containers of dynamic real-time data elements so screen readers can gracefully announce the changing state. For log lists, use `aria-relevant="additions"` to speak out newly inserted items.
