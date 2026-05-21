## 2026-05-21 - Custom Controls Missing Focus States
**Learning:** Custom interactive UI controls (like sliders and styled buttons) with `outline: none` lack native focus styles and break keyboard navigation accessibility.
**Action:** Always implement explicit `:focus-visible` outlines using design tokens (e.g., `var(--primary)`) when styling custom controls to ensure proper keyboard accessibility without affecting mouse users.
