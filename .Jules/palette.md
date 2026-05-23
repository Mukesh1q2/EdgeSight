## 2026-05-23 - Add keyboard focus-visible styles
**Learning:** Custom interactive UI controls (like sliders and styled buttons) in this project lack native focus styles.
**Action:** Always implement `:focus-visible` outlines using existing design tokens (e.g., `var(--primary)`) to ensure proper keyboard accessibility without disrupting the visual design for mouse users.