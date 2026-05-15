## 2026-05-15 - Focus Visibility on Custom Controls
**Learning:** Custom interactive UI controls (like sliders and styled buttons) lack native focus styles and must implement `:focus-visible` outlines to ensure proper keyboard accessibility.
**Action:** Always add `:focus-visible` outlines using existing design tokens (e.g. `--primary`) when creating or modifying custom interactive elements.
