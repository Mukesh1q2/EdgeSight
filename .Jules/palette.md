## 2026-05-17 - Custom Interactive UI Controls Focus Styles
**Learning:** Custom interactive UI controls (like sliders and styled buttons) lack native focus styles and must implement `:focus-visible` outlines to ensure proper keyboard accessibility.
**Action:** Always add `:focus-visible` states utilizing existing design tokens (e.g., `--primary`) when styling custom controls to maintain visibility for keyboard users without affecting mouse users.
